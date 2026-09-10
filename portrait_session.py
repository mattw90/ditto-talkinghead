"""Reusable PyTorch/MLX portrait renderer with incremental audio and frame output.

Uses the released offline motion checkpoint with its original overlapping windows.
Only finalized motion is emitted; streaming does not change the checkpoint or
pretend that its 80-frame lookahead has disappeared. Input is 24 kHz mono PCM16.
"""
import hashlib
import math
from pathlib import Path
import pickle
import tempfile
import time

import cv2
import numpy as np
import soxr
import torch

from stream_pipeline_offline import StreamSDK


class PortraitSession:
    def __init__(self, models, device='mps', *, tensor_path=True, fast_mask=True,
                 render_batch_size=4, prepare_decoder=True, render_backend='torch'):
        if render_batch_size not in (1, 2, 4):
            raise ValueError('Render batch size must be 1, 2, or 4')
        if render_backend not in ('torch', 'mlx'):
            raise ValueError('Render backend must be torch or mlx')
        if render_backend == 'mlx' and device != 'mps':
            raise ValueError('The MLX backend requires Apple Silicon')
        self.render_backend = render_backend
        started = time.perf_counter()
        models = Path(models)
        torch.set_num_threads(4)
        torch.manual_seed(42)
        np.random.seed(42)
        if device == 'mps':
            torch.mps.set_per_process_memory_fraction(.5)
        # Only load a trusted publisher config from the server-owned model directory.
        with (models / 'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl').open('rb') as source:
            config = pickle.load(source)
        for name, component in config['base_cfg'].items():
            component['device'] = device if name in {
                'appearance_extractor_cfg', 'motion_extractor_cfg', 'stitch_network_cfg',
                'warp_network_cfg', 'decoder_cfg'} else 'cpu'
            if render_backend == 'mlx' and name in ('warp_network_cfg', 'decoder_cfg'):
                component['device'] = 'cpu'
        config['audio2motion_cfg']['device'] = device
        config['base_cfg']['landmark478_cfg'] = {
            'device': 'cpu', 'force_ori_type': False,
            'blaze_face_model_path': str((models / 'ditto_onnx/blaze_face.onnx').resolve()),
            'face_mesh_model_path': str((models / 'ditto_onnx/face_mesh.onnx').resolve()),
        }
        with tempfile.TemporaryDirectory(prefix='ditto-config-') as directory:
            path = Path(directory) / 'config.pkl'
            path.write_bytes(pickle.dumps(config))
            self.sdk = StreamSDK(str(path), str(models / 'ditto_pytorch'))
        self.sdk.audio2motion.lmdm.model.device = device
        self.sdk.warp_f3d.warp_net.model.dense_motion_network.use_mps_2d_mask = fast_mask and device == 'mps'
        if prepare_decoder and render_backend == 'torch':
            with torch.no_grad(), torch.autocast(device_type=device[:4], dtype=torch.float16):
                self.sdk.decode_f3d.decoder.model.prepare_inference()
            if device == 'mps':
                # Convolution already runs in fp16 under autocast. Keep its fixed
                # weights in that precision instead of recasting every batch.
                self.sdk.decode_f3d.decoder.model.half()
        if render_backend == 'mlx':
            from core.models.mlx_renderer import MlxFrameRenderer
            self.mlx_renderer = MlxFrameRenderer(models)
            # One image-rendering backbone. Ditto's small motion/registration
            # networks continue to use the existing PyTorch implementation.
            del self.sdk.warp_f3d, self.sdk.decode_f3d
        self.render_batch_size = render_batch_size
        self.device, self.tensor_path = device, tensor_path
        self.source_key = self.source = self.feature = None
        self.load_seconds = time.perf_counter() - started
        self.cpu_rng = torch.get_rng_state()
        self.device_rng = torch.mps.get_rng_state() if device == 'mps' else None
        self.metrics = {}

    def prepare(self, portrait):
        key = hashlib.sha256(Path(portrait).read_bytes()).hexdigest()
        cached = key == self.source_key
        if not cached:
            self.source = self.sdk.avatar_registrar(str(portrait), max_dim=512, n_frames=1,
                                                   crop_scale=2.3, crop_vx_ratio=0, crop_vy_ratio=-.125)
            feature = self.source['f_s_lst'][0]
            if self.render_backend == 'mlx':
                self.mlx_renderer.prepare(feature)
                self.feature = None
            else:
                self.feature = torch.as_tensor(feature, device=self.device) if self.tensor_path else feature
            self.source_key = key
        return cached

    def frames(self, portrait, audio_chunks, *, sampling_steps=50, motion_profile='original'):
        if sampling_steps not in (10, 20, 50):
            raise ValueError('Supported sampling steps: 10, 20, 50')
        if motion_profile not in ('original', 'bounded'):
            raise ValueError('Supported motion profiles: original, bounded')
        started = time.perf_counter()
        cached = self.prepare(portrait)
        prepared = time.perf_counter()
        sdk, source = self.sdk, self.source
        # Reusing the process must not reuse the preceding turn's noise/history.
        torch.set_rng_state(self.cpu_rng)
        if self.device_rng is not None:
            torch.mps.set_rng_state(self.device_rng)
        np.random.seed(42)
        sdk.audio2motion.lmdm.model.sampling_timesteps = None
        sdk.condition_handler.setup(source, emo=4)
        sdk.audio2motion.setup(sdk.condition_handler.x_s_info_0,
                               sampling_timesteps=sampling_steps, online_mode=False)
        source_info = source['x_s_info_lst'][0]
        sdk.motion_stitch.setup(relative_d=True, is_image_flag=True, x_s_info=source_info,
                               motion_limits={} if motion_profile == 'original' else {
                                   'pose_degrees': 8., 'pose_step': 2., 'lip_radius': .04,
                                   'eye_radius': .025, 'expression_step': .015})
        pcm = bytearray()
        # Same HuBERT context and zero padding as Wav2Feat.wav2feat: 5 new frames,
        # three left-context units and two right-context units of 40 ms.
        audio16 = np.zeros(2000, dtype=np.float32)
        resampler = soxr.ResampleStream(24000, 16000, 1, dtype='float32')
        features = np.empty((0, 1024), dtype=np.float32)
        feature_offset = motion_offset = emitted = 0
        motion = None
        first = None
        seq = sdk.audio2motion.seq_frames
        stride = sdk.audio2motion.valid_clip_len
        motion_seconds = render_seconds = 0.

        def inputs():
            for chunk in audio_chunks:
                if not chunk or len(chunk) % 2:
                    raise ValueError('Audio chunks must contain whole PCM16 samples')
                yield chunk, False
            yield b'', True

        for chunk, ended in inputs():
            pcm.extend(chunk)
            if len(pcm) > 24000 * 2 * 120:
                raise ValueError('Speech exceeds the 120-second turn limit')
            samples = np.frombuffer(chunk, dtype='<i2').astype(np.float32) / 32768
            audio16 = np.concatenate((audio16, resampler.resample_chunk(samples, last=ended)))
            frame_count = math.ceil(len(pcm) / 1920)
            if ended:
                audio16 = np.concatenate((audio16, np.zeros(6480, dtype=np.float32)))
            while feature_offset + 6480 <= len(audio16) and (not ended or len(features) < frame_count):
                block = sdk.wav2feat(audio16[feature_offset:feature_offset + 6480])
                features = np.concatenate((features, block))
                feature_offset += 3200
            if ended:
                features = features[:frame_count]
            while motion_offset < len(features):
                block = features[motion_offset:motion_offset + seq]
                if len(block) < seq and not ended:
                    break
                if len(block) < seq:
                    block = np.concatenate((block, np.repeat(block[-1:], seq - len(block), axis=0)))
                t = time.perf_counter()
                conditions = sdk.condition_handler(block, motion_offset)[None]
                motion = sdk.audio2motion(conditions, motion)
                motion_offset += stride
                motion_seconds += time.perf_counter() - t
                if not ended:
                    # Preserve the overlap and one frame of final smoothing lookahead.
                    stop = motion.shape[1] - sdk.audio2motion.overlap_v2 - sdk.audio2motion.smo_k_d // 2
                    finalized = sdk.audio2motion._smo(motion.copy(), 0, motion.shape[1])
                    for batch in self.driving_batches(sdk.audio2motion.cvt_fmt(finalized[:, emitted:stop]), first=emitted == 0):
                        t = time.perf_counter()
                        images = self.render_batch(source_info, batch)
                        render_seconds += time.perf_counter() - t
                        first = first or time.perf_counter() - started
                        for pixels in images:
                            yield pixels, bytes(pcm[emitted * 1920:(emitted + 1) * 1920])
                            emitted += 1
            if ended and motion is not None:
                finalized = sdk.audio2motion._smo(motion[:, :frame_count].copy(), 0, frame_count)
                for batch in self.driving_batches(sdk.audio2motion.cvt_fmt(finalized[:, emitted:]), first=emitted == 0):
                    t = time.perf_counter()
                    images = self.render_batch(source_info, batch)
                    render_seconds += time.perf_counter() - t
                    first = first or time.perf_counter() - started
                    for pixels in images:
                        yield pixels, bytes(pcm[emitted * 1920:(emitted + 1) * 1920]).ljust(1920, b'\0')
                        emitted += 1
        self.metrics = dict(frames=emitted, video_seconds=emitted / 25, first_frame_seconds=first,
                            turn_seconds=time.perf_counter() - started, portrait_cached=cached,
                            portrait_seconds=prepared - started, motion_seconds=motion_seconds,
                            render_seconds=render_seconds, sampling_steps=sampling_steps,
                            motion_profile=motion_profile, render_batch_size=self.render_batch_size,
                            render_backend=self.render_backend)

    def driving_batches(self, driving, *, first=False):
        """Render the first frame immediately, then bounded groups without dropping frames."""
        batch = []
        for item in driving:
            batch.append(item)
            if first or len(batch) == self.render_batch_size:
                yield batch
                batch, first = [], False
        if batch:
            yield batch

    def render_batch(self, source_info, driving):
        if self.render_backend == 'mlx':
            points = [self.sdk.motion_stitch(source_info, item) for item in driving]
            crops = self.mlx_renderer.render(np.concatenate([pair[0] for pair in points]),
                                              np.concatenate([pair[1] for pair in points]))
            return [self.putback(crop) for crop in crops]
        if len(driving) == 1 or not self.tensor_path:
            return [self.render(source_info, item) for item in driving]
        sdk = self.sdk
        # Stitching contains temporal state, so its order must stay sequential.
        points = [sdk.motion_stitch(source_info, item) for item in driving]
        x_s = np.concatenate([pair[0] for pair in points])
        x_d = np.concatenate([pair[1] for pair in points])
        feature = sdk.warp_f3d.warp_net(self.feature.expand(len(driving), -1, -1, -1, -1),
                                       x_s, x_d, return_tensor=True)
        crops = sdk.decode_f3d.decoder(feature, return_batch=True)
        return [self.putback(crop) for crop in crops]

    def putback(self, crop):
        image = self.sdk.putback(self.source['img_rgb_lst'][0], crop, self.source['M_c2o_lst'][0])
        return cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def render(self, source_info, driving):
        if self.render_backend == 'mlx':
            return self.render_batch(source_info, [driving])[0]
        sdk = self.sdk
        x_s, x_d = sdk.motion_stitch(source_info, driving)
        if self.tensor_path:
            feature = sdk.warp_f3d.warp_net(self.feature, x_s, x_d, return_tensor=True)
        else:
            feature = sdk.warp_f3d(self.feature, x_s, x_d)
        crop = sdk.decode_f3d(feature)
        return self.putback(crop)
