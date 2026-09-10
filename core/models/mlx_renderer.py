"""Ditto's existing portrait weights on MLX; every motion frame is rendered.

The rendering core derives from FasterLivePortrait-MLX. See mlx_modules/PROVENANCE.md.
This adapter owns immutable source caches, conversion receipts and bounded batches.
"""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


def prepare_mlx_weights(models):
    """Convert trusted, installed PyTorch weights once; never download or train."""
    models = Path(models)
    destination = models / 'mlx'
    paths = {name: models / f'ditto_pytorch/models/{name}.pth'
             for name in ('warp_network', 'decoder')}
    signature = {'format': 1, 'sources': {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}}
    receipt = destination / 'conversion.json'
    if receipt.is_file():
        try:
            cached = json.loads(receipt.read_text())
        except (ValueError, OSError):
            cached = None
        if cached == signature and all((destination / f'{name}.npz').is_file()
                                       and (destination / f'{name}.npz').stat().st_size > 0
                                       for name in paths):
            return destination
    destination.mkdir(parents=True, exist_ok=True)
    from .mlx_modules.weight_convert import save_converted_npz
    with tempfile.TemporaryDirectory(dir=destination, prefix='convert-') as temporary:
        temporary = Path(temporary)
        for name, source in paths.items():
            result = temporary / f'{name}.npz'
            save_converted_npz(str(source), str(result))
        result = temporary / 'conversion.json'
        result.write_text(json.dumps(signature, indent=2))
        # Finish both conversions before publishing either checkpoint. Once
        # publishing begins, a crash must not leave an old successful receipt.
        receipt.unlink(missing_ok=True)
        for name in paths:
            (temporary / f'{name}.npz').replace(destination / f'{name}.npz')
        result.replace(receipt)
    return destination


class MlxFrameRenderer:
    def __init__(self, models):
        from .mlx_profiles import apply_mlx_profile
        apply_mlx_profile('quality')
        import mlx.core as mx
        from .mlx_liveportrait import MlxWarpingSpadeModel
        weights = prepare_mlx_weights(models)
        self.mx = mx
        self.model = MlxWarpingSpadeModel(
            model_path=[str(weights / 'warp_network.npz'), str(weights / 'decoder.npz')],
            dtype='fp16', precise_geometry=True, optimize=True,
            temporal_warp_interval=1, temporal_warp_threshold=0)
        self.feature = self.source_points = None
        self.batches = {}

    def prepare(self, feature):
        mx = self.mx
        # Own the source; external numpy mutation must never alter a cached face.
        source = np.array(feature, dtype=np.float32, copy=True)
        if source.shape != (1, 32, 16, 64, 64):
            raise ValueError('Expected a single registered LivePortrait feature volume')
        self.feature = mx.array(source.transpose(0, 2, 3, 4, 1)).astype(mx.float16)
        mx.eval(self.feature)
        self.source_points = None
        self.batches.clear()
        self.model.reset_temporal_cache()

    def render(self, source_points, driving_points):
        mx = self.mx
        count = len(driving_points)
        if self.feature is None or not 1 <= count <= 4:
            raise ValueError('Prepare a portrait and render one to four frames')
        source = np.asarray(source_points, dtype=np.float32)
        driving = np.asarray(driving_points, dtype=np.float32)
        if source.shape != (count, 21, 3) or driving.shape != source.shape:
            raise ValueError('Expected paired batches of 21 facial keypoints')
        if not np.isfinite(source).all() or not np.isfinite(driving).all():
            raise ValueError('Facial keypoints must be finite')
        if not np.all(source == source[:1]):
            raise ValueError('A render batch must share one source portrait')
        if self.source_points is None or not np.array_equal(source[:1], self.source_points):
            self.source_points = source[:1].copy()
            self.batches.clear()
        if count not in self.batches:
            self.batches[count] = (
                mx.broadcast_to(self.feature, (count, *self.feature.shape[1:])),
                mx.broadcast_to(mx.array(self.source_points), (count, 21, 3)))
        feature, points = self.batches[count]
        output = np.array(self.model.predict(feature, points, mx.array(driving), return_mx=True))
        if not np.isfinite(output).all():
            raise RuntimeError('MLX renderer produced a nonfinite frame')
        return output
