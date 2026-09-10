from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from core.atomic_components.motion_bounds import MotionBounds
from core.atomic_components.motion_stitch import ctrl_vad
from core.models.modules.conv3d_mps import conv3d_as_conv2d


def portrait():
    return dict(scale=np.ones((1, 1), np.float32),
                pitch=np.zeros((1, 66), np.float32), yaw=np.zeros((1, 66), np.float32),
                roll=np.zeros((1, 66), np.float32), t=np.zeros((1, 3), np.float32),
                exp=np.zeros((1, 63), np.float32), kp=np.zeros((1, 63), np.float32))


@pytest.mark.parametrize('kernel,padding,stride', [(3, 1, 1), (7, 3, 1), ((3, 5, 3), (1, 2, 1), (1, 2, 2))])
def test_conv_rewrite_preserves_operation_and_boundary_padding(kernel, padding, stride):
    torch.manual_seed(8)
    conv = torch.nn.Conv3d(3, 4, kernel, padding=padding, stride=stride).double()
    value = torch.randn(2, 3, 5, 7, 9, dtype=torch.float64)
    torch.testing.assert_close(conv3d_as_conv2d(value, conv), conv(value), rtol=1e-10, atol=1e-10)


def test_bounds_limit_outliers_and_preserve_source():
    source = portrait()
    for key in ('pitch', 'yaw', 'roll'):
        source[key] = np.zeros((1,), np.float32)
    bounds = MotionBounds(source)
    first = bounds(source)
    driving = {k: v.copy() for k, v in source.items()}
    driving['pitch'][:] = 90
    driving['exp'][:] = 1
    for _ in range(10):
        result = bounds(driving)
        assert np.max(np.abs(result['pitch'])) <= 8
        assert np.max(np.abs(result['pitch'] - first['pitch'])) <= 2
        delta = (result['exp'] - source['exp']).reshape(21, 3)
        assert np.max(np.linalg.norm(delta[[6, 12, 14, 17, 19, 20]], axis=-1)) <= .040001
        assert np.max(np.linalg.norm(delta[[11, 13, 15, 16, 18]], axis=-1)) <= .025001
        first = result
    assert not source['exp'].any()
    assert np.all(driving['exp'] == 1)


def test_silence_control_changes_lips_only():
    source = portrait()
    driving = {'exp': np.ones((1, 63), np.float32)}
    result = ctrl_vad(driving, source, 0)['exp'].reshape(21, 3)
    lips = [6, 12, 14, 17, 19, 20]
    assert not result[lips].any()
    assert np.all(np.delete(result, lips, axis=0) == 1)


def session():
    from core.atomic_components.audio2motion import Audio2Motion
    from portrait_session import PortraitSession
    class MotionModel:
        seq_frames = 80
        model = SimpleNamespace(sampling_timesteps=None)
        def setup(self, steps): pass
        def __call__(self, kp, cond, steps):
            return np.repeat(cond[:, :, :1], 265, axis=2).copy()
    class Features:
        def __call__(self, chunk):
            values = np.array([chunk[2000 + i * 640:2000 + (i + 1) * 640].mean() for i in range(5)])
            return np.repeat(values[:, None], 1024, axis=1)
    from core.atomic_components.condition_handler import ConditionHandler
    source_info = portrait()
    source = {'x_s_info_lst': [source_info], 'sc': np.zeros(63),
              'eye_open_lst': [np.ones((1, 2))], 'eye_ball_lst': [np.zeros((1, 6))]}
    motion = Audio2Motion.__new__(Audio2Motion); motion.lmdm = MotionModel()
    obj = PortraitSession.__new__(PortraitSession)
    obj.prepare = lambda _: True
    obj.source = source
    obj.cpu_rng = torch.get_rng_state(); obj.device_rng = None; obj.device = 'cpu'
    obj.sdk = SimpleNamespace(audio2motion=motion, condition_handler=ConditionHandler(),
                              motion_stitch=SimpleNamespace(setup=Mock()), wav2feat=Features())
    obj.render = lambda source, driving: np.concatenate((driving['pitch'], driving['exp']), axis=1).copy()
    return obj


@pytest.mark.parametrize('count', [1, 34, 70, 80, 150, 301])
def test_audio_chunk_boundaries_preserve_motion_and_frame_count(count):
    rng = np.random.default_rng(5)
    pcm = rng.integers(-10000, 10000, count * 960 - 7, dtype=np.int16).tobytes()
    whole = list(session().frames('portrait', [pcm]))
    chunked = list(session().frames('portrait', (pcm[i:i + 24000] for i in range(0, len(pcm), 24000))))
    assert len(whole) == len(chunked) == count
    np.testing.assert_allclose(np.stack([f for f, _ in whole]), np.stack([f for f, _ in chunked]), rtol=1e-5, atol=1e-6)
    assert b''.join(p for _, p in chunked) == pcm.ljust(count * 1920, b'\0')


def test_frames_are_emitted_before_all_audio_is_consumed():
    consumed = []
    def chunks():
        for i in range(24):
            consumed.append(i)
            yield bytes(24000)  # half a second
    iterator = session().frames('portrait', chunks())
    next(iterator)
    assert len(consumed) < 24
    iterator.close()
