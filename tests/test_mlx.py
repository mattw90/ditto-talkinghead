"""Apple-only arithmetic/cache regressions; no checkpoints or downloads needed."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
nn = pytest.importorskip('mlx.nn')

from core.models.mlx_modules.spade_generator import SPADE, residual_conv
from core.models.mlx_renderer import MlxFrameRenderer


def test_scaled_residual_preserves_bias_and_cancelling_large_activations():
    conv = nn.Conv2d(128, 2, 3, padding=1)
    # Large opposing sums, with a small final result and nonzero bias.
    weights = np.ones((2, 3, 3, 128), np.float16)
    weights[..., 64:] = -1
    conv.weight = mx.array(weights)
    conv.bias = mx.array([3., -2.], dtype=mx.float16)
    x = mx.full((1, 6, 6, 128), 2048., dtype=mx.float16)
    x = x.at[..., 0].add(2.)
    expected = mx.conv2d(x.astype(mx.float32), conv.weight.astype(mx.float32),
                         padding=1) + conv.bias.astype(mx.float32)
    actual = np.array(residual_conv(conv, x, scaled=True))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.array(expected), atol=.1, rtol=0)


def test_fused_spade_matches_original_and_can_be_prepared_twice():
    mx.random.seed(17)
    norm = SPADE(4, 3, hidden=8)
    x = mx.random.normal((2, 8, 8, 4))
    seg = mx.random.normal((2, 4, 4, 3))
    expected = np.array(norm(x, seg))
    norm.prepare_inference()
    norm.prepare_inference()
    np.testing.assert_allclose(np.array(norm(x, seg)), expected, atol=2e-6, rtol=2e-6)


def renderer_stub():
    renderer = MlxFrameRenderer.__new__(MlxFrameRenderer)
    renderer.mx = mx
    renderer.feature = renderer.source_points = None
    renderer.batches = {}
    renderer.model = SimpleNamespace(reset_temporal_cache=Mock(), predict=Mock(
        side_effect=lambda f, s, d, **kw: mx.zeros((len(d), 2, 2, 3))))
    return renderer


def test_portrait_owns_source_and_invalidates_keypoint_and_batch_caches():
    renderer = renderer_stub()
    feature = np.ones((1, 32, 16, 64, 64), np.float32)
    renderer.prepare(feature)
    feature.fill(2)
    assert bool(mx.all(renderer.feature == 1))
    source = np.full((4, 21, 3), .1234567, np.float32)
    renderer.render(source, source)
    cached = renderer.batches[4]
    assert cached[1].dtype == mx.float32
    np.testing.assert_array_equal(np.array(cached[1]), source)
    renderer.render(source.copy(), source + .01)
    assert renderer.batches[4] is cached
    renderer.render(source + .02, source)
    assert renderer.batches[4] is not cached
    renderer.prepare(feature)
    assert renderer.batches == {} and renderer.source_points is None
    assert bool(mx.all(renderer.feature == 2))
    assert renderer.model.reset_temporal_cache.call_count == 2


def test_renderer_rejects_bad_batches_and_nonfinite_outputs():
    renderer = renderer_stub()
    source = np.zeros((1, 21, 3), np.float32)
    with pytest.raises(ValueError, match='Prepare'):
        renderer.render(source, source)
    renderer.prepare(np.zeros((1, 32, 16, 64, 64), np.float32))
    with pytest.raises(ValueError, match='finite'):
        renderer.render(source, source + np.nan)
    with pytest.raises(ValueError, match='paired'):
        renderer.render(source, source[:, :20])
    with pytest.raises(ValueError, match='one source'):
        renderer.render(np.concatenate([source, source + 1]), np.repeat(source, 2, axis=0))
    renderer.model.predict.side_effect = lambda *a, **kw: mx.full((1, 2, 2, 3), float('nan'))
    with pytest.raises(RuntimeError, match='nonfinite'):
        renderer.render(source, source)


def test_conversion_reuses_only_matching_complete_receipts(tmp_path, monkeypatch):
    from core.models.mlx_renderer import prepare_mlx_weights
    from core.models.mlx_modules import weight_convert
    originals = tmp_path / 'ditto_pytorch/models'
    originals.mkdir(parents=True)
    for name in ('warp_network', 'decoder'):
        (originals / f'{name}.pth').write_bytes(b'installed-weights')
    converter = Mock(side_effect=lambda src, dest: np.savez(dest, weight=np.ones(1)))
    monkeypatch.setattr(weight_convert, 'save_converted_npz', converter)
    cache = prepare_mlx_weights(tmp_path)
    prepare_mlx_weights(tmp_path)
    assert converter.call_count == 2
    (originals / 'decoder.pth').write_bytes(b'new-weights')
    prepare_mlx_weights(tmp_path)
    assert converter.call_count == 4
    (cache / 'decoder.npz').unlink()
    prepare_mlx_weights(tmp_path)
    assert converter.call_count == 6
    (cache / 'conversion.json').write_text('{incomplete')
    prepare_mlx_weights(tmp_path)
    assert converter.call_count == 8
