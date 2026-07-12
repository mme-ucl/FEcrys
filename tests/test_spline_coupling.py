"""Architecture tests for spline coupling half and full layers."""

from __future__ import annotations

import pytest


tf = pytest.importorskip("tensorflow", reason="TensorFlow architecture dependency")
np = pytest.importorskip("numpy", reason="NumPy architecture dependency")

from O.NN.spline_layer import (  # noqa: E402
    SPLINE_COUPLING_HALF_LAYER,
    SPLINE_COUPLING_HALF_LAYER_AT,
    SPLINE_COUPLING_LAYER,
)


def _inputs(batch=12, n_mol=2, n_variables=5):
    """Generate reproducible values strictly inside the spline interval."""

    generator = tf.random.Generator.from_seed(2025)
    return generator.uniform(
        (batch, n_mol, n_variables), minval=-0.85, maxval=0.85
    )


def _assert_round_trip(layer, x, atol=3.0e-4):
    """Check value recovery, log-Jacobian cancellation, shape, and finiteness."""

    y, forward_log_det = layer.forward(x)
    restored, inverse_log_det = layer.inverse(y)

    assert y.shape == x.shape
    assert forward_log_det.shape == (x.shape[0], 1)
    assert np.isfinite(y.numpy()).all()
    assert np.isfinite(forward_log_det.numpy()).all()
    np.testing.assert_allclose(restored.numpy(), x.numpy(), atol=atol, rtol=5.0e-5)
    np.testing.assert_allclose(
        (forward_log_det + inverse_log_det).numpy(), 0.0, atol=8.0e-4, rtol=0.0
    )


@pytest.mark.architecture
def test_half_layer_round_trip_with_periodic_and_ordinary_variables():
    """A mixed-variable half layer must invert its conditioned transformation."""

    layer = SPLINE_COUPLING_HALF_LAYER(
        periodic_mask=[1, 0, 1, 0, 0],
        cond_mask=[1, 1, 0, 0, 0],
        n_bins=5,
        n_hidden=1,
        dims_hidden=[24],
        identity_init=False,
    )
    x = _inputs()
    y, forward_log_det = layer.forward_(x)
    restored, inverse_log_det = layer.inverse_(y)

    np.testing.assert_allclose(restored.numpy(), x.numpy(), atol=3.0e-4, rtol=5.0e-5)
    np.testing.assert_allclose(
        (forward_log_det + inverse_log_det).numpy(), 0.0, atol=8.0e-4, rtol=0.0
    )


@pytest.mark.architecture
def test_identity_initialised_full_coupling_is_neutral():
    """Identity initialisation must leave values and volume unchanged."""

    layer = SPLINE_COUPLING_LAYER(
        periodic_mask=[1, 0, 1, 0, 0],
        cond_mask=[1, 0, 2, 1, 0],
        n_bins=5,
        kwargs_for_given_half_layer_class={
            "n_hidden": 1,
            "dims_hidden": [20],
            "hidden_activation": tf.nn.silu,
            "identity_init": True,
        },
    )
    x = _inputs()
    y, log_det = layer.forward(x)

    np.testing.assert_allclose(y.numpy(), x.numpy(), atol=2.0e-5, rtol=0.0)
    np.testing.assert_allclose(log_det.numpy(), 0.0, atol=5.0e-5, rtol=0.0)


@pytest.mark.architecture
def test_full_coupling_round_trip_and_auxiliary_preservation():
    """Both half layers must invert, while mask value two remains auxiliary-only."""

    layer = SPLINE_COUPLING_LAYER(
        periodic_mask=[1, 0, 1, 0, 0],
        cond_mask=[1, 0, 2, 1, 0],
        n_bins=4,
        kwargs_for_given_half_layer_class={
            "n_hidden": 1,
            "dims_hidden": [18],
            "hidden_activation": tf.nn.tanh,
            "identity_init": False,
        },
    )
    x = _inputs()
    y, _ = layer.forward(x)

    np.testing.assert_allclose(y[..., 2].numpy(), x[..., 2].numpy(), atol=0.0)
    _assert_round_trip(layer, x)


@pytest.mark.architecture
def test_attention_half_layer_flow_mask_freezes_selected_variables():
    """A zero flow-mask entry must preserve its value and contribute no log volume."""

    flow_mask = np.ones((1, 2, 4), dtype=np.float32)
    flow_mask[0, 0, 1] = 0.0
    flow_mask[0, 1, 0] = 0.0
    layer = SPLINE_COUPLING_HALF_LAYER_AT(
        periodic_mask=[1, 0, 1, 0],
        cond_mask=[1, 1, 0, 0],
        flow_mask=flow_mask,
        n_mol=2,
        n_heads=1,
        embedding_dim=6,
        n_hidden_kqv=[1, 1, 1],
        one_hot_kqv=[False, False, False],
        n_hidden_decode=1,
        n_bins=4,
        identity_init=False,
    )
    x = _inputs(n_variables=4)
    y, forward_log_det = layer.forward_(x)
    restored, inverse_log_det = layer.inverse_(y)

    np.testing.assert_allclose(y[:, 0, 1].numpy(), x[:, 0, 1].numpy(), atol=0.0)
    np.testing.assert_allclose(y[:, 1, 0].numpy(), x[:, 1, 0].numpy(), atol=0.0)
    np.testing.assert_allclose(restored.numpy(), x.numpy(), atol=4.0e-4, rtol=5.0e-5)
    np.testing.assert_allclose(
        (forward_log_det + inverse_log_det).numpy(), 0.0, atol=1.0e-3, rtol=0.0
    )
