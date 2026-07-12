"""Mathematical invariant tests for rational-quadratic splines."""

from __future__ import annotations

import pytest


tf = pytest.importorskip("tensorflow", reason="TensorFlow architecture dependency")
np = pytest.importorskip("numpy", reason="NumPy architecture dependency")

from O.NN.rqs import (  # noqa: E402
    clip_by_value_preserve_gradient,
    get_grid_,
    periodic_rqs_,
    rqs_,
    rqs_bin_,
    shift_,
)


ATOL = 5.0e-5


def _random_spline_parameters(batch: int, dim: int, n_bins: int):
    """Return reproducible unconstrained width, height, and slope tensors."""

    generator = tf.random.Generator.from_seed(1729)
    widths = generator.normal((batch, dim * n_bins), stddev=0.7)
    heights = generator.normal((batch, dim * n_bins), stddev=0.7)
    slopes = generator.normal((batch, dim * (n_bins + 1)), stddev=0.4)
    return widths, heights, slopes


@pytest.mark.architecture
def test_single_rqs_bin_is_identity_for_unit_secant_and_slopes():
    """A diagonal bin with unit endpoint slopes must be numerically neutral."""

    x = tf.linspace(0.05, 0.95, 19)
    y, forward_log_det = rqs_bin_(
        x, xA=0.0, xB=1.0, yA=0.0, yB=1.0, sA=1.0, sB=1.0
    )
    restored, inverse_log_det = rqs_bin_(
        y, xA=0.0, xB=1.0, yA=0.0, yB=1.0, sA=1.0, sB=1.0, forward=False
    )

    # ``tf.linspace`` defaults to float32 here.  A few rational operations can
    # accumulate several float32 ULPs even for the analytical identity case.
    float32_atol = 2.0e-6
    np.testing.assert_allclose(y.numpy(), x.numpy(), atol=float32_atol, rtol=0.0)
    np.testing.assert_allclose(
        restored.numpy(), x.numpy(), atol=float32_atol, rtol=0.0
    )
    np.testing.assert_allclose(forward_log_det.numpy(), 0.0, atol=float32_atol)
    np.testing.assert_allclose(inverse_log_det.numpy(), 0.0, atol=float32_atol)


@pytest.mark.architecture
@pytest.mark.parametrize("identity_boundaries", [False, True])
def test_rqs_forward_inverse_and_log_jacobian_cancel(identity_boundaries):
    """Inverse evaluation must recover values and cancel elementwise log-Jacobians."""

    batch, dim, n_bins = 32, 5, 6
    x = tf.reshape(tf.linspace(-0.93, 0.93, batch * dim), (batch, dim))
    widths, heights, slopes = _random_spline_parameters(batch, dim, n_bins)

    y, forward_log_det = rqs_(
        x,
        widths,
        heights,
        slopes,
        identity_BCs=identity_boundaries,
        forward=True,
    )
    restored, inverse_log_det = rqs_(
        y,
        widths,
        heights,
        slopes,
        identity_BCs=identity_boundaries,
        forward=False,
    )

    np.testing.assert_allclose(restored.numpy(), x.numpy(), atol=ATOL, rtol=2.0e-5)
    np.testing.assert_allclose(
        (forward_log_det + inverse_log_det).numpy(), 0.0, atol=2.0e-4, rtol=0.0
    )
    assert np.isfinite(y.numpy()).all()
    assert np.max(np.abs(y.numpy())) <= 1.0 + ATOL


@pytest.mark.architecture
def test_periodic_shift_wraps_and_round_trips():
    """Periodic translations must stay in-domain and invert modulo the interval."""

    x = tf.constant([[-1.0, -0.75, 0.25, 0.999]], dtype=tf.float32)
    shifted = shift_(x, shifts=0.63)
    restored = shift_(shifted, shifts=0.63, forward=False)

    assert np.all(shifted.numpy() >= -1.0)
    assert np.all(shifted.numpy() < 1.0)
    np.testing.assert_allclose(restored.numpy(), x.numpy(), atol=1.0e-6, rtol=0.0)


@pytest.mark.architecture
def test_periodic_rqs_forward_inverse_and_log_jacobian_cancel():
    """The two-shift periodic composition must remain a bijection on the circle."""

    transforms, batch, dim, n_bins = 2, 24, 3, 5
    generator = tf.random.Generator.from_seed(31415)
    x = generator.uniform((batch, dim), minval=-0.98, maxval=0.98)
    widths = generator.normal((transforms, batch, dim * n_bins), stddev=0.5)
    heights = generator.normal((transforms, batch, dim * n_bins), stddev=0.5)
    slopes = generator.normal((transforms, batch, dim * (n_bins + 1)), stddev=0.3)
    shifts = generator.uniform((transforms, batch, dim), minval=-1.0, maxval=1.0)

    y, forward_log_det = periodic_rqs_(x, widths, heights, slopes, shifts)
    restored, inverse_log_det = periodic_rqs_(
        y, widths, heights, slopes, shifts, forward=False
    )

    np.testing.assert_allclose(restored.numpy(), x.numpy(), atol=1.5e-4, rtol=5.0e-5)
    np.testing.assert_allclose(
        (forward_log_det + inverse_log_det).numpy(), 0.0, atol=4.0e-4, rtol=0.0
    )


@pytest.mark.architecture
def test_grid_endpoints_ordering_and_minimum_width():
    """Network logits must produce ordered knots spanning the requested interval."""

    dim, n_bins = 3, 4
    grid = get_grid_(
        tf.zeros((2, dim * n_bins)),
        dim=dim,
        n_bins=n_bins,
        min_bin_w=0.05,
        interval=[-1.0, 1.0],
        shape_parallel=[],
    ).numpy()

    np.testing.assert_allclose(grid[..., 0], -1.0, atol=1.0e-7)
    np.testing.assert_allclose(grid[..., -1], 1.0, atol=1.0e-7)
    assert np.all(np.diff(grid, axis=-1) >= 0.1 - 1.0e-7)


@pytest.mark.architecture
def test_gradient_preserving_clip_has_identity_gradient_outside_bounds():
    """Stability clipping must not suppress gradients used for optimisation."""

    x = tf.Variable([-3.0, 0.25, 4.0], dtype=tf.float32)
    with tf.GradientTape() as tape:
        y = clip_by_value_preserve_gradient(x, -1.0, 1.0)
        loss = tf.reduce_sum(y)
    gradient = tape.gradient(loss, x)

    np.testing.assert_allclose(y.numpy(), [-1.0, 0.25, 1.0], atol=0.0)
    np.testing.assert_allclose(gradient.numpy(), np.ones(3), atol=0.0)
