"""Regression tests for public model-constructor compatibility."""

from __future__ import annotations

import pytest


tf = pytest.importorskip("tensorflow", reason="TensorFlow architecture dependency")
np = pytest.importorskip("numpy", reason="NumPy architecture dependency")

from O.NN.pgm import PGMmol  # noqa: E402


class _MinimalMoleculeMap:
    """Small protocol-compatible map for constructor-only model tests."""

    n_mol = 1
    n_atoms_mol = 4
    periodic_mask = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

    @staticmethod
    def ln_base_(z):
        """Return a placeholder base log density with one value per sample."""

        return tf.zeros((z.shape[0], 1), dtype=tf.float32)

    @staticmethod
    def sample_base_(m):
        """Return placeholder molecular base variables."""

        return [
            tf.zeros((m, 0), dtype=tf.float32),
            tf.zeros((m, 1, 6), dtype=tf.float32),
        ]


@pytest.mark.architecture
def test_pgmmol_accepts_and_propagates_identity_initialisation():
    """The interface's identity option must reach PGMmol spline conditioners."""

    model = PGMmol(
        ic_maps=[_MinimalMoleculeMap()],
        n_layers=2,
        identity_init=True,
        initialise=False,
    )

    assert model.identity_init is True
    assert model.init_args["identity_init"] is True
    assert len(model.layers_C) == 2
    for layer in model.layers_C:
        assert layer.kwargs_for_given_half_layer_class["identity_init"] is True
