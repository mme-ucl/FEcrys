"""Small integration tests derived from the ``training_PGMmol`` workflow.

These tests deliberately stop before molecular-mechanics energy evaluation,
training, and BAR reweighting.  They protect the deterministic mathematical
core that those expensive workflow stages depend on.
"""

from __future__ import annotations

from pathlib import Path

import pytest


tf = pytest.importorskip("tensorflow", reason="TensorFlow model dependency")
np = pytest.importorskip("numpy", reason="NumPy model dependency")
pytest.importorskip("rdkit", reason="RDKit coordinate-map dependency")
pytest.importorskip("mdtraj", reason="MDTraj coordinate-map dependency")

from O.NN.pgm import PGMmol  # noqa: E402
from O.NN.representation_layers import SingleMolecule_map  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUTANE_PDB = REPOSITORY_ROOT / "butane" / "seed_anti.pdb"


def _pdb_coordinates_nm(path: Path) -> np.ndarray:
    """Read PDB ``ATOM`` coordinates and convert Angstrom to nanometres."""

    coordinates = []
    with path.open(encoding="utf-8") as pdb_file:
        for line in pdb_file:
            if line.startswith(("ATOM  ", "HETATM")):
                coordinates.append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
    return np.asarray(coordinates, dtype=np.float32) / 10.0


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    """Return all Cartesian pair distances for every molecular frame."""

    displacement = coordinates[:, :, None, :] - coordinates[:, None, :, :]
    return np.linalg.norm(displacement, axis=-1)


@pytest.fixture(scope="module")
def fitted_butane_map_and_coordinates():
    """Fit the notebook's single-molecule map to a tiny synthetic trajectory."""

    base = _pdb_coordinates_nm(BUTANE_PDB)
    rng = np.random.default_rng(2026)
    coordinates = base[None, ...] + rng.normal(
        loc=0.0,
        scale=0.002,
        size=(32, *base.shape),
    ).astype(np.float32)

    coordinate_map = SingleMolecule_map(str(BUTANE_PDB))
    coordinate_map.set_ABCD_(ind_root_atom=1)
    coordinate_map.initalise_(r_dataset=coordinates, focused=True)
    return coordinate_map, coordinates


@pytest.mark.architecture
@pytest.mark.integration
def test_single_molecule_map_round_trip_preserves_geometry_and_jacobian(
    fitted_butane_map_and_coordinates,
):
    """The representation inverse must preserve geometry and cancel its Jacobian."""

    coordinate_map, coordinates = fitted_butane_map_and_coordinates
    batch = tf.convert_to_tensor(coordinates[:8], dtype=tf.float32)

    variables, forward_log_det = coordinate_map.forward_(batch)
    reconstructed, inverse_log_det = coordinate_map.inverse_(variables)
    repeated_variables, _ = coordinate_map.forward_(reconstructed)

    # An isolated-molecule map intentionally discards global translation and
    # rotation.  Pair distances, rather than absolute Cartesian positions, are
    # therefore the correct physical invariant for this round trip.
    np.testing.assert_allclose(
        _pairwise_distances(reconstructed.numpy()),
        _pairwise_distances(batch.numpy()),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        repeated_variables[1].numpy(),
        variables[1].numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        (forward_log_det + inverse_log_det).numpy(),
        0.0,
        atol=2e-4,
        rtol=0.0,
    )


@pytest.mark.architecture
@pytest.mark.integration
def test_identity_initialised_pgmmol_round_trip(
    fitted_butane_map_and_coordinates,
):
    """A complete identity-initialised PGMmol must invert its input geometry."""

    coordinate_map, coordinates = fitted_butane_map_and_coordinates
    batch = tf.convert_to_tensor(coordinates[:8], dtype=tf.float32)
    model = PGMmol(
        ic_maps=[coordinate_map],
        n_layers=2,
        identity_init=True,
        initialise=False,
    )

    representation, _ = coordinate_map.forward_(batch)
    latent, coupling_log_det = model._forward_coupling_(representation)
    reconstructed, inverse_log_det = model.inverse_(latent)
    repeated_latent, repeated_forward_log_det = model.forward_(reconstructed)

    np.testing.assert_allclose(
        latent[1].numpy(),
        representation[1].numpy(),
        atol=3e-6,
        rtol=3e-6,
    )
    np.testing.assert_allclose(
        coupling_log_det.numpy(),
        0.0,
        atol=3e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        repeated_latent[1].numpy(),
        latent[1].numpy(),
        atol=3e-5,
        rtol=3e-5,
    )
    np.testing.assert_allclose(
        (repeated_forward_log_det + inverse_log_det).numpy(),
        0.0,
        atol=3e-4,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        _pairwise_distances(reconstructed.numpy()),
        _pairwise_distances(batch.numpy()),
        atol=3e-5,
        rtol=3e-5,
    )
