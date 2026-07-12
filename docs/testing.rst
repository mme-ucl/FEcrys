Testing the model architecture
==============================

The first test layer protects the mathematical contracts of the TensorFlow
normalising-flow implementation. It is deliberately deterministic and small
enough to run on a CPU.

Run the complete fast suite from the repository root inside the ``fecrys``
environment:

.. code-block:: console

   conda activate fecrys
   python -m pytest

Run only neural-network architecture invariants with:

.. code-block:: console

   python -m pytest -m architecture

Current coverage
----------------

``tests/test_rqs.py`` checks:

* the exact identity case for one rational-quadratic spline bin;
* forward/inverse recovery for general splines;
* cancellation of forward and inverse log-Jacobians;
* periodic shifts and two-spline periodic compositions;
* ordered knot grids and minimum bin widths; and
* the identity-gradient contract of stability clipping.

``tests/test_spline_coupling.py`` checks:

* mixed periodic/ordinary half-layer inversion;
* neutral behaviour under identity initialisation;
* full two-half-layer inversion and log-Jacobian cancellation;
* preservation of auxiliary-only variables; and
* attention-layer flow masks that freeze selected variables.

``tests/test_pgmmol_integration.py`` takes the butane topology used by
``training_PGMmol.ipynb`` and checks:

* isolated Cartesian-to-internal-coordinate inversion;
* preservation of molecular geometry after translation and rotation are
  intentionally removed;
* cancellation of representation log-Jacobians;
* neutral coupling behaviour under ``identity_init=True``; and
* inversion of the complete representation-plus-``PGMmol`` composition.

The integration test constructs a small deterministic coordinate ensemble
from ``butane/seed_anti.pdb``. It does not load the production trajectory,
evaluate MACE energies, train a model, write artifacts, or run BAR, so it
remains suitable for routine local testing.

The tests use ``pytest.importorskip`` for TensorFlow. Documentation-only or
molecular-mechanics-only environments therefore report the architecture tests
as skipped rather than failing during collection.

Notebook-scale validation
-------------------------

The full ``training_PGMmol.ipynb`` remains an end-to-end scientific workflow:
it prepares a real trajectory and energies, trains the flow, samples it, and
estimates the absolute free energy by reweighting. Those expensive and
environment-dependent stages should be tested separately from the fast suite.
The next additions should cover one small training update, model save/load
parity, a reduced reweighting smoke test, complete ``PGMcrys`` composition,
and the analytical harmonic-system benchmark.
