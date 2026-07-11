Building the documentation
==========================

Activate the ``fecrys`` Conda environment and run the following command from
the repository root:

.. code-block:: console

   python -m sphinx -W --keep-going -b html docs docs/_build/html

The generated website starts at ``docs/_build/html/index.html``.  API pages are
regenerated from the source docstrings whenever Sphinx starts, so no FECrys
module is imported during the build.

To regenerate only the reStructuredText API pages while editing docstrings:

.. code-block:: console

   python docs/generate_api.py

Docstring conventions
---------------------

Use concise NumPy-style docstrings.  Explain the scientific meaning of inputs
and outputs, include units and array shapes where known, and distinguish
whole-crystal quantities from per-molecule quantities.  A function should be
understandable from its docstring without reading its implementation first.
