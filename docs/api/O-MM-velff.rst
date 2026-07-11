.. _api-O-MM-velff:

O.MM.velff
==========

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/velff.py>`__

.. rubric:: Docstring

.. code-block:: text

   Backward-compatible import location for the :class:`velff` force field.

   The active implementation lives in :mod:`O.MM.ff_setup`. This module keeps the
   historical ``O.MM.velff.velff`` import path available so that older notebooks
   and serialized Python objects can still resolve the class during unpickling.
   New code may import the class from either location; both names refer to the
   same class object.


This module defines no active Python classes or functions.
