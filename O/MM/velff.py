"""Backward-compatible import location for the :class:`velff` force field.

The active implementation lives in :mod:`O.MM.ff_setup`. This module keeps the
historical ``O.MM.velff.velff`` import path available so that older notebooks
and serialized Python objects can still resolve the class during unpickling.
New code may import the class from either location; both names refer to the
same class object.
"""

from .ff_setup import velff

__all__ = ["velff"]
