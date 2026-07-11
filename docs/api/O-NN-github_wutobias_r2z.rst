.. _api-O-NN-github_wutobias_r2z:

O.NN.github_wutobias_r2z
========================

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py>`__

.. rubric:: Docstring

.. code-block:: text

   Convert molecular Cartesian coordinates to and from Z-matrices.

   This unit-aware implementation is adapted from the ``wutobias/r2z`` project.
   It uses RDKit molecular graphs to choose a deterministic atom order and Pint
   quantities to make length and angle units explicit. Internal lengths are
   nanometres and internal angles are radians; formatted Z-matrices use angstroms
   and degrees for compatibility with common quantum-chemistry programs.


Classes and functions
---------------------

``pts_to_bond`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L21>`__

.. code-block:: python

   def pts_to_bond(A, B)

.. rubric:: Docstring

.. code-block:: text

   Return the distance between two Pint coordinate vectors in nanometres.


``pts_to_angle`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L27>`__

.. code-block:: python

   def pts_to_angle(A, B, C)

.. rubric:: Docstring

.. code-block:: text

   Return the A-B-C angle as a Pint quantity in radians.

   Point ``B`` is the vertex. Inputs must be length-valued three-component
   Pint quantities convertible to nanometres.


``pts_to_dihedral`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L40>`__

.. code-block:: python

   def pts_to_dihedral(A, B, C, D)

.. rubric:: Docstring

.. code-block:: text

   Return the signed A-B-C-D dihedral angle in radians.

   The sign is determined by the orientation of the two plane normals about
   the B-C bond. Degenerate collinear inputs can produce undefined values.


``ZMatrix`` (class)
^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L64>`__

.. code-block:: python

   class ZMatrix(rdmol, root_atm_idx=0)

.. rubric:: Docstring

.. code-block:: text

   Build and apply a graph-derived molecular Z-matrix definition.

   Parameters
   ----------
   rdmol : rdkit.Chem.Mol
       Connected molecular graph used to determine neighbours and symmetry
       ranks. A conformer is not required for defining the Z-matrix.
   root_atm_idx : int, default=0
       RDKit atom index from which graph traversal begins.

   Attributes
   ----------
   ordered_atom_list : list of int
       RDKit atom indices in Z-matrix construction order.
   z : dict
       Mapping from Z-order row to RDKit atom indices. Each row contains the
       placed atom followed by its bond, angle, and dihedral reference atoms
       when those references exist.
   zz : dict
       Equivalent mapping expressed entirely in Z-order indices.


``ZMatrix.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L91>`__

.. code-block:: python

   def __init__(self, rdmol, root_atm_idx=0)

.. rubric:: Docstring

.. code-block:: text

   Derive atom ordering and reference rows from an RDKit molecule.


``ZMatrix.z2a`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L108>`__

.. code-block:: python

   def z2a(self, z_idx)

.. rubric:: Docstring

.. code-block:: text

   Map a Z-matrix row index to its RDKit atom index.


``ZMatrix.a2z`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L112>`__

.. code-block:: python

   def a2z(self, atm_idx)

.. rubric:: Docstring

.. code-block:: text

   Map an RDKit atom index to its Z-matrix row index.


``ZMatrix.zzit`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L116>`__

.. code-block:: python

   def zzit(self)

.. rubric:: Docstring

.. code-block:: text

   Rebuild ``zz`` by converting all reference atoms to Z-order indices.


``ZMatrix.get_neighbor_idxs`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L124>`__

.. code-block:: python

   def get_neighbor_idxs(self, atm_idx)

.. rubric:: Docstring

.. code-block:: text

   Yield bonded neighbour indices ordered by RDKit canonical rank.


``ZMatrix.get_path_length`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L135>`__

.. code-block:: python

   def get_path_length(self, atm_idx1, atm_idx2, maxlength=100)

.. rubric:: Docstring

.. code-block:: text

   Return the shortest bond-path length between two atoms.

   Returns ``-1`` when no path is found before ``maxlength``. A path from
   an atom to itself has length zero and directly bonded atoms have length
   one.


``ZMatrix.get_shortest_paths`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L160>`__

.. code-block:: python

   def get_shortest_paths(self, atm_idx1, atm_idx2, query_pool=list(), maxattempts=100)

.. rubric:: Docstring

.. code-block:: text

   Enumerate candidate shortest bond paths between two atoms.

   ``query_pool`` optionally restricts permitted intermediate atoms and
   ``maxattempts`` limits tested permutations. Returns a list of atom-index
   paths, including both endpoints. The empty list means no path was
   constructed under the supplied restrictions.


``ZMatrix.get_k_nearest_neighbors`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L208>`__

.. code-block:: python

   def get_k_nearest_neighbors(self, atm_idx, k=3)

.. rubric:: Docstring

.. code-block:: text

   Return unique atoms reachable within at most ``k`` bonds.

   With ``k=0`` the result contains only ``atm_idx``. For positive ``k``
   the source atom itself is excluded.


``ZMatrix.add_atom`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L238>`__

.. code-block:: python

   def add_atom(self, atm_idx)

.. rubric:: Docstring

.. code-block:: text

   Append one atom and choose its Z-matrix reference atoms.

   Existing atoms are ignored and return ``False``. For a new atom the
   method favours reference patterns from symmetry-equivalent atoms, then
   chemically connected paths, and finally any geometrically valid set.
   Returns whether a row was successfully added and mutates ``z``,
   ``ordered_atom_list``, and atom counters.


``ZMatrix.order_atoms`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L338>`__

.. code-block:: python

   def order_atoms(self, atm_idx)

.. rubric:: Docstring

.. code-block:: text

   Recursively traverse neighbours and populate the Z-matrix ordering.

   Terminal atoms are postponed until at least four atoms have been
   placed where possible, providing non-degenerate reference choices.


``ZMatrix.is_dead_end`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L360>`__

.. code-block:: python

   def is_dead_end(self, atm_idx)

.. rubric:: Docstring

.. code-block:: text

   Return whether an atom has fewer than two bonded neighbours.


``ZMatrix.is_neighbor_of`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L368>`__

.. code-block:: python

   def is_neighbor_of(self, atm_idx1, atm_idx2)

.. rubric:: Docstring

.. code-block:: text

   Return whether two RDKit atoms share a bond.


``ZMatrix.build_cart_crds`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L376>`__

.. code-block:: python

   def build_cart_crds(self, z_crds, virtual_bond=None, virtual_angles=None, virtual_dihedrals=None, attach_crds=None, z_order=False)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct Cartesian coordinates from internal coordinates.

   Parameters
   ----------
   z_crds : mapping or sequence
       One row per Z-matrix atom. Rows after the root contain bond length,
       then angle, then dihedral as these become defined; entries must be
       Pint quantities.
   virtual_bond, virtual_angles, virtual_dihedrals : Pint quantities, optional
       Reference geometry used to place the first three atoms.
   attach_crds : Pint quantity, shape (3, 3), optional
       Three virtual Cartesian reference points in columns.
   z_order : bool, default=False
       Return rows in Z-matrix order instead of original RDKit atom order.

   Returns
   -------
   pint.Quantity, shape (n_atoms, 3)
       Cartesian coordinates in nanometres.

   Notes
   -----
   Placement uses the Natural Extension Reference Frame algorithm. An
   exception is raised if a row refers to an atom not yet placed.


``ZMatrix.build_pretty_zcrds`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L484>`__

.. code-block:: python

   def build_pretty_zcrds(self, crds)

.. rubric:: Docstring

.. code-block:: text

   Format Cartesian coordinates as a conventional Z-matrix string.

   Atom labels come from RDKit. References are one-based, bond lengths are
   printed in angstroms, and angles/dihedrals in degrees.


``ZMatrix.build_z_crds`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/github_wutobias_r2z.py#L508>`__

.. code-block:: python

   def build_z_crds(self, crds)

.. rubric:: Docstring

.. code-block:: text

   Convert Cartesian coordinates into unit-aware Z-matrix values.

   Parameters
   ----------
   crds : Pint quantity, shape (n_atoms, 3)
       Coordinates indexed in the original RDKit atom order.

   Returns
   -------
   dict
       Mapping from Z row to lists of Pint quantities. Bond lengths are in
       nanometres; angles and dihedrals are in degrees. The root row stores
       its Cartesian position because it has no internal references.
