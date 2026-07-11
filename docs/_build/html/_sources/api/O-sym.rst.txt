.. _api-O-sym:

O.sym
=====

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py>`__

.. rubric:: Docstring

.. code-block:: text

   Symmetry-based relabelling and augmentation of molecular trajectories.

   The routines permute chemically equivalent atom indices or crystallographically
   equivalent unit-cell blocks without changing the represented physical state.
   They are preprocessing tools for models that are not intrinsically invariant
   to these permutations. Coordinate values are generally preserved or translated
   by periodic lattice vectors; atom ordering is what changes.


Classes and functions
---------------------

``cluster_symmetric_torsion_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L16>`__

.. code-block:: python

   def cluster_symmetric_torsion_(x, symmetry_order: int=1, offset=np.pi)

.. rubric:: Docstring

.. code-block:: text

   Assign periodic angles to symmetry-equivalent angular sectors.

   Parameters
   ----------
   x : array-like
       Angles in radians; flattened before classification.
   symmetry_order : int, default=1
       Number of equally spaced sectors across ``[-pi, pi)``.
   offset : float, default=pi
       Phase controlling the sector boundaries.

   Returns
   -------
   numpy.ndarray
       Integer sector labels from zero to ``symmetry_order - 1``.


``test_cluster_symmetric_torsion_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L45>`__

.. code-block:: python

   def test_cluster_symmetric_torsion_(symmetry_order=3, m=200, flattness=0.5)

.. rubric:: Docstring

.. code-block:: text

   test_cluster_symmetric_torsion_(symmetry_order=10, m=1000, flattness=0.2)


``DatasetSymmetryReduction`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L60>`__

.. code-block:: python

   class DatasetSymmetryReduction(path_dataset: str, r=None, n_mol=None, n_atoms_mol=None, PDB_single_mol=None)

.. rubric:: Docstring

.. code-block:: text

   Chemically identical atoms can be swapped, but the current version of PGM does not understand this.
   We can swap them in the training data to help ergodicity.
   This .py file provides a few methods to force ergodic sampling of some of the symmetry orbits,
   even if this was not explicitly sampled in the finite MD trajectories.

   More methods are expected to be added over time covering different use-cases.

   This .py file can be deleted as soon as a fully symmetry-aware PGM (based on atom types) is added (future work).


``DatasetSymmetryReduction.save_sym_reduced_dataset_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L74>`__

.. code-block:: python

   def save_sym_reduced_dataset_(self, path_dataset=None, key='_sym_reduced')

.. rubric:: Docstring

.. code-block:: text

   Before saving the processed dataset, can check_energy_() to confirm that energy did not change. 
   Configurations before (self.r_init) and after (self.r) the reduction should have exactly the same energy.
   [Allowing a slight noise if comparing to energies saved during trajectory; because of numerical precision]
   This is because only the atomic indices are swapped in this processing step (coordinates are not changed).


``DatasetSymmetryReduction.check_energy_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L89>`__

.. code-block:: python

   def check_energy_(self, m=None)

.. rubric:: Docstring

.. code-block:: text

   Re-evaluate processed frames and report maximum energy deviation.

   At most the first ``m`` frames are checked. The new reduced energies
   are stored as ``u_sym`` and compared against the original simulation
   energies, which should agree up to numerical precision.


``DatasetSymmetryReduction.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L99>`__

.. code-block:: python

   def __init__(self, path_dataset: str, r=None, n_mol=None, n_atoms_mol=None, PDB_single_mol=None)

.. rubric:: Docstring

.. code-block:: text

   Load or directly configure a trajectory for symmetry reduction.

   Supplying ``path_dataset`` reconstructs ``SingleComponent`` metadata,
   coordinates, boxes, molecule counts, and the single-molecule topology.
   For direct construction, ``r``, ``n_mol``, ``n_atoms_mol``, and
   ``PDB_single_mol`` must be supplied; current code also expects boxes to
   be available through the dataset path. The working coordinates ``r``
   are a copy of immutable baseline ``r_init``.


``DatasetSymmetryReduction.restart_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L171>`__

.. code-block:: python

   def restart_(self)

.. rubric:: Docstring

.. code-block:: text

   Reset working coordinates to the original trajectory copy.


``DatasetSymmetryReduction.set_ABCD_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L178>`__

.. code-block:: python

   def set_ABCD_(self, ind_root_atom, option: int=None)

.. rubric:: Docstring

.. code-block:: text

   Define internal-coordinate references and the molecular anchor atom.


``DatasetSymmetryReduction.cluster_symmetric_torsion_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L185>`__

.. code-block:: python

   def cluster_symmetric_torsion_(self, phi, symmetry_order: int, offset=np.pi)

.. rubric:: Docstring

.. code-block:: text

   Return all cyclic permutations of torsion-sector assignments.

   ``phi`` must contain one value per frame. The result has shape
   ``(n_frames, symmetry_order)`` and each row is a permutation of sector
   labels used to canonicalise equivalent atoms.


``DatasetSymmetryReduction._prepare_sort_methyl_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L199>`__

.. code-block:: python

   def _prepare_sort_methyl_(self)

.. rubric:: Docstring

.. code-block:: text

   Find methyl groups and cache their three hydrogen torsion rows.


``DatasetSymmetryReduction._sort_methyl_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L215>`__

.. code-block:: python

   def _sort_methyl_(self, ind_methyl, ind_mol, lookup_index=0, offset=np.pi)

.. rubric:: Docstring

.. code-block:: text

   Relabel one methyl group's hydrogens across all frames.

   ``lookup_index`` selects one of four deterministic canonicalisation
   tables. A negative value applies a random cyclic rotation per frame.
   Coordinates in ``self.r`` are mutated in place.


``DatasetSymmetryReduction.sort_methyl_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L257>`__

.. code-block:: python

   def sort_methyl_(self, lookup_indices=[0, 3], offsets=[np.pi])

.. rubric:: Docstring

.. code-block:: text

   Canonicalise every detected methyl group in every molecule.

   Lookup patterns may be shared by all methyl groups or supplied per
   group; offsets are radians and follow the same grouping. The operation
   only permutes equivalent hydrogen coordinates in ``self.r``.


``DatasetSymmetryReduction.plot_methyl_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L282>`__

.. code-block:: python

   def plot_methyl_(self, axes_off=True, figsize=(6, 6))

.. rubric:: Docstring

.. code-block:: text

   Plot three hydrogen-torsion distributions for each methyl group.


``DatasetSymmetryReduction._prepare_sort_trimethyl_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L314>`__

.. code-block:: python

   def _prepare_sort_trimethyl_(self)

.. rubric:: Docstring

.. code-block:: text

   Find tert-butyl-like trimethyl groups and cache atom partitions.


``DatasetSymmetryReduction._sort_trimethyl_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L352>`__

.. code-block:: python

   def _sort_trimethyl_(self, ind_trimethyl, ind_mol, lookup_index=0, offset=0)

.. rubric:: Docstring

.. code-block:: text

   Relabel three methyl branches of one trimethyl group.

   Each branch carbon and its three attached hydrogens move together.
   Deterministic lookup tables are used for non-negative ``lookup_index``;
   a negative index chooses a random cyclic branch rotation per frame.


``DatasetSymmetryReduction.sort_trimethyl_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L416>`__

.. code-block:: python

   def sort_trimethyl_(self, lookup_indices=[0, 3], offset=0)

.. rubric:: Docstring

.. code-block:: text

   Canonicalise all detected trimethyl branch permutations in place.


``DatasetSymmetryReduction.plot_trimethyl_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L428>`__

.. code-block:: python

   def plot_trimethyl_(self, mask_0=True, axes_off=True, figsize=(2, 10))

.. rubric:: Docstring

.. code-block:: text

   Plot branch-torsion distributions for detected trimethyl groups.


``DatasetSymmetryReduction.plot_torsion_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L459>`__

.. code-block:: python

   def plot_torsion_(self, index_of_atom: int, axes_off=False)

.. rubric:: Docstring

.. code-block:: text

   # plot_mol_larger_(self.sc.mol) ; to see atoms
   # self.ic_map.ABCD_IC           ; to see all indices


``DatasetSymmetryReduction.sort_n2_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L489>`__

.. code-block:: python

   def sort_n2_(self, inds_AB: list, lookup_indices: list=[-1], offset=0)

.. rubric:: Docstring

.. code-block:: text

   Relabel a user-selected pair of symmetry-equivalent atoms.

   The two atoms must share the same B-C-D internal-coordinate references.
   ``lookup_indices`` selects identity/swap canonicalisation per molecule;
   ``-1`` randomises the pair independently in each frame.


``DatasetSymmetryReduction._sort_n2_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L506>`__

.. code-block:: python

   def _sort_n2_(self, inds_A, inds_B, ind_mol, lookup_index=0, offset=0)

.. rubric:: Docstring

.. code-block:: text

   Apply one two-atom relabelling rule to a single molecule.


``DatasetSymmetryReduction.plot_n2_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L531>`__

.. code-block:: python

   def plot_n2_(self, inds_AB: list, axes_off=True, figsize=(6, 6))

.. rubric:: Docstring

.. code-block:: text

   Plot torsion distributions for a selected equivalent-atom pair.


``DatasetSymmetryReduction.sort_unitcells_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L559>`__

.. code-block:: python

   def sort_unitcells_(self, n_mol_unitcell, n_images_search=1)

.. rubric:: Docstring

.. code-block:: text

   Experimental method to improve ergodicity of finite MD data, from a point of view of a simple, 
   non-permutationally invariant (symmetry unaware) model. This method only applicable in supercell 
   data where there are more than one unit cell building blocks. 
   Symmetry augmentation : randomisation, such that each region of space inside a supercell 
   becomes effectively sampled by more than one crystallographically equivalent molecule.  
   Each molecule inside a unit cell building block is considered crystallographically unique. 
   Therefore, whole unit cells are reshuffled, rather than individual molecules.
   The output dataset (self.r) will always have the same energy as the input (self.r_init), 
   which is true for any method in this .py file, but the output from this method should still 
   be treated carefully as explained blow. This aspect matters form the point of view of a native 
   model (for accurate entropy differences between states; purpose of this .py file).


``DatasetSymmetryReduction.check_sorted_unitcells_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L591>`__

.. code-block:: python

   def check_sorted_unitcells_(self, batch_size=10000)

.. rubric:: Docstring

.. code-block:: text

   check that sort_unitcells_ worked correctly


``PermuteUnitcell_SingleComponent`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L633>`__

.. code-block:: python

   class PermuteUnitcell_SingleComponent(n_atoms_mol, n_mol, n_mol_unitcell, ind_rO, n_images_search=1)

.. rubric:: Docstring

.. code-block:: text

   Randomly translate and relabel equivalent unit-cell blocks.

   This augmentation targets single-component supercells containing repeated
   crystallographic unit cells. A random unit-cell origin is chosen per frame,
   and translated blocks are assigned back to reference sites using a minimum
   distance linear assignment. Whole molecules remain intact.


``PermuteUnitcell_SingleComponent.put_in_box_m_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L642>`__

.. code-block:: python

   def put_in_box_m_(self, r, b)

.. rubric:: Docstring

.. code-block:: text

   adapted from SingleComponent_map_rb.remove_COM_from_data_ in O/NN/pgm_rb.py


``PermuteUnitcell_SingleComponent.sq_distances_from_ref_general_1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L671>`__

.. code-block:: python

   def sq_distances_from_ref_general_1_(self, r, r_ref, b)

.. rubric:: Docstring

.. code-block:: text

   adapted from vectors_between_atoms_ in O/MM/mm_helper.py


``PermuteUnitcell_SingleComponent.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L700>`__

.. code-block:: python

   def __init__(self, n_atoms_mol, n_mol, n_mol_unitcell, ind_rO, n_images_search=1)

.. rubric:: Docstring

.. code-block:: text

   Configure unit-cell dimensions and periodic image search.

   ``n_mol_unitcell`` is the number of molecules per crystallographic unit
   cell and must divide ``n_mol``. ``ind_rO`` is the within-molecule anchor
   atom. ``n_images_search`` controls the lattice-image cube used for
   general-cell minimum distances.


``PermuteUnitcell_SingleComponent.__call__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L736>`__

.. code-block:: python

   def __call__(self, r, b)

.. rubric:: Docstring

.. code-block:: text

   Augment every trajectory frame and return a new coordinate array.

   ``r`` and ``b`` must contain the same number of frames. Coordinates
   have shape ``(m, N, 3)`` and boxes ``(m, 3, 3)``. When only one unit
   cell is present, the original ``r`` object is returned unchanged.


``PermuteUnitcell_SingleComponent.permute_unitcells_single_frame_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L759>`__

.. code-block:: python

   def permute_unitcells_single_frame_(self, r, box)

.. rubric:: Docstring

.. code-block:: text

   Randomly translate and reassign unit-cell blocks in one frame.

   The returned ``(N, 3)`` array is wrapped relative to molecular anchor
   atoms. Translation by lattice vectors and block relabelling preserve
   the periodic physical configuration.


``PermuteUnitcell_SingleComponent.permute_unitcells_single_frame_.wrap_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/sym.py#L769>`__

.. code-block:: python

   def wrap_(r, b)

.. rubric:: Docstring

.. code-block:: text

   Wrap Cartesian points into the primary cell of ``b``.
