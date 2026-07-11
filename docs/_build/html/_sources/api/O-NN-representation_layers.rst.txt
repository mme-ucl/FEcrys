.. _api-O-NN-representation_layers:

O.NN.representation_layers
==========================

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py>`__

.. rubric:: Docstring

.. code-block:: text

   Invertible coordinate representations for molecular flow models.

   The maps separate molecular translation, rotation, and internal coordinates;
   scale them to model domains; and track log-absolute Jacobian determinants.
   Coordinate tensors use ``(batch, molecule, atom, xyz)`` when molecule and atom
   axes are explicit. The current implementation targets single-component systems
   and molecules with more than three atoms unless a specialised class says
   otherwise.


Classes and functions
---------------------

``SC_helper`` (class)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L22>`__

.. code-block:: python

   class SC_helper(PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   This is longer than needed, mostly as a placholder method for other work.


``SC_helper.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L26>`__

.. code-block:: python

   def __init__(self, PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   Load a single-molecule topology and derive atom metadata.

   The PDB must contain explicit hydrogens. RDKit atom indices, masses,
   heavy/hydrogen masks, adjacency, and atom degrees are cached for later
   coordinate construction.


``SC_helper.set_ABCD_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L52>`__

.. code-block:: python

   def set_ABCD_(self, ind_root_atom=10, option: int=None)

.. rubric:: Docstring

.. code-block:: text

   for a given molecule these inputs are chosen by the user:
   Inputs:
       ind_root_atom : int
           index of atom in the molecule to be treated as the centre of the Cartesian block
       option        : int
           Specifies the other two atoms of the Cartesian block.
               the handy method ZMatrix (from wutobias/r2z) will generate a physical Zmatrix (self.ABCD)
               As any Zmatrix, this matrix starts with 3 atoms that are different from the rest.
               While the ind_root_atom is chosen above, the other two atoms need to be also chosen.
               Try different integeres 0,1,2,3,... for the option, until the option you prefer is printer.
               After trying, each time the text is printed below to explain what happened.
               When what happened looks good on self.mol image (atoms on the image are labeled with the indices),
               remember the option that was made for later work with this molecule.


``SC_helper.get_r_shape_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L130>`__

.. code-block:: python

   def get_r_shape_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Infer a supported coordinate layout and molecule count.

   Returns ``(shape_name, n_mol)`` where the name is ``single_frame``,
   ``flat``, ``atoms``, or ``molecules``. Shapes must be compatible with
   ``n_atoms_mol`` and three Cartesian components.


``SC_helper.r_reshape_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L158>`__

.. code-block:: python

   def r_reshape_(self, r, shape_out='molecules', numpy=True)

.. rubric:: Docstring

.. code-block:: text

   Convert coordinates between flat, atom, and molecule layouts.

   ``numpy`` selects NumPy or TensorFlow reshape helpers. Coordinate order
   and values are unchanged; a single-frame input receives a batch axis.


``SC_helper.b_reshape_m_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L192>`__

.. code-block:: python

   def b_reshape_m_(self, b, m=1, numpy=True, verbose=False)

.. rubric:: Docstring

.. code-block:: text

   Broadcast or validate periodic boxes for ``m`` coordinate frames.

   A vector-like box representation is stacked, while a single ``(3, 3)``
   box with an explicit leading axis is repeated for NVT data. NumPy or
   TensorFlow output is selected by ``numpy``.


``SC_helper.wrap_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L225>`__

.. code-block:: python

   def wrap_(self, r, b, b_inv=None, output_shape=None, numpy=True)

.. rubric:: Docstring

.. code-block:: text

   Wrap each molecule's atoms independently into the primary cell.

   Coordinates are transformed to fractional space, reduced modulo one,
   and transformed back. Returns ``(wrapped_coordinates, [b, b_inv])``;
   the input layout is preserved unless ``output_shape`` is requested.
   This does not preserve whole molecules across boundaries.


``SC_helper.unwrap_molecules_np_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L256>`__

.. code-block:: python

   def unwrap_molecules_np_(self, r, b, output_shape=None)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct whole molecules using the Z-matrix bond tree.

   Starting from the root atom, each atom is placed in the nearest
   periodic image of its already reconstructed reference atom. Output
   layout defaults to the input layout.


``SC_helper.no_jump_molecules_np_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L293>`__

.. code-block:: python

   def no_jump_molecules_np_(self, r, b)

.. rubric:: Docstring

.. code-block:: text

   Remove whole-molecule jumps from a trajectory using the anchor atom.


``SC_helper.unwrap_np_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L301>`__

.. code-block:: python

   def unwrap_np_(self, r, b, output_shape=None)

.. rubric:: Docstring

.. code-block:: text

   Make molecules whole, then remove their frame-to-frame PBC jumps.


``SingleComponent_map`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L321>`__

.. code-block:: python

   class SingleComponent_map(PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   !! : molecule must have >3 atoms to use this M_{IC} layer


``SingleComponent_map.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L323>`__

.. code-block:: python

   def __init__(self, PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   Create an uninitialised crystal coordinate map for one molecule type.


``SingleComponent_map.remove_COM_from_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L330>`__

.. code-block:: python

   def remove_COM_from_data_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Subtract the mean molecular-anchor position from every frame.

   This translation removal ignores periodic boundaries and returns atom
   layout ``(batch, N, 3)`` as float32.


``SingleComponent_map._forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L351>`__

.. code-block:: python

   def _forward_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Convert Cartesian molecules into internal and rigid-body variables.


``SingleComponent_map._inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L358>`__

.. code-block:: python

   def _inverse_(self, X_IC, X_CB)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct atom-layout Cartesian coordinates and map Jacobians.


``SingleComponent_map._first_times_crystal_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L367>`__

.. code-block:: python

   def _first_times_crystal_(self, shape)

.. rubric:: Docstring

.. code-block:: text

   Infer and cache crystal molecule, atom, and degree-of-freedom counts.


``SingleComponent_map._forward_init_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L382>`__

.. code-block:: python

   def _forward_init_(self, r, batch_size=10000)

.. rubric:: Docstring

.. code-block:: text

   Transform an initialisation dataset in batches using eager tensors.

   Returns internal coordinates, rigid-body components, and their two
   Jacobian terms as NumPy arrays.


``SingleComponent_map.reshape_to_unitcells_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L425>`__

.. code-block:: python

   def reshape_to_unitcells_(self, x, combined_unitcells_with_batch_axis=False, forward=True, numpy=False)

.. rubric:: Docstring

.. code-block:: text

   Split or merge the supercell molecule axis into unit-cell blocks.

   When ``combined_unitcells_with_batch_axis`` is true, the unit-cell axis
   is folded into the batch; otherwise it remains explicit. ``forward``
   selects splitting versus reconstruction.


``SingleComponent_map.initalise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L456>`__

.. code-block:: python

   def initalise_(self, r_dataset, b0, batch_size=10000, n_mol_unitcell=1, COM_remover=NotWhitenFlow, focused=True, assert_no_jumping_molecules=True)

.. rubric:: Docstring

.. code-block:: text

   Fit coordinate scalings and base-space metadata from a trajectory.

   Parameters define the Cartesian dataset, fixed box or per-frame boxes,
   batching, molecules per crystallographic unit cell, translation-removal
   transform, and whether focused marginal scaling is used. The historical
   method spelling is retained. This method mutates the map and returns
   ``None``.


``SingleComponent_map.set_periodic_mask_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L550>`__

.. code-block:: python

   def set_periodic_mask_(self)

.. rubric:: Docstring

.. code-block:: text

   Assemble the per-molecule periodic-variable mask in model order.


``SingleComponent_map.flexible_torsions`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L561>`__

.. code-block:: python

   def flexible_torsions(self)

.. rubric:: Docstring

.. code-block:: text

   Internal-coordinate rows classified as fully periodic torsions.


``SingleComponent_map.current_masks_periodic_torsions_and_Phi`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L573>`__

.. code-block:: python

   def current_masks_periodic_torsions_and_Phi(self)

.. rubric:: Docstring

.. code-block:: text

   for match_topology_ when different instances of this obj are used for the same set of coupling layers (multimap)


``SingleComponent_map.match_topology_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L581>`__

.. code-block:: python

   def match_topology_(self, ic_maps: list)

.. rubric:: Docstring

.. code-block:: text

   for the multimap functionality, when different instances of this obj are used for the same set of coupling layers


``SingleComponent_map.ln_base_C_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L594>`__

.. code-block:: python

   def ln_base_C_(self, z)

.. rubric:: Docstring

.. code-block:: text

   Return the constant log density for conformation/rotation base variables.


``SingleComponent_map.ln_base_P_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L599>`__

.. code-block:: python

   def ln_base_P_(self, z)

.. rubric:: Docstring

.. code-block:: text

   Return the constant log density for translational base variables.


``SingleComponent_map.ln_base_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L604>`__

.. code-block:: python

   def ln_base_(self, inputs)

.. rubric:: Docstring

.. code-block:: text

   Return the sum of positional and conformational base log densities.


``SingleComponent_map.sample_base_C_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L608>`__

.. code-block:: python

   def sample_base_C_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Sample ``m`` conformation/rotation vectors uniformly on [-1, 1].


``SingleComponent_map.sample_base_P_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L613>`__

.. code-block:: python

   def sample_base_P_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Sample ``m`` translation vectors uniformly on [-1, 1].


``SingleComponent_map.sample_base_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L618>`__

.. code-block:: python

   def sample_base_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Sample positional and conformation/rotation base variables.


``SingleComponent_map.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L625>`__

.. code-block:: python

   def forward_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Map Cartesian crystal coordinates to scaled flow variables.

   Returns ``[xO, X]`` and a ``(batch, 1)``-compatible log-Jacobian.
   ``xO`` contains translation variables (or raw anchors for variable-box
   data); ``X`` contains bonds, angles, torsions, and rotations.


``SingleComponent_map.sample_nvt_h_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L725>`__

.. code-block:: python

   def sample_nvt_h_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Repeat the fixed NVT box representation ``m`` times.


``SingleComponent_map.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L730>`__

.. code-block:: python

   def inverse_(self, variables_in)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct Cartesian coordinates from translation/internal variables.


``SingleComponent_map.xO_reshape_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L796>`__

.. code-block:: python

   def xO_reshape_(self, x, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Insert or remove the dummy fixed translation used by PGMcrys v2.


``SingleComponent_map.flow_mask_xO`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L817>`__

.. code-block:: python

   def flow_mask_xO(self)

.. rubric:: Docstring

.. code-block:: text

   Translation flow mask with the dummy first molecule fixed to zero.


``SingleComponent_map.flow_mask_X`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L827>`__

.. code-block:: python

   def flow_mask_X(self)

.. rubric:: Docstring

.. code-block:: text

   All-ones mask for molecular internal and rotational variables.


``SingleMolecule_map`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L836>`__

.. code-block:: python

   class SingleMolecule_map(PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   !! : molecule must have >3 atoms to use this M_{IC} layer


``SingleMolecule_map.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L838>`__

.. code-block:: python

   def __init__(self, PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   Create a coordinate map specialised to one isolated molecule.


``SingleMolecule_map._forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L847>`__

.. code-block:: python

   def _forward_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Extract internal coordinates and rotation-free three-atom geometry.


``SingleMolecule_map._inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L854>`__

.. code-block:: python

   def _inverse_(self, X_IC, X_CB)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct an isolated molecule from internal and anchor geometry.


``SingleMolecule_map._forward_init_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L863>`__

.. code-block:: python

   def _forward_init_(self, r, batch_size=10000)

.. rubric:: Docstring

.. code-block:: text

   Batch-transform an isolated-molecule dataset for scaler fitting.


``SingleMolecule_map.initalise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L898>`__

.. code-block:: python

   def initalise_(self, r_dataset, b0=None, batch_size=10000, n_mol_unitcell=None, COM_remover=None, focused=True)

.. rubric:: Docstring

.. code-block:: text

   Fit isolated-molecule bond, angle, and torsion transformations.

   Box, unit-cell, and translation-removal arguments are accepted only for
   API compatibility and are unused. The dataset must contain exactly one
   molecule per frame.


``SingleMolecule_map.set_periodic_mask_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L961>`__

.. code-block:: python

   def set_periodic_mask_(self)

.. rubric:: Docstring

.. code-block:: text

   Assemble periodic flags for isolated-molecule internal variables.


``SingleMolecule_map.current_masks_periodic_torsions_and_Phi`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L973>`__

.. code-block:: python

   def current_masks_periodic_torsions_and_Phi(self)

.. rubric:: Docstring

.. code-block:: text

   for match_topology_ when different instances of this obj are used for the same set of coupling layers (multimap)


``SingleMolecule_map.match_topology_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L981>`__

.. code-block:: python

   def match_topology_(self, ic_maps: list)

.. rubric:: Docstring

.. code-block:: text

   for the multimap functionality, when different instances of this obj are used for the same set of coupling layers


``SingleMolecule_map.sample_base_P_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L996>`__

.. code-block:: python

   def sample_base_P_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Return ``None`` because isolated-molecule translations are removed.


``SingleMolecule_map.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1003>`__

.. code-block:: python

   def forward_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Map isolated Cartesian coordinates to scaled internal variables.

   Returns ``[None, X]`` plus the complete coordinate/scaling log-Jacobian.


``SingleMolecule_map.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1069>`__

.. code-block:: python

   def inverse_(self, variables_in)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct isolated Cartesian coordinates from ``[None, X]``.


``SingleComponent_map_r`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1118>`__

.. code-block:: python

   class SingleComponent_map_r(PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   rO atoms

   a small udpate for the NVT model (PGMcrys_v1) to allow jumping whole molecules in the fixed periodic box

   SingleComponent_map parent class here still deals with all other atoms (not rO), as in original NVT version

   rO steps here are mostly copied from the more general case of the NPT model (seperate in pgm_rb.py)


``SingleComponent_map_r.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1133>`__

.. code-block:: python

   def __init__(self, PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   Create an NVT map that permits whole molecules to cross boundaries.


``SingleComponent_map_r.initalise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1147>`__

.. code-block:: python

   def initalise_(self, r_dataset, b0, batch_size=10000, n_mol_unitcell=1, COM_remover='blank', focused='blank', whiten_setting=0)

.. rubric:: Docstring

.. code-block:: text

   Fit an NVT periodic-position representation and inherited internals.

   ``b0`` is treated as one fixed box; if a box trajectory is supplied,
   only its first box is used. Translation invariance removes one molecular
   anchor. ``whiten_setting=1`` additionally whitens the remaining scaled
   fractional positions.


``SingleComponent_map_r.white_setting_0_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1230>`__

.. code-block:: python

   def white_setting_0_(self, xO, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Identity positional whitening with zero log-Jacobian.


``SingleComponent_map_r.white_setting_1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1234>`__

.. code-block:: python

   def white_setting_1_(self, xO, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Apply or invert fitted positional whitening and range scaling.


``SingleComponent_map_r.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1265>`__

.. code-block:: python

   def forward_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Map NVT coordinates to periodic translations and internal variables.


``SingleComponent_map_r.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/representation_layers.py#L1299>`__

.. code-block:: python

   def inverse_(self, variables)

.. rubric:: Docstring

.. code-block:: text

   Invert periodic translations and internals into NVT coordinates.
