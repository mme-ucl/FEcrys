.. _api-O-MM-ff_setup:

O.MM.ff_setup
=============

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py>`__

.. rubric:: Docstring

.. code-block:: text

   Prepare molecular force fields and reconcile topology atom ordering.

   This module edits GROMACS topology fragments, builds OpenMM systems through
   ParmEd, and provides force-field variants used by :class:`SingleComponent`.
   Coordinate arrays exposed to FEcrys retain the input-PDB atom order even when a
   loaded topology uses a different order; :class:`methods_for_permutation`
   performs that translation at the OpenMM boundary.


Classes and functions
---------------------

``change_charges_itp_top_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L15>`__

.. code-block:: python

   def change_charges_itp_top_(path_top_or_itp_file_in: str, path_top_or_itp_file_out: str, n_atoms_mol: int, replacement_charges: np.ndarray=None, neutralise_charge: bool=True)

.. rubric:: Docstring

.. code-block:: text

   Write a topology copy with modified per-atom partial charges.

   Parameters
   ----------
   path_top_or_itp_file_in : str
       Input GROMACS ``.top`` or ``.itp`` file containing an ``[ atoms ]``
       section.
   path_top_or_itp_file_out : str
       Destination file. Existing content is overwritten.
   n_atoms_mol : int
       Number of atom records to read from the first ``[ atoms ]`` section.
   replacement_charges : numpy.ndarray, optional
       Explicit charges in topology atom order, in elementary-charge units.
       Its length must equal ``n_atoms_mol``.
   neutralise_charge : bool, optional
       When no replacement is supplied, subtract the mean atomic charge so
       the molecule's total charge is zero. If false, copy charges unchanged.

   Returns
   -------
   numpy.ndarray
       Charges written to the output file, in elementary-charge units.

   Notes
   -----
   The routine performs textual replacement and assumes the charge and mass
   occupy the conventional columns of the GROMACS atom records.


``change_n_mol_top_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L114>`__

.. code-block:: python

   def change_n_mol_top_(path_top_file_in: str, path_top_file_out: str, replace_n_mol: int, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Write a topology copy with a new molecule count.

   Parameters
   ----------
   path_top_file_in : str
       Input GROMACS topology containing one entry in ``[ molecules ]``.
   path_top_file_out : str
       Destination topology. Existing content is overwritten.
   replace_n_mol : int
       New molecule count for the first non-comment entry in the section.
   verbose : bool, optional
       Print the replaced line and output path.

   Returns
   -------
   None
       The result is written to ``path_top_file_out``.


``methods_for_permutation`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L182>`__

.. code-block:: python

   class methods_for_permutation

.. rubric:: Docstring

.. code-block:: text

   Adapt simulation access when topology and PDB atom orders differ.

   This mixin is injected into a force-field object only when a non-identity
   atom permutation is detected. OpenMM positions are permuted on input and
   restored to the FEcrys/PDB order on output.


``methods_for_permutation._current_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L194>`__

.. code-block:: python

   def _current_r_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current positions in PDB order, in nanometres.


``methods_for_permutation._current_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L201>`__

.. code-block:: python

   def _current_v_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current velocities in PDB order, in nm/ps.


``methods_for_permutation._current_F_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L208>`__

.. code-block:: python

   def _current_F_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current forces in PDB order, in kJ mol⁻¹ nm⁻¹.


``methods_for_permutation._system_mass_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L215>`__

.. code-block:: python

   def _system_mass_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Particle masses in PDB order, in daltons.


``methods_for_permutation._set_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L220>`__

.. code-block:: python

   def _set_r_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Set OpenMM positions from a PDB-ordered coordinate array.

   Parameters
   ----------
   r : numpy.ndarray
       Cartesian coordinates shaped ``(N, 3)`` in nanometres.


``methods_for_permutation._set_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L231>`__

.. code-block:: python

   def _set_v_(self, v)

.. rubric:: Docstring

.. code-block:: text

   Set OpenMM velocities from a PDB-ordered array.

   Parameters
   ----------
   v : numpy.ndarray
       Velocities shaped ``(N, 3)`` in nanometres per picosecond.


``methods_for_permutation.forward_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L242>`__

.. code-block:: python

   def forward_atom_index_(self, inds)

.. rubric:: Docstring

.. code-block:: text

   Map PDB-order atom indices to indices used by OpenMM.

   Parameters
   ----------
   inds : int or array_like of int
       Atom indices in the public FEcrys/PDB ordering.

   Returns
   -------
   int or numpy.ndarray
       Corresponding topology-order indices.


``methods_for_permutation.inverse_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L258>`__

.. code-block:: python

   def inverse_atom_index_(self, inds)

.. rubric:: Docstring

.. code-block:: text

   Map OpenMM topology indices back to the PDB atom ordering.


``methods_for_permutation.set_arrays_blank_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L265>`__

.. code-block:: python

   def set_arrays_blank_(self)

.. rubric:: Docstring

.. code-block:: text

   Reset all in-memory trajectory buffers and the saved-frame count.


``methods_for_permutation.save_frame_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L277>`__

.. code-block:: python

   def save_frame_(self)

.. rubric:: Docstring

.. code-block:: text

   Append the current OpenMM state to trajectory buffers.

   Positions are temporarily retained in topology order; reduced potential
   energy, temperature, box vectors, and centre of mass are saved in their
   standard FEcrys units.


``methods_for_permutation.run_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L293>`__

.. code-block:: python

   def run_simulation_(self, n_saves, stride_save_frame: int=100, verbose_info: str='')

.. rubric:: Docstring

.. code-block:: text

   Advance a simulation and save frames in public PDB atom order.

   Parameters
   ----------
   n_saves : int
       Number of trajectory frames to append.
   stride_save_frame : int, optional
       OpenMM integration steps between saved frames.
   verbose_info : str, optional
       Text appended to the live progress message.

   Notes
   -----
   Topology-ordered positions are converted only after the requested block
   finishes. Interrupting the loop can therefore lose coordinates from the
   unfinished block even though other frame data were appended.


``methods_for_permutation.run_simulation_w_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L322>`__

.. code-block:: python

   def run_simulation_w_(self, n_saves, stride_save_frame: int=100, verbose_info: str='')

.. rubric:: Docstring

.. code-block:: text

   Run and save frames after wrapping molecules into the periodic box.

   Parameters are identical to :meth:`run_simulation_`. Before every save,
   OpenMM positions are requested with ``enforcePeriodicBox=True`` and set
   back into the context, then converted from topology to PDB atom order.


``methods_for_permutation.u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L342>`__

.. code-block:: python

   def u_(self, r, b=None)

.. rubric:: Docstring

.. code-block:: text

   Evaluate reduced potential energies for a batch of structures.

   Parameters
   ----------
   r : numpy.ndarray
       PDB-ordered coordinates shaped ``(frames, N, 3)`` in nanometres.
   b : numpy.ndarray, optional
       Per-frame box matrices shaped ``(frames, 3, 3)`` in nanometres. If
       omitted, the context's current box is used for every frame.

   Returns
   -------
   numpy.ndarray
       Reduced potential energies ``beta * U`` shaped ``(frames, 1)``.

   Notes
   -----
   The original context positions and box are restored before returning.


``itp2FF`` (class)
^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L386>`__

.. code-block:: python

   class itp2FF()

.. rubric:: Docstring

.. code-block:: text

   Base class for force fields loaded from GROMACS ``.itp`` files.

   Subclasses define force-field naming, GROMACS defaults, and optional
   post-loading corrections. Generated support files live in ``misc_dir`` and
   are named from the molecular-system ``name``.


``itp2FF.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L394>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Initialise the shared MM helper and select the GROMACS loader.


``itp2FF.itp_mol`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L400>`__

.. code-block:: python

   def itp_mol(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Input molecule include-topology file.


``itp2FF.itp_mol_adjusted_charges`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L405>`__

.. code-block:: python

   def itp_mol_adjusted_charges(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Generated topology with adjusted partial charges.


``itp2FF.top_crys`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L411>`__

.. code-block:: python

   def top_crys(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Generated crystal-level GROMACS topology.


``itp2FF.gro_mol`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L416>`__

.. code-block:: python

   def gro_mol(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Molecule coordinates associated with the topology.


``itp2FF.pdb_mol`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L421>`__

.. code-block:: python

   def pdb_mol(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: PDB converted from the topology's GRO coordinates.


``itp2FF.single_mol_pdb`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L426>`__

.. code-block:: python

   def single_mol_pdb(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Single-molecule PDB extracted from the input crystal.


``itp2FF._single_mol_pdb_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L430>`__

.. code-block:: python

   def _single_mol_pdb_file_(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Compatibility alias for :attr:`single_mol_pdb`.


``itp2FF.single_mol_pdb_permuted`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L435>`__

.. code-block:: python

   def single_mol_pdb_permuted(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Single-molecule PDB reordered to topology atom order.


``itp2FF.single_mol_permutations`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L440>`__

.. code-block:: python

   def single_mol_permutations(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Pickle containing forward and inverse atom permutations.


``itp2FF.set_pemutation_to_match_topology_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L444>`__

.. code-block:: python

   def set_pemutation_to_match_topology_(self)

.. rubric:: Docstring

.. code-block:: text

   Determine and install the PDB-to-topology atom permutation.

   A cached permutation is loaded when available. Otherwise the topology
   GRO file is converted to PDB, reordered against the input molecule, and
   matched by a Cartesian distance matrix. Molecular permutations are then
   tiled over ``n_mol`` to form whole-crystal mappings.

   Notes
   -----
   If the resulting mapping is non-identity, methods from
   :class:`methods_for_permutation` are injected into the instance so all
   public coordinates remain in PDB order. The algorithm assumes every
   crystal molecule has the same internal atom ordering as the first.


``itp2FF.a_step_after_initialise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L514>`__

.. code-block:: python

   def a_step_after_initialise_(self)

.. rubric:: Docstring

.. code-block:: text

   Extension hook executed after topology files and charges are prepared.


``itp2FF.initialise_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L518>`__

.. code-block:: python

   def initialise_FF_(self, neuralise_net_charge=False, replacement_charges=None)

.. rubric:: Docstring

.. code-block:: text

   Prepare topology files after molecular counts have been defined.

   Parameters
   ----------
   neuralise_net_charge : bool, optional
       Subtract the mean molecular partial charge before topology loading.
       The historical parameter name is retained for compatibility.
   replacement_charges : array_like, optional
       Explicit per-atom charges in topology order. Supplying this also
       selects the adjusted-charge topology.

   Notes
   -----
   ``n_mol`` and ``n_atoms_mol`` must already be set by the molecular-system
   constructor. The method extracts a single-molecule PDB if necessary,
   determines atom permutation, and runs the subclass extension hook.


``itp2FF.set_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L559>`__

.. code-block:: python

   def set_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   Generate the crystal topology and load it with ParmEd.

   This method is called immediately before OpenMM system construction and
   sets ``self.ff`` to a :class:`parmed.gromacs.GromacsTopologyFile`.


``itp2FF.reset_n_mol_top_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L569>`__

.. code-block:: python

   def reset_n_mol_top_(self)

.. rubric:: Docstring

.. code-block:: text

   Rewrite the crystal topology for the current molecule count.

   The generated file includes either the original or charge-adjusted ITP,
   followed by the subclass-specific defaults, system, and molecule
   sections required by the GROMACS topology loader.


``OPLS_general`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L601>`__

.. code-block:: python

   class OPLS_general()

.. rubric:: Docstring

.. code-block:: text

   Generic OPLS topology loader with geometric Lennard-Jones mixing.


``OPLS_general.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L604>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Configure OPLS GROMACS defaults and generic topology labels.


``OPLS_general.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L615>`__

.. code-block:: python

   def FF_name(self)

.. rubric:: Docstring

.. code-block:: text

   str: Public force-field identifier ``'OPLS'``.


``GAFF_general`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L619>`__

.. code-block:: python

   class GAFF_general()

.. rubric:: Docstring

.. code-block:: text

   Generic GAFF topology loader with Lorentz-Berthelot mixing.


``GAFF_general.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L622>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Configure GAFF scaling factors and generic topology labels.


``GAFF_general.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L633>`__

.. code-block:: python

   def FF_name(self)

.. rubric:: Docstring

.. code-block:: text

   str: Public force-field identifier ``'GAFF'``.


``remove_force_by_names_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L639>`__

.. code-block:: python

   def remove_force_by_names_(system, names: list, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Remove every OpenMM force whose name is in ``names``.

   Parameters
   ----------
   system : openmm.System
       System modified in place.
   names : list of str
       Force names to remove. Repeated forces with the same name are all
       removed.
   verbose : bool, optional
       Report how many forces were removed.

   Returns
   -------
   None


``remove_force_by_names_.remove_force_by_name_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L657>`__

.. code-block:: python

   def remove_force_by_name_(_name)

.. rubric:: Docstring

.. code-block:: text

   Remove the first force named ``_name`` and report its name.


``_get_pairs_mol_inner_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L674>`__

.. code-block:: python

   def _get_pairs_mol_inner_(single_mol_pdb_file, n=3)

.. rubric:: Docstring

.. code-block:: text

   Mark intramolecular atom pairs separated by fewer than ``n`` atoms.

   Parameters
   ----------
   single_mol_pdb_file : str or pathlib.Path
       Molecule PDB in the same atom order as the topology.
   n : int, optional
       Maximum number of atoms in the shortest graph path to mark. Because a
       path of ``n`` atoms contains ``n - 1`` bonds, ``n=3`` marks 1–2 and 1–3
       pairs while retaining 1–4 nonbonded interactions.

   Returns
   -------
   numpy.ndarray
       Square molecular mask where one means the pair is internal/excluded and
       zero means it remains eligible for nonbonded interaction.


``_get_pairs_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L721>`__

.. code-block:: python

   def _get_pairs_(remove_mol_ij, n_mol)

.. rubric:: Docstring

.. code-block:: text

   Tile a molecular exclusion mask across a crystal.

   Parameters
   ----------
   remove_mol_ij : numpy.ndarray
       Square single-molecule mask; one denotes an excluded intramolecular pair.
   n_mol : int
       Number of identical molecules in the crystal.

   Returns
   -------
   numpy.ndarray
       ``(N, N)`` mask where one denotes an included nonbonded pair and zero an
       excluded pair. Intermolecular pairs are always included.


``custom_LJ_force_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L748>`__

.. code-block:: python

   def custom_LJ_force_(sc, C6_C12_types_dictionary)

.. rubric:: Docstring

.. code-block:: text

   Construct a tabulated all-pairs Lennard-Jones force.

   Parameters
   ----------
   sc : SingleComponent
       Initialised molecular system providing topology atom types, cutoff
       settings, particle count, and molecule PDB.
   C6_C12_types_dictionary : dict
       Mapping ``(atom_type_A, atom_type_B)`` to ``(C6, C12)`` parameters in
       the units expected by OpenMM's reduced Lennard-Jones expression.

   Returns
   -------
   list of openmm.CustomNonbondedForce
       One force with intramolecular 1–2 and 1–3 exclusions. Periodic crystals
       use a switched cutoff and long-range correction; isolated molecules use
       no cutoff.


``custom_C_force_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L834>`__

.. code-block:: python

   def custom_C_force_(sc)

.. rubric:: Docstring

.. code-block:: text

   Construct the Coulombic complement to :func:`custom_LJ_force_`.

   Parameters
   ----------
   sc : SingleComponent
       Initialised system providing molecular charges and nonbonded settings.

   Returns
   -------
   list of openmm.NonbondedForce
       One charge-only force with zero Lennard-Jones parameters and molecular
       1–2/1–3 exceptions. Its nonbonded method follows the system's original
       ``NonbondedForce``.


``tmFF`` (class)
^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L924>`__

.. code-block:: python

   class tmFF()

.. rubric:: Docstring

.. code-block:: text

   Load tailor-made pair parameters not supported by standard mixing rules.

   ``[ nonbond_params ]`` supplies explicit C6/C12 values for every unordered
   atom-type pair. The automatically generated OpenMM nonbonded forces are
   replaced by separate tabulated Lennard-Jones and Coulombic forces.


``tmFF.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L932>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Configure topology names and unit scaling factors for ``tmFF``.


``tmFF.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L943>`__

.. code-block:: python

   def FF_name(self)

.. rubric:: Docstring

.. code-block:: text

   str: Public force-field identifier ``'tmFF'``.


``tmFF.a_step_after_initialise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L947>`__

.. code-block:: python

   def a_step_after_initialise_(self)

.. rubric:: Docstring

.. code-block:: text

   Read and validate explicit C6/C12 atom-type pair parameters.

   The ``[ nonbond_params ]`` section of :attr:`itp_mol` is parsed into
   ``nonbond_params``. The number of records must equal the triangular
   number ``n_types * (n_types + 1) / 2``, ensuring every unordered type
   pair has an explicit parameter set.


``tmFF.recast_NB_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L983>`__

.. code-block:: python

   def recast_NB_(self, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Replace automatically loaded nonbonded forces with custom forces.

   Parameters
   ----------
   verbose : bool, optional
       Report removed and added OpenMM force names.

   Notes
   -----
   The system is modified in place. Existing ``CustomNonbondedForce`` and
   ``NonbondedForce`` instances are removed before the tabulated
   Lennard-Jones and charge-only replacements are added.


``tmFF.corrections_to_ff_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L1011>`__

.. code-block:: python

   def corrections_to_ff_(self, verbos=True)

.. rubric:: Docstring

.. code-block:: text

   Apply tailor-made nonbonded corrections after system construction.

   Parameters
   ----------
   verbos : bool, optional
       Historical spelling of the verbosity flag passed to
       :meth:`recast_NB_`.


``velff`` (class)
^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L1022>`__

.. code-block:: python

   class velff()

.. rubric:: Docstring

.. code-block:: text

   Veliparib-specific name-compatible subclass of :class:`tmFF`.

   The separate class is retained so historical pickled simulations continue
   to resolve their original class path.


``velff.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L1029>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Configure veliparib topology identifiers and GROMACS defaults.


``velff.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L1040>`__

.. code-block:: python

   def FF_name(self)

.. rubric:: Docstring

.. code-block:: text

   str: Public force-field identifier ``'velff'``.
