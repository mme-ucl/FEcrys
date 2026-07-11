.. _api-O-MM-sc_system:

O.MM.sc_system
==============

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py>`__

.. rubric:: Docstring

.. code-block:: text

   Construct, simulate, save, and restore single-component molecular systems.

   ``SingleComponent`` combines a common OpenMM lifecycle with a dynamically
   selected force-field mixin. Public coordinate arrays contain only physical
   atoms, use nanometres, and retain the input-PDB atom order; individual mixins
   translate to topology order or add virtual sites at the OpenMM boundary.


Classes and functions
---------------------

``SingleComponent`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L18>`__

.. code-block:: python

   class SingleComponent(PDB: str, n_atoms_mol: int, name: str, FF_name: str='GAFF', atom_order_PDB_match_itp=False, FF_class=None)

.. rubric:: Docstring

.. code-block:: text

   Represent a crystal or isolated system containing one molecular species.

   The class owns molecular metadata, an OpenMM ``System`` and ``Simulation``,
   trajectory buffers, and the arguments needed to reconstruct a saved run.
   A force-field mixin is selected dynamically by :meth:`__new__`.


``SingleComponent.__new__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L26>`__

.. code-block:: python

   def __new__(cls, PDB, n_atoms_mol, name, FF_name='GAFF', atom_order_PDB_match_itp=False, FF_class=None)

.. rubric:: Docstring

.. code-block:: text

   Create a runtime subclass combining the system and force-field APIs.

   Parameters
   ----------
   PDB, n_atoms_mol, name
       Forwarded to :meth:`__init__`; they are accepted here because object
       construction uses the public ``SingleComponent`` signature.
   FF_name : {'GAFF', 'OPLS', 'TIP4P'}, optional
       Built-in force-field mixin selected when ``FF_class`` is omitted.
   atom_order_PDB_match_itp : bool, optional
       Deprecated compatibility argument; atom ordering is detected later.
   FF_class : type, optional
       Custom mixin implementing at least ``initialise_FF_`` and ``set_FF_``.

   Returns
   -------
   SingleComponent
       Instance of a generated subclass whose method resolution order also
       includes the selected force-field class.


``SingleComponent.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L63>`__

.. code-block:: python

   def __init__(self, PDB: str, n_atoms_mol: int, name: str, FF_name: str='GAFF', atom_order_PDB_match_itp=False, FF_class=None)

.. rubric:: Docstring

.. code-block:: text

   Load molecular geometry and prepare force-field input files.

   Parameters
   ----------
   PDB : str
       Crystal or isolated-molecule PDB including hydrogens and periodic
       box metadata. Coordinates are interpreted in nanometres by MDTraj.
   n_atoms_mol : int
       Number of physical atoms per molecule, including hydrogens.
   name : str
       Molecular-system identifier used in generated filenames.
   FF_name : {'GAFF', 'OPLS', 'TIP4P'}, optional
       Built-in force field; ignored when ``FF_class`` is supplied.
   atom_order_PDB_match_itp : bool, optional
       Deprecated and retained only for reconstruction of historical saves.
   FF_class : type, optional
       Custom force-field mixin selected during :meth:`__new__`.

   Notes
   -----
   The atom count must be an integer multiple of ``n_atoms_mol``. Box
   vectors are converted to OpenMM's reduced form when necessary. This
   stage does not yet construct an OpenMM ``System`` or ``Simulation``.


``SingleComponent.set_rdkit_mol_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L159>`__

.. code-block:: python

   def set_rdkit_mol_(self)

.. rubric:: Docstring

.. code-block:: text

   Create an index-labelled, conformer-free RDKit reference molecule.


``SingleComponent.print`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L166>`__

.. code-block:: python

   def print(self, *text, verbose=False)

.. rubric:: Docstring

.. code-block:: text

   Print diagnostic text under the instance verbosity policy.

   Parameters
   ----------
   *text
       Objects forwarded to the built-in :func:`print`.
   verbose : bool, optional
       Force output even when ``self.verbose`` is false.


``SingleComponent.initialise_system_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L180>`__

.. code-block:: python

   def initialise_system_(self, PME_cutoff: float=None, removeCMMotion: bool=True, nonbondedMethod=app.PME, custom_EwaldErrorTolerance=0.0001, constraints=None, SwitchingFunction_factor=0.95)

.. rubric:: Docstring

.. code-block:: text

   Construct and configure the OpenMM force-field system.

   Parameters
   ----------
   PME_cutoff : float, optional
       Nonbonded cutoff in nanometres. By default, 95% of half the smallest
       diagonal box length is used.
   removeCMMotion : bool, optional
       Add OpenMM centre-of-mass motion removal and subtract three degrees
       of freedom.
   nonbondedMethod : openmm.app nonbonded method, optional
       Method passed to the topology loader. Isolated single molecules are
       forced to ``NoCutoff``.
   custom_EwaldErrorTolerance : float, optional
       Ewald/PME relative force-error tolerance.
   constraints : openmm.app constraint option, optional
       Bond constraints passed to ``createSystem``.
   SwitchingFunction_factor : float, optional
       Switching distance as a fraction of ``PME_cutoff``.

   Notes
   -----
   The method sets ``ff``, ``topology``, ``system``, mass and charge data,
   constraint counts, and ``n_DOF``. Force-field-specific corrections are
   applied last. No integrator or OpenMM context is created here.


``SingleComponent.initialise_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L309>`__

.. code-block:: python

   def initialise_simulation_(self, rbv=None, minimise=True, T=300.0, timestep_ps=0.0005, collision_rate=1, P=None, barostat_type=2, barostat_1_scaling=[True, True, True], stride_barostat=25, custom_integrator=None)

.. rubric:: Docstring

.. code-block:: text

   Create an OpenMM simulation context and initialise its thermodynamic state.

   Parameters
   ----------
   rbv : sequence, optional
       Restart tuple ``(positions, box, velocities)`` in public atom order;
       positions and box use nanometres and velocities use nm/ps. Defaults
       to the input PDB structure and box.
   minimise : bool, optional
       Minimise potential energy after loading the initial state.
   T : float, optional
       Temperature in kelvin.
   timestep_ps : float, optional
       Integration timestep in picoseconds.
   collision_rate : float, optional
       Stored Langevin collision rate in inverse picoseconds. The default
       integrator currently uses ``1 / ps`` directly.
   P : float, optional
       Pressure in atmospheres. ``None`` selects NVT; otherwise a barostat
       is added for NPT simulation.
   barostat_type : int, optional
       Barostat implementation selected by the MM helper.
   barostat_1_scaling : sequence of bool, optional
       Allowed axis scaling for the anisotropic barostat variant.
   stride_barostat : int, optional
       Monte Carlo barostat attempt interval in integration steps.
   custom_integrator : callable, optional
       Factory accepting temperature, collision rate, and timestep OpenMM
       quantities and returning an integrator.

   Notes
   -----
   Sets ``kT`` in kJ/mol and ``beta`` in mol/kJ. When centre-of-mass motion
   removal is enabled, initial positions are translated to zero COM. The
   reduced-energy evaluator defaults to ``u_GPU_`` if a mixin did not
   provide a specialised implementation.


``SingleComponent.save_simulation_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L433>`__

.. code-block:: python

   def save_simulation_data_(self, path_and_name: str)

.. rubric:: Docstring

.. code-block:: text

   Save trajectory data and all reconstruction arguments.

   Parameters
   ----------
   path_and_name : str
       Pickle destination or filename prefix accepted by :func:`save_pickle_`.

   Notes
   -----
   Saved MD fields are positions and COMs in nanometres, box matrices in
   nanometres, reduced potential energies ``beta * U``, and temperatures in
   kelvin. ``rbv`` stores the current position, box, and velocity state for
   resuming. The sampled physical time is
   ``n_frames * stride_save_frame * timestep_ps`` picoseconds.


``SingleComponent.load_simulation_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L468>`__

.. code-block:: python

   def load_simulation_data_(path_and_name)

.. rubric:: Docstring

.. code-block:: text

   Load a saved simulation-data dictionary.

   Parameters
   ----------
   path_and_name : str
       Pickle path or prefix previously passed to
       :meth:`save_simulation_data_`.

   Returns
   -------
   dict
       MD dataset plus object, system, and simulation initialisation args.


``SingleComponent.initialise_from_save_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L485>`__

.. code-block:: python

   def initialise_from_save_(path_and_name: str, resume_simulation=True, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct a system and repopulate its trajectory buffers.

   Parameters
   ----------
   path_and_name : str or dict
       Saved simulation path, or an already loaded simulation dictionary.
   resume_simulation : bool, optional
       Initialise the OpenMM context from saved ``rbv`` state. If false,
       use the original simulation initial conditions while still loading
       trajectory history.
   verbose : bool, optional
       Verbosity assigned before system and simulation initialisation.

   Returns
   -------
   SingleComponent
       Fully reconstructed object with saved frames restored to its
       in-memory buffers.


``concatenate_datasets_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L528>`__

.. code-block:: python

   def concatenate_datasets_(paths_datasets: list, remove_warmup=None, stride=1)

.. rubric:: Docstring

.. code-block:: text

   Concatenate compatible saved MD datasets.

   Parameters
   ----------
   paths_datasets : list of str
       Saved simulation dictionaries in concatenation order.
   remove_warmup : int or None, optional
       Slice start applied independently to every trajectory. ``None`` starts
       from the first frame.
   stride : int, optional
       Frame stride applied after warm-up removal.

   Returns
   -------
   dict
       Combined simulation-data dictionary. Position, COM, box, reduced-energy,
       and temperature arrays are concatenated; reconstruction arguments and
       save stride must match exactly across inputs. The restart state ``rbv``
       is retained from the first dataset.


``LJ`` (class)
^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L583>`__

.. code-block:: python

   class LJ()

.. rubric:: Docstring

.. code-block:: text

   Force-field mixin for monatomic Lennard-Jones systems.


``LJ.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L586>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Initialise common molecular-mechanics helper state.


``LJ.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L592>`__

.. code-block:: python

   def FF_name(self)

.. rubric:: Docstring

.. code-block:: text

   str: Public force-field identifier ``'LJ'``.


``LJ._single_mol_pdb_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L597>`__

.. code-block:: python

   def _single_mol_pdb_file_(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Single-particle PDB used for index inspection.


``LJ.initialise_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L601>`__

.. code-block:: python

   def initialise_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   Validate a monatomic system and create its single-particle PDB.

   Every physical atom must be treated as a separate one-atom molecule.
   The actual particle parameters are expected in the associated ITP file.


``LJ.top_file_gmx_crys`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L615>`__

.. code-block:: python

   def top_file_gmx_crys(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Generated crystal GROMACS topology.


``LJ.reset_n_mol_top_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L619>`__

.. code-block:: python

   def reset_n_mol_top_(self)

.. rubric:: Docstring

.. code-block:: text

   Write a minimal GROMACS topology with the current particle count.


``LJ.set_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L635>`__

.. code-block:: python

   def set_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   Regenerate and load the Lennard-Jones GROMACS topology with ParmEd.


``TIP4P`` (class)
^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L643>`__

.. code-block:: python

   class TIP4P()

.. rubric:: Docstring

.. code-block:: text

   Force-field mixin for three-site input mapped to four-site TIP4P water.

   FEcrys exposes only O, H1, and H2 coordinates. The massless M site is added
   before values enter OpenMM and removed from queried coordinates, velocities,
   forces, and masses.


``TIP4P.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L651>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Initialise common molecular-mechanics helper state.


``TIP4P.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L657>`__

.. code-block:: python

   def FF_name(self)

.. rubric:: Docstring

.. code-block:: text

   str: Public force-field identifier ``'TIP4P'``.


``TIP4P._single_mol_pdb_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L662>`__

.. code-block:: python

   def _single_mol_pdb_file_(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Extracted three-atom water PDB.


``TIP4P.initialise_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L666>`__

.. code-block:: python

   def initialise_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   Validate three physical atoms per water and extract one molecule.


``TIP4P.top_file_gmx_crys`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L680>`__

.. code-block:: python

   def top_file_gmx_crys(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Generated crystal topology including TIP4P-ice files.


``TIP4P.reset_n_mol_top_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L684>`__

.. code-block:: python

   def reset_n_mol_top_(self)

.. rubric:: Docstring

.. code-block:: text

   Write a GROMACS topology containing the current number of waters.


``TIP4P.set_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L701>`__

.. code-block:: python

   def set_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   Regenerate and load the TIP4P-ice GROMACS topology with ParmEd.


``TIP4P.add_v_sites_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L709>`__

.. code-block:: python

   def add_v_sites_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Insert the TIP4P massless M site into water coordinates.

   Parameters
   ----------
   r : numpy.ndarray
       Coordinates shaped ``(N, 3)`` or ``(batch, N, 3)`` with three
       physical sites per molecule, in nanometres.

   Returns
   -------
   numpy.ndarray
       Coordinates with four sites per molecule and the same optional batch
       convention. ``M = O + 0.13458335 * (H1 + H2 - 2 O)``.


``TIP4P.remove_v_sites_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L739>`__

.. code-block:: python

   def remove_v_sites_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Remove the fourth TIP4P site from batched or unbatched arrays.

   The final site of each four-site molecule is discarded. Units are
   unchanged, so the helper is shared by positions, velocities, and forces.


``TIP4P._current_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L761>`__

.. code-block:: python

   def _current_r_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current physical-site positions in nanometres.


``TIP4P._current_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L768>`__

.. code-block:: python

   def _current_v_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current physical-site velocities in nm/ps.


``TIP4P._current_F_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L775>`__

.. code-block:: python

   def _current_F_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current physical-site forces in kJ mol⁻¹ nm⁻¹.


``TIP4P._set_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L781>`__

.. code-block:: python

   def _set_r_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Set physical-site coordinates after constructing M sites.

   Parameters
   ----------
   r : numpy.ndarray
       Three-site coordinates shaped ``(N, 3)`` in nanometres.


``TIP4P._set_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L792>`__

.. code-block:: python

   def _set_v_(self, v)

.. rubric:: Docstring

.. code-block:: text

   Set physical-site velocities after expanding to four sites.


``TIP4P.forward_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L797>`__

.. code-block:: python

   def forward_atom_index_(self, inds)

.. rubric:: Docstring

.. code-block:: text

   Map three-site public atom indices to four-site OpenMM indices.


``TIP4P._system_mass_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L803>`__

.. code-block:: python

   def _system_mass_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Masses of physical sites only, in daltons.


``TIP4P.inverse_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L808>`__

.. code-block:: python

   def inverse_atom_index_(self, inds)

.. rubric:: Docstring

.. code-block:: text

   Report that a general four-site-to-three-site index map is undefined.

   Returns
   -------
   None
       The method currently emits a warning and performs no conversion.


``GAFF`` (class)
^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L820>`__

.. code-block:: python

   class GAFF()

.. rubric:: Docstring

.. code-block:: text

   Force-field mixin that prepares GAFF parameters with AmberTools.

   A single molecule is parameterised by ``antechamber`` and ``parmchk2``;
   ``tleap`` creates an Amber topology that is converted to GROMACS format for
   the supported ParmEd/OpenMM loading route.


``GAFF.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L828>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Initialise common molecular-mechanics helper state.


``GAFF.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L834>`__

.. code-block:: python

   def FF_name(self)

.. rubric:: Docstring

.. code-block:: text

   str: Public force-field identifier ``'GAFF'``.


``GAFF.initialise_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L838>`__

.. code-block:: python

   def initialise_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   Ensure AmberTools inputs exist and build a one-molecule PRMTOP.

   ``n_mol`` and ``n_atoms_mol`` must already be defined. Missing PREPI,
   FRCMOD, and MOL2 files trigger :meth:`first_time_molecule_`; the PRMTOP
   is then regenerated for the GROMACS conversion workflow.


``GAFF._single_mol_pdb_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L859>`__

.. code-block:: python

   def _single_mol_pdb_file_(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Extracted single-molecule PDB.


``GAFF._single_mol_prepi_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L863>`__

.. code-block:: python

   def _single_mol_prepi_file_(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Antechamber PREPI parameter file.


``GAFF._single_mol_frcmod_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L867>`__

.. code-block:: python

   def _single_mol_frcmod_file_(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Parmchk2 force-field modification file.


``GAFF._single_mol_mol2_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L871>`__

.. code-block:: python

   def _single_mol_mol2_file_(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Antechamber MOL2 file with GAFF types and charges.


``GAFF.first_time_molecule_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L875>`__

.. code-block:: python

   def first_time_molecule_(self)

.. rubric:: Docstring

.. code-block:: text

   Generate the initial GAFF molecule files with AmberTools.

   The first molecule is extracted from the crystal PDB. ``antechamber``
   generates PREPI and MOL2 files with AM1-BCC charges, and ``parmchk2``
   generates missing parameters in FRCMOD format.

   Notes
   -----
   External executables must be available on ``PATH``. Their output files
   are written beneath ``misc_dir`` and existing files may be replaced.


``GAFF.can_run_tleap`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L947>`__

.. code-block:: python

   def can_run_tleap(self)

.. rubric:: Docstring

.. code-block:: text

   bool: Whether PREPI, FRCMOD, and MOL2 inputs all exist.


``GAFF.tleap_file`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L958>`__

.. code-block:: python

   def tleap_file(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Generated tleap instruction file.


``GAFF.prmtop_file`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L962>`__

.. code-block:: python

   def prmtop_file(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Amber topology generated by tleap.


``GAFF._inpcrd_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L966>`__

.. code-block:: python

   def _inpcrd_file_(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Amber coordinate file generated alongside PRMTOP.


``GAFF.create_prmtop_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L970>`__

.. code-block:: python

   def create_prmtop_(self)

.. rubric:: Docstring

.. code-block:: text

   Run tleap to create Amber topology and coordinate files.

   With the supported GROMACS loader, the Amber topology contains one
   molecule and is later replicated by editing the GROMACS topology. The
   deprecated Amber loader instead requests ``n_molecules`` copies.


``GAFF.top_file_gmx`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L1010>`__

.. code-block:: python

   def top_file_gmx(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: ParmEd conversion of the one-molecule Amber topology.


``GAFF.top_file_gmx_adjusted_charges`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L1015>`__

.. code-block:: python

   def top_file_gmx_adjusted_charges(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: One-molecule GROMACS topology with neutralised charge.


``GAFF.top_file_gmx_crys`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L1020>`__

.. code-block:: python

   def top_file_gmx_crys(self)

.. rubric:: Docstring

.. code-block:: text

   pathlib.Path: Crystal GROMACS topology with the current molecule count.


``GAFF.reset_n_mol_top_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L1024>`__

.. code-block:: python

   def reset_n_mol_top_(self)

.. rubric:: Docstring

.. code-block:: text

   Regenerate the crystal topology from the neutralised molecule topology.


``GAFF.set_FF_amber_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L1035>`__

.. code-block:: python

   def set_FF_amber_(self)

.. rubric:: Docstring

.. code-block:: text

   Load GAFF directly through OpenMM's deprecated Amber topology route.


``GAFF.set_FF_gmx_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L1040>`__

.. code-block:: python

   def set_FF_gmx_(self)

.. rubric:: Docstring

.. code-block:: text

   Convert, neutralise, replicate, and load the GAFF topology.

   ParmEd converts PRMTOP to GROMACS format on first use. Molecular charges
   are neutralised, the topology molecule count is set to ``n_mol``, and
   the resulting crystal topology is loaded with ParmEd.


``GAFF.set_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L1071>`__

.. code-block:: python

   def set_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   Select the supported GROMACS loader or deprecated Amber loader.
