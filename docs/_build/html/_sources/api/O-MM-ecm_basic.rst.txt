.. _api-O-MM-ecm_basic:

O.MM.ecm_basic
==============

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py>`__

.. rubric:: Docstring

.. code-block:: text

   ecm_basic.py

   ECM:
       f : FE_lambda0_
       c : LAMBDA_0
       c : LAMBDA_SYSTEM
       c : ECM_basic

   simulation parameters for specific systems:
       f : succinic_acid_ARGS_oss
       f : veliparib_ARGS_oss
       f : mivebresib_ARGS_oss


Classes and functions
---------------------

``FE_lambda0_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L21>`__

.. code-block:: python

   def FE_lambda0_(n_bodies, N, V, T, k, mu)

.. rubric:: Docstring

.. code-block:: text

   REF: https://doi.org/10.1063/5.0044833

   Inputs:
       n_bodies : number of molecules in supercell 
       N        : number of atoms in supercell (NB: virtual atoms are not atoms)
       V        : volume of the supercell
       T        : temperature of the canonical ensemble 
       k        : scalar string constant of the harmonic potential (Einstein crystal)
       mu       : two options:
           COM-free simulations         : normalised masses of the atoms := m_{i} / ( \sum_{j}^{N} m_{j} ) ; m_i = mass of i'th atom
           simulations with fixed atoms : any 'one-hot' vector, can be np.array([1])

   Output:
       f_0 = f_C_minus_f_C_CM + f_EC_CM_minus_f_EC + f_EC

           f_EC               : FE of Einstein crystal (EC)
           f_EC_CM_minus_f_EC : FE difference associated with removing COM from the EC
           f_C_minus_f_C_CM   : FE difference associated with removing COM unperturbed crystal (C)
               COM = centres of mass, or one atom

           f_C = (f_C - f_C_CM) + [f_C_CM - f_EC_CM] + (f_EC_CM - f_EC) + f_EC
               = f_0 + [f_C_CM - f_EC_CM] ; [...] via FEP


``LAMBDA_0`` (class)
^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L64>`__

.. code-block:: python

   class LAMBDA_0(n_bodies: int, mu: np.ndarray, V: float, T: float, k: float, R0: np.ndarray, inds_valid: np.ndarray, n_atoms_in_molecule: int)

.. rubric:: Docstring

.. code-block:: text

   Analytical Einstein-crystal reference state at coupling ``lambda=0``.

   The object stores the reference centroid, harmonic spring constant, COM
   convention, and analytical dimensionless free-energy contributions.


``LAMBDA_0.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L70>`__

.. code-block:: python

   def __init__(self, n_bodies: int, mu: np.ndarray, V: float, T: float, k: float, R0: np.ndarray, inds_valid: np.ndarray, n_atoms_in_molecule: int)

.. rubric:: Docstring

.. code-block:: text

   Inputs:
   (n_bodies, N, V, T, k, mu) : same as in FE_lambda0_ above
   R0 : centroid of the Einestein crystal state, not used here 
   inds_valid : was only relevant when v-sites are present (TIP4P in the old version)
   n_atoms_in_molecule : number of atoms in a single molecule


``LAMBDA_0.energy0_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L107>`__

.. code-block:: python

   def energy0_(self, r)

.. rubric:: Docstring

.. code-block:: text

   also was not used, but can compare with openMM (self.lambda_systems[0.0].u_ should evalaute to the same values as this function;
   the input (r) should have the relevant COM already removed if checking this)


``LAMBDA_SYSTEM`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L116>`__

.. code-block:: python

   class LAMBDA_SYSTEM(args_initialise_object, args_initialise_system, args_initialise_simulation, COM_removal_by_fixing_one_atom_index_of_this_atom: int=None, lam=1.0, k_EC=6000.0, stride_save_frame=50, remove_warmup=200)

.. rubric:: Docstring

.. code-block:: text

   One simulated state along the Einstein-crystal coupling path.

   The reduced potential is a linear coupling between the physical crystal and
   a harmonic Einstein crystal: the physical force field is scaled by
   ``lambda`` and the harmonic restraint by ``1 - lambda``. Translational
   freedom is removed either globally or by fixing one atom.


``LAMBDA_SYSTEM.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L125>`__

.. code-block:: python

   def __init__(self, args_initialise_object, args_initialise_system, args_initialise_simulation, COM_removal_by_fixing_one_atom_index_of_this_atom: int=None, lam=1.0, k_EC=6000.0, stride_save_frame=50, remove_warmup=200)

.. rubric:: Docstring

.. code-block:: text

   Construct and initialise one lambda-state simulation.

   Parameters
   ----------
   args_initialise_object, args_initialise_system, args_initialise_simulation : dict
       Arguments forwarded to the three :class:`SingleComponent` setup
       stages. The simulation dictionary is modified to disable its first
       minimisation.
   COM_removal_by_fixing_one_atom_index_of_this_atom : int, optional
       Physical atom index assigned zero mass. If omitted, OpenMM COM
       removal and mass-weighted recentering are used. A non-integer,
       non-``None`` value selects the custom constrained-COM integrator.
   lam : float, optional
       Coupling coordinate in the closed interval ``[0, 1]``.
   k_EC : float, optional
       Einstein-crystal spring constant in kJ mol⁻¹ nm⁻².
   stride_save_frame : int, optional
       Integration steps between saved trajectory frames.
   remove_warmup : int, optional
       Number of initial saved frames discarded as equilibration.

   Notes
   -----
   The centroid ``r0`` is the recentered initial structure. At ``lambda=0``
   an accompanying :class:`LAMBDA_0` provides the analytical reference free
   energy. Reduced-energy evaluations are exposed through ``u_``.


``LAMBDA_SYSTEM.plot_check_harmonic`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L266>`__

.. code-block:: python

   def plot_check_harmonic(self)

.. rubric:: Docstring

.. code-block:: text

   Plot the one-dimensional harmonic Boltzmann factor over the cutoff.


``LAMBDA_SYSTEM.reinitialise_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L276>`__

.. code-block:: python

   def reinitialise_simulation_(self)

.. rubric:: Docstring

.. code-block:: text

   Recreate the OpenMM context and reset positions to the EC centroid.


``LAMBDA_SYSTEM.simulation_timescale`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L282>`__

.. code-block:: python

   def simulation_timescale(self)

.. rubric:: Docstring

.. code-block:: text

   not used here


``LAMBDA_SYSTEM.set_arrays_blank_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L288>`__

.. code-block:: python

   def set_arrays_blank_(self)

.. rubric:: Docstring

.. code-block:: text

   Clear the underlying trajectory buffers.


``LAMBDA_SYSTEM.run_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L292>`__

.. code-block:: python

   def run_simulation_(self, n_saves, verbose_info: str='')

.. rubric:: Docstring

.. code-block:: text

   Sample this lambda state, removing warm-up data once.

   Parameters
   ----------
   n_saves : int
       Number of production frames to append after any equilibration.
   verbose_info : str, optional
       Additional text in the live simulation status.

   Notes
   -----
   Non-fixed systems are recentered before sampling. On the first call,
   ``remove_warmup`` frames are generated and discarded before production.


``LAMBDA_SYSTEM.xyz`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L334>`__

.. code-block:: python

   def xyz(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Saved physical coordinates in nanometres.


``LAMBDA_SYSTEM.u`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L339>`__

.. code-block:: python

   def u(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Saved reduced potential energies, shaped ``(frames, 1)``.


``LAMBDA_SYSTEM.temperature`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L344>`__

.. code-block:: python

   def temperature(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Saved instantaneous temperatures shaped ``(frames, 1)``.


``ECM_basic`` (class)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L350>`__

.. code-block:: python

   class ECM_basic(name, working_dir_folder_name: str, ARGS_oss: list, k_EC=6000.0, COM_removal_by_fixing_one_atom_index_of_this_atom=None, overwrite=False, path_lambda_1_dataset=None)

.. rubric:: Docstring

.. code-block:: text

   Manage an Einstein-crystal coupling path and its BAR/MBAR analysis.

   The workflow owns simulations and datasets at multiple lambda values,
   caches every cross-potential evaluation, adaptively adds or samples states,
   and combines the analytical lambda-zero free energy with numerical
   free-energy differences to obtain the physical crystal free energy.


``ECM_basic.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L359>`__

.. code-block:: python

   def __init__(self, name, working_dir_folder_name: str, ARGS_oss: list, k_EC=6000.0, COM_removal_by_fixing_one_atom_index_of_this_atom=None, overwrite=False, path_lambda_1_dataset=None)

.. rubric:: Docstring

.. code-block:: text

   Initialise a coupling-path workflow and optionally restore saved data.

   Parameters
   ----------
   name : str
       Prefix identifying this polymorph or molecular system.
   working_dir_folder_name : str
       Directory containing per-lambda trajectory arrays and analysis logs.
   ARGS_oss : list of dict
       ``[object_args, system_args, simulation_args]`` used to construct
       every :class:`LAMBDA_SYSTEM`.
   k_EC : float, optional
       Einstein-crystal spring constant in kJ mol⁻¹ nm⁻².
   COM_removal_by_fixing_one_atom_index_of_this_atom : int, optional
       Atom fixed to remove translation; otherwise global mass-weighted COM
       removal is used.
   overwrite : bool, optional
       Skip automatic import of existing datasets and analysis results.
   path_lambda_1_dataset : str, optional
       External physical-state MD dataset used instead of new lambda-one
       sampling.


``ECM_basic.save_lambda_evaluations_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L444>`__

.. code-block:: python

   def save_lambda_evaluations_(self)

.. rubric:: Docstring

.. code-block:: text

   Persist cached cross-potential evaluations for all lambda pairs.


``ECM_basic.import_lambda_evaluations_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L449>`__

.. code-block:: python

   def import_lambda_evaluations_(self)

.. rubric:: Docstring

.. code-block:: text

   Restore cached cross-potential evaluations from the working directory.


``ECM_basic.save_BAR_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L454>`__

.. code-block:: python

   def save_BAR_results_(self)

.. rubric:: Docstring

.. code-block:: text

   Persist the chronological log of cumulative two-state BAR results.


``ECM_basic.import_BAR_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L459>`__

.. code-block:: python

   def import_BAR_results_(self)

.. rubric:: Docstring

.. code-block:: text

   Restore the saved two-state BAR result log.


``ECM_basic.save_mBAR_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L464>`__

.. code-block:: python

   def save_mBAR_results_(self)

.. rubric:: Docstring

.. code-block:: text

   Persist the latest multistate BAR result log.


``ECM_basic.import_mBAR_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L469>`__

.. code-block:: python

   def import_mBAR_results_(self)

.. rubric:: Docstring

.. code-block:: text

   Restore the saved multistate BAR result log.


``ECM_basic.save_usupervised_sample_sizes_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L474>`__

.. code-block:: python

   def save_usupervised_sample_sizes_(self)

.. rubric:: Docstring

.. code-block:: text

   Persist adaptive per-lambda sample-size targets.

   The filename retains the historical ``usupervised`` spelling.


``ECM_basic.import_usupervised_sample_sizes_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L482>`__

.. code-block:: python

   def import_usupervised_sample_sizes_(self)

.. rubric:: Docstring

.. code-block:: text

   Restore adaptive per-lambda sample-size targets.


``ECM_basic.save_inds_rand_lambda1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L487>`__

.. code-block:: python

   def save_inds_rand_lambda1_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Save representative physical-state indices for subset size ``m``.


``ECM_basic.import_inds_rand_lambda1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L492>`__

.. code-block:: python

   def import_inds_rand_lambda1_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Load representative physical-state indices for subset size ``m``.


``ECM_basic.lambdas`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L500>`__

.. code-block:: python

   def lambdas(self)

.. rubric:: Docstring

.. code-block:: text

   list of float: Active coupling values in ascending order.


``ECM_basic.n_lambdas`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L505>`__

.. code-block:: python

   def n_lambdas(self)

.. rubric:: Docstring

.. code-block:: text

   int: Number of active coupling states.


``ECM_basic.lambda_exists_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L509>`__

.. code-block:: python

   def lambda_exists_(self, lam)

.. rubric:: Docstring

.. code-block:: text

   Return whether coupling value ``lam`` is active.


``ECM_basic.add_lambda_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L513>`__

.. code-block:: python

   def add_lambda_(self, lam, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Construct and register a coupling-state simulation.

   Parameters
   ----------
   lam : float
       Coupling value in ``[0, 1]``.
   verbose : bool, optional
       Explain when the state already exists.

   Notes
   -----
   New coordinate, energy, temperature, and plotting entries are initialised
   alongside the :class:`LAMBDA_SYSTEM`. The save stride corresponds to
   approximately 0.1 ps under the configured integration timestep.


``ECM_basic.remove_lambda_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L554>`__

.. code-block:: python

   def remove_lambda_(self, lam, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Remove a state and invalidate cached evaluations involving it.

   The in-memory trajectory metadata and colour are deleted. If a saved
   evaluation cache exists, entries whose key contains ``lam`` are removed
   from that file and the cache is reloaded.


``ECM_basic.is_converged_else_remove_all_unconverged_datasets_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L585>`__

.. code-block:: python

   def is_converged_else_remove_all_unconverged_datasets_(self, remove=True)

.. rubric:: Docstring

.. code-block:: text

   Check sampled-energy convergence for every lambda state.

   Parameters
   ----------
   remove : bool, optional
       Remove states failing :class:`TestConverged_1D`. An externally
       supplied physical-state dataset is never allowed to be removed.

   Returns
   -------
   bool
       True only when all active self-potential energy series pass.


``ECM_basic.run_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L622>`__

.. code-block:: python

   def run_(self, lam: float, n_saves: int)

.. rubric:: Docstring

.. code-block:: text

   Sample, save, and reload one lambda-state dataset.

   Parameters
   ----------
   lam : float
       State to run; it is constructed if absent.
   n_saves : int
       Number of production frames requested from its simulation.


``ECM_basic.last_m_frames_converged_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L640>`__

.. code-block:: python

   def last_m_frames_converged_(self, lam, m, average_temperature_error_allowed=2.0)

.. rubric:: Docstring

.. code-block:: text

   Test whether the trailing cumulative temperatures remain near target.

   Parameters
   ----------
   lam : float
       Coupling state to inspect.
   m : int
       Number of final cumulative-average values required.
   average_temperature_error_allowed : float, optional
       Maximum absolute deviation from target temperature in kelvin.


``ECM_basic.run_to_get_m_coverged_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L666>`__

.. code-block:: python

   def run_to_get_m_coverged_(self, lam, m=5000, average_temperature_error_allowed=2.0)

.. rubric:: Docstring

.. code-block:: text

   Sample in blocks until ``m`` temperature-converged frames are available.

   Sampling stops after at most ``m`` newly requested frames. The outcome is
   stored in ``lambda_dataset_converged[lam]``.


``ECM_basic.lambda_sample_sizes_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L686>`__

.. code-block:: python

   def lambda_sample_sizes_(self, lam=None, m=None)

.. rubric:: Docstring

.. code-block:: text

   Get or set adaptive sample-size targets.

   With ``m=None``, return the target for ``lam``. Otherwise set one state,
   or all active states when ``lam=None``, to ``m``.


``ECM_basic.which_lambda_to_add_next_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L700>`__

.. code-block:: python

   def which_lambda_to_add_next_(self, max_dataset_size_per_lambda)

.. rubric:: Docstring

.. code-block:: text

   Prioritise a new midpoint and existing states using BAR uncertainty.

   Returns the midpoint of the adjacent pair with largest finite-adjusted
   error, that pair itself, and existing lambdas ordered by accumulated
   neighbouring error while below the sample-size ceiling.


``ECM_basic.unsupervised_FE_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L747>`__

.. code-block:: python

   def unsupervised_FE_(self, batch_size_increments=10000, max_dataset_size_per_lambda=50000, max_n_lambdas=30, SE_tol_per_molecule=0.03125, re_evaluate=False, rerun_questionable_data=False)

.. rubric:: Docstring

.. code-block:: text

   Adaptively sample lambda states until the BAR uncertainty target is met.

   Parameters
   ----------
   batch_size_increments : int, optional
       Frames added when creating or extending a state.
   max_dataset_size_per_lambda : int, optional
       Per-state sampling ceiling.
   max_n_lambdas : int, optional
       Maximum number of coupling states, including endpoints.
   SE_tol_per_molecule : float, optional
       Target standard error per molecule in reduced ``kT`` units.
   re_evaluate : bool, optional
       Clear cached cross-potential evaluations before BAR calculations.
   rerun_questionable_data : bool, optional
       Remove energy-series convergence failures and attempt resampling, up
       to three adaptive passes.

   Notes
   -----
   Endpoint datasets are filled first. New midpoint states are introduced
   at the least-overlapping interval, after which sampling is added to
   existing states in uncertainty-priority order.


``ECM_basic.unsupervised_FE_.main_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L802>`__

.. code-block:: python

   def main_()

.. rubric:: Docstring

.. code-block:: text

   Run one adaptive state-addition and sample-enrichment pass.


``ECM_basic.save_simulations_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L853>`__

.. code-block:: python

   def save_simulations_(self, which_lambdas: list=None)

.. rubric:: Docstring

.. code-block:: text

   Append newly sampled frames to per-lambda files.

   Parameters
   ----------
   which_lambdas : list of float, optional
       States to save; defaults to every active lambda.

   Notes
   -----
   Coordinates, reduced energies, and temperatures are stored separately.
   In-memory simulation buffers are cleared after a successful append.


``ECM_basic.which_lambdas_exist_in_folder`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L891>`__

.. code-block:: python

   def which_lambdas_exist_in_folder(self)

.. rubric:: Docstring

.. code-block:: text

   list of float or None: Lambda values inferred from saved filenames.


``ECM_basic.load_dataset_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L907>`__

.. code-block:: python

   def load_dataset_(self, lam: float, verbose=False, custom_generic_name=None)

.. rubric:: Docstring

.. code-block:: text

   Load coordinate, energy, and temperature arrays for one state.

   Parameters
   ----------
   lam : float
       Coupling value; its simulation object is added if absent.
   verbose : bool, optional
       Report when matching files cannot be found.
   custom_generic_name : str, optional
       Alternative filename prefix for importing compatible data.

   Returns
   -------
   tuple
       ``(xyz, u, T)`` arrays, or three ``None`` values when absent.


``ECM_basic.load_lambda1_dataset_seperately_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L953>`__

.. code-block:: python

   def load_lambda1_dataset_seperately_(self, m=None)

.. rubric:: Docstring

.. code-block:: text

   Import the physical endpoint from a standard saved MD dataset.

   Parameters
   ----------
   m : int, optional
       Select a representative subset of this size. A cached permutation is
       reused, or a new split is searched for whose subset and complement
       have mean energies close to the full trajectory.


``ECM_basic.import_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1006>`__

.. code-block:: python

   def import_data_(self, which_lambdas: list, verbose=True, custom_generic_name=None)

.. rubric:: Docstring

.. code-block:: text

   Load multiple per-lambda datasets into the workflow.

   Parameters
   ----------
   which_lambdas : list of float or None
       Coupling states to import; ``None`` uses currently active states.
   verbose : bool, optional
       Report how many datasets were located.
   custom_generic_name : str, optional
       Alternative filename prefix passed to :meth:`load_dataset_`.


``ECM_basic.stat_amount_of_data`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1039>`__

.. code-block:: python

   def stat_amount_of_data(self, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Record and optionally print the number of samples at each lambda.


``ECM_basic.global_COM_remover_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1049>`__

.. code-block:: python

   def global_COM_remover_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Remove the mass-weighted global centre of mass from coordinates ``r``.


``ECM_basic.atom_based_COM_remover_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1054>`__

.. code-block:: python

   def atom_based_COM_remover_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Translate coordinates so the configured fixed atom lies at the origin.


``ECM_basic.u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1059>`__

.. code-block:: python

   def u_(self, r, lam)

.. rubric:: Docstring

.. code-block:: text

   Evaluate a lambda potential on coordinates after translation removal.

   Returns ``None`` if the requested lambda state has not been constructed.


``ECM_basic.u_on_r_faster_helper_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1068>`__

.. code-block:: python

   def u_on_r_faster_helper_(self, lam_u, lam_r, _from)

.. rubric:: Docstring

.. code-block:: text

   Evaluate potential ``lam_u`` on unsolved samples from state ``lam_r``.


``ECM_basic.u_on_r_faster_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1074>`__

.. code-block:: python

   def u_on_r_faster_(self, lam_u, lam_r, m=None)

.. rubric:: Docstring

.. code-block:: text

   Return cached cross-potential energies, evaluating only missing frames.

   Parameters
   ----------
   lam_u : float
       Lambda of the evaluated reduced potential.
   lam_r : float
       Lambda whose coordinate dataset supplies configurations.
   m : int, optional
       Return only the first ``m`` evaluations after updating the full cache.

   Returns
   -------
   numpy.ndarray
       Reduced energies shaped ``(frames, 1)``.


``ECM_basic.plot_overlaps_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1113>`__

.. code-block:: python

   def plot_overlaps_(self, lam_i, lam_j, m=None, figsize=(10, 1.5), separate=False)

.. rubric:: Docstring

.. code-block:: text

   Plot cross-evaluated energy histograms for two lambda states.

   Parameters
   ----------
   lam_i, lam_j : float
       Coupling states whose mutual phase-space overlap is shown.
   m : int, optional
       Maximum samples used from each state.
   figsize : tuple, optional
       Matplotlib figure size.
   separate : bool, optional
       Place the two target-potential comparisons in separate figures.


``ECM_basic.estimate_local_FE_difference_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1149>`__

.. code-block:: python

   def estimate_local_FE_difference_(self, lam_i, lam_j, m_i: int=None, m_j: int=None, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Estimate a neighbouring-state free-energy difference with MBAR/BAR.

   Parameters
   ----------
   lam_i, lam_j : float
       Two sampled coupling states; results are stored under the lower value.
   m_i, m_j : int, optional
       Sample limits for the corresponding input states.
   verbose : bool, optional
       Print the dimensionless estimate and standard error.

   Returns
   -------
   Delta_f, dDelta_f : float
       Dimensionless free-energy difference from lower to higher lambda and
       its PyMBAR standard error.


``ECM_basic.estimate_FE_using_mBAR_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1229>`__

.. code-block:: python

   def estimate_FE_using_mBAR_(self, re_evaluate=False)

.. rubric:: Docstring

.. code-block:: text

   Estimate the endpoint free-energy difference using all states jointly.

   Parameters
   ----------
   re_evaluate : bool, optional
       Discard cached cross-potential energies before constructing the full
       MBAR matrix.

   Notes
   -----
   Stores pairwise free-energy and uncertainty matrices, the lambda-zero to
   lambda-one result, neighbouring-pair errors, and a persistent analysis
   log. The printed absolute value includes the analytical EC reference.


``ECM_basic.estimate_FE_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1288>`__

.. code-block:: python

   def estimate_FE_(self, m: int=None, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Estimate absolute crystal free energy by summing adjacent BAR windows.

   Parameters
   ----------
   m : int, optional
       Common sample limit per lambda state.
   verbose : bool, optional
       Print window and cumulative results.

   Notes
   -----
   The analytical lambda-zero free energy is added to all adjacent
   ``Delta_f`` values. Historical behaviour sums window standard errors
   linearly rather than in quadrature.


``ECM_basic.gather_BAR_info_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1342>`__

.. code-block:: python

   def gather_BAR_info_(self, m_lambdas=None)

.. rubric:: Docstring

.. code-block:: text

   Pack the current cumulative BAR result and sampling diagnostics.

   Parameters
   ----------
   m_lambdas : dict, optional
       Per-state prefix lengths used for cumulative-window analysis.

   Returns
   -------
   list
       Absolute FE and SE, numerical and analytical contributions, window
       estimates/errors, a ``(n_lambda, 3)`` array of lambda/mean-energy/
       mean-temperature, and per-state sample counts.


``ECM_basic.rerun_cumulative_BAR_result_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1388>`__

.. code-block:: python

   def rerun_cumulative_BAR_result_(self, n_windows: int=1, re_evaluate=False, reruning_logs=False, save_evaluations=True)

.. rubric:: Docstring

.. code-block:: text

   Recompute BAR convergence as progressively larger data prefixes.

   Parameters
   ----------
   n_windows : int, optional
       Number of equal-count cumulative checkpoints per state.
   re_evaluate : bool, optional
       Clear cached energy evaluations first.
   reruning_logs : bool, optional
       Clear previous BAR history before appending the checkpoints.
   save_evaluations : bool, optional
       Persist the updated cross-potential cache.


``succinic_acid_ARGS_oss`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1467>`__

.. code-block:: python

   def succinic_acid_ARGS_oss(form, cell, key='_')

.. rubric:: Docstring

.. code-block:: text

   Return standard 300 K NVT ECM setup dictionaries for succinic acid.

   ``form`` and ``cell`` select the equilibrated PDB filename; ``key`` controls
   the separator before ``cell``. The system contains 14 atoms per molecule and
   uses GAFF with a 0.36 nm PME cutoff and 2 fs timestep.


``veliparib_ARGS_oss`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1501>`__

.. code-block:: python

   def veliparib_ARGS_oss(form, cell, key='_')

.. rubric:: Docstring

.. code-block:: text

   Return standard 300 K NVT ECM setup dictionaries for veliparib.

   The generated configuration uses 34 atoms per molecule, GAFF, a 0.36 nm PME
   cutoff, and a 0.5 fs integration timestep.


``mivebresib_ARGS_oss`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1534>`__

.. code-block:: python

   def mivebresib_ARGS_oss(form, cell, key='_')

.. rubric:: Docstring

.. code-block:: text

   Return standard 300 K NVT ECM setup dictionaries for mivebresib.

   The generated configuration uses 51 atoms per molecule, GAFF, a 0.36 nm PME
   cutoff, and a 0.5 fs integration timestep.


``remove_lambda_from_lambda_evaluations_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1569>`__

.. code-block:: python

   def remove_lambda_from_lambda_evaluations_(name_old: str, lam, name_new=None)

.. rubric:: Docstring

.. code-block:: text

   1/2 parts of patch breakage
