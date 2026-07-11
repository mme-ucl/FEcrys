.. _api-O-MM-mm_helper:

O.MM.mm_helper
==============

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py>`__

.. rubric:: Docstring

.. code-block:: text

   mm_helper.py
   ECM:
       f : update_HarmonicBondForce_
       f : update_HarmonicAngleForce_
       f : update_PeriodicTorsionForce_
       f : update_RBTorsionForce_ ; not tested yet, because not in GAFF
       f : update_NonbondedForce_
       f : update_CustomNonbondedForce_ ; carefull with param1 vs 2, in GAFF was ok
       f : put_lambda_into_system_ ; ! add method for update_RBTorsionForce_

   MM:
       c : MM_system_helper
       f : plot_simulation_info_
       f : cell_lengths_and_angles_ ; mdtraj also has this method

   make supercells:
       f : get_unitcell_stack_order_
       f : supercell_from_unitcell_

   check box for opnemm:
       f : box_in_reduced_form_
       f : reducePeriodicBoxVectors_

   fix ordering of atoms in a molecule:
       f : reorder_atoms_mol_
       f : validate_reorder_atoms_mol_
       f : reorder_atoms_unitcell_

   misc:
       f : vectors_between_atoms_
       f : change_box_
       f : remove_clashes_
       f : rename_atoms_ ; not used, blank
       f : process_mercury_output_ ; not used?
       
   not used:
       f : extract_subcell_from_supercell_
       f : CustomIntegrator_


Classes and functions
---------------------

``get_force_by_name_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L64>`__

.. code-block:: python

   def get_force_by_name_(system, name: str)

.. rubric:: Docstring

.. code-block:: text

   Return the OpenMM force whose concrete class name equals ``name``.


``update_HarmonicBondForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L70>`__

.. code-block:: python

   def update_HarmonicBondForce_(_force, _lam, deepcopy=True)

.. rubric:: Docstring

.. code-block:: text

   Scale every harmonic bond force constant by ``_lam``.


``update_HarmonicAngleForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L82>`__

.. code-block:: python

   def update_HarmonicAngleForce_(_force, _lam, deepcopy=True)

.. rubric:: Docstring

.. code-block:: text

   Scale every harmonic angle force constant by ``_lam``.


``update_PeriodicTorsionForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L94>`__

.. code-block:: python

   def update_PeriodicTorsionForce_(_force, _lam, deepcopy=True)

.. rubric:: Docstring

.. code-block:: text

   Scale every periodic-torsion energy coefficient by ``_lam``.


``update_RBTorsionForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L107>`__

.. code-block:: python

   def update_RBTorsionForce_(_force, _lam, deepcopy=False)

.. rubric:: Docstring

.. code-block:: text

   only in OPLS, not in GAFF or TIP4P


``update_NonbondedForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L127>`__

.. code-block:: python

   def update_NonbondedForce_(_force, _lam, deepcopy=True)

.. rubric:: Docstring

.. code-block:: text

   Scale Lennard-Jones and electrostatic energies by ``_lam``.

   Particle epsilon values scale linearly and charges by ``sqrt(_lam)``;
   exception epsilon and charge-product values scale linearly. The input is
   copied by default and mutated only when ``deepcopy=False``.


``update_CustomNonbondedForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L162>`__

.. code-block:: python

   def update_CustomNonbondedForce_(_force, _lam, deepcopy=True)

.. rubric:: Docstring

.. code-block:: text

   Scale a supported custom nonbonded force by ``_lam``.

   Forces exposing the global parameter ``ecm_lambda`` are scaled through that
   parameter; older epsilon/sigma forces are scaled per particle.


``put_lambda_into_system_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L190>`__

.. code-block:: python

   def put_lambda_into_system_(system, lam, R0, k_EC=5000.0, verbose=True, inds_true_atoms=None)

.. rubric:: Docstring

.. code-block:: text

   REF : http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html

   force.getEnergyFunction()


``MM_system_helper`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L266>`__

.. code-block:: python

   class MM_system_helper()

.. rubric:: Docstring

.. code-block:: text

   Shared OpenMM context, evaluation, trajectory, and export utilities.


``MM_system_helper.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L269>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Initialise drift and ensemble-state flags.


``MM_system_helper.inject_methods_from_another_class_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L274>`__

.. code-block:: python

   def inject_methods_from_another_class_(self, class_to_inject_methods_from, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Bind methods and optional properties from another class to this instance.


``MM_system_helper.corrections_to_ff_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L278>`__

.. code-block:: python

   def corrections_to_ff_(self, verbose)

.. rubric:: Docstring

.. code-block:: text

   Default no-op hook for force-field-specific system corrections.


``MM_system_helper._system_mass_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L284>`__

.. code-block:: python

   def _system_mass_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: OpenMM particle masses in daltons.


``MM_system_helper.define_mu_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L288>`__

.. code-block:: python

   def define_mu_(self, index_atom=None)

.. rubric:: Docstring

.. code-block:: text

   Define masses and normalised weights used for COM removal.

   ``index_atom=None`` uses physical mass fractions. An integer produces a
   one-hot fixed-atom reference; a non-integer index selection is repeated
   across molecules and mass-weighted.


``MM_system_helper._set_b_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L313>`__

.. code-block:: python

   def _set_b_(self, b)

.. rubric:: Docstring

.. code-block:: text

   Set a ``(3, 3)`` periodic box matrix in nanometres.


``MM_system_helper._set_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L318>`__

.. code-block:: python

   def _set_r_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Set ``(N, 3)`` Cartesian positions in nanometres.


``MM_system_helper._set_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L323>`__

.. code-block:: python

   def _set_v_(self, v)

.. rubric:: Docstring

.. code-block:: text

   Set ``(N, 3)`` velocities in nanometres per picosecond.


``MM_system_helper.forward_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L328>`__

.. code-block:: python

   def forward_atom_index_(self, inds)

.. rubric:: Docstring

.. code-block:: text

   Map public indices to OpenMM indices; identity in the base helper.


``MM_system_helper.inverse_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L332>`__

.. code-block:: python

   def inverse_atom_index_(self, inds)

.. rubric:: Docstring

.. code-block:: text

   Map OpenMM indices to public indices; identity in the base helper.


``MM_system_helper._current_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L337>`__

.. code-block:: python

   def _current_r_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current positions in nanometres.


``MM_system_helper._current_COM_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L344>`__

.. code-block:: python

   def _current_COM_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current weighted centre of mass shaped ``(1, 3)``.


``MM_system_helper._recenter_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L349>`__

.. code-block:: python

   def _recenter_simulation_(self)

.. rubric:: Docstring

.. code-block:: text

   Translate the current structure so its configured COM is zero.


``MM_system_helper._current_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L354>`__

.. code-block:: python

   def _current_v_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current velocities in nanometres per picosecond.


``MM_system_helper._current_p_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L361>`__

.. code-block:: python

   def _current_p_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current momenta in dalton nanometres per picosecond.


``MM_system_helper._current_K_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L368>`__

.. code-block:: python

   def _current_K_(self)

.. rubric:: Docstring

.. code-block:: text

   float: Current kinetic energy in kJ/mol.


``MM_system_helper._current_T_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L375>`__

.. code-block:: python

   def _current_T_(self)

.. rubric:: Docstring

.. code-block:: text

   float: Instantaneous kinetic temperature in kelvin.


``MM_system_helper._current_U_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L383>`__

.. code-block:: python

   def _current_U_(self)

.. rubric:: Docstring

.. code-block:: text

   float: Current potential energy in kJ/mol.


``MM_system_helper._current_u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L390>`__

.. code-block:: python

   def _current_u_(self)

.. rubric:: Docstring

.. code-block:: text

   float: Current reduced potential energy ``beta * U``.


``MM_system_helper._current_F_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L396>`__

.. code-block:: python

   def _current_F_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current forces in kJ mol⁻¹ nm⁻¹.


``MM_system_helper._current_b_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L403>`__

.. code-block:: python

   def _current_b_(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: Current row-vector box matrix in nanometres.


``MM_system_helper._current_V_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L410>`__

.. code-block:: python

   def _current_V_(self)

.. rubric:: Docstring

.. code-block:: text

   float: Current box volume in nm³.


``MM_system_helper._current_rho_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L417>`__

.. code-block:: python

   def _current_rho_(self)

.. rubric:: Docstring

.. code-block:: text

   float: Current mass density in g/cm³.


``MM_system_helper._add_barostat_to_system_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L435>`__

.. code-block:: python

   def _add_barostat_to_system_(self)

.. rubric:: Docstring

.. code-block:: text

   Add the configured isotropic, anisotropic, or flexible barostat.


``MM_system_helper._remove_barostat_from_system_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L459>`__

.. code-block:: python

   def _remove_barostat_from_system_(self)

.. rubric:: Docstring

.. code-block:: text

   Remove every recognised Monte Carlo barostat from the system.


``MM_system_helper.initialise_integrator_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L479>`__

.. code-block:: python

   def initialise_integrator_(self, integrator_class, collision_rate=1)

.. rubric:: Docstring

.. code-block:: text

   Construct a supported integrator at the configured T and timestep.


``MM_system_helper._list_forces_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L506>`__

.. code-block:: python

   def _list_forces_(self)

.. rubric:: Docstring

.. code-block:: text

   Print all OpenMM force names in system order.


``MM_system_helper.turn_ON_nonbonded_SwitchingFunction`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L513>`__

.. code-block:: python

   def turn_ON_nonbonded_SwitchingFunction(self, factor=0.95)

.. rubric:: Docstring

.. code-block:: text

   Enable switching for all standard/custom nonbonded forces.


``MM_system_helper.adjust_EwaldErrorTolerance`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L521>`__

.. code-block:: python

   def adjust_EwaldErrorTolerance(self, tol, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Set Ewald error tolerance on every standard nonbonded force.


``MM_system_helper._reset_temperature_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L531>`__

.. code-block:: python

   def _reset_temperature_(self, T: float)

.. rubric:: Docstring

.. code-block:: text

   Update stored and integrator temperatures in kelvin.


``MM_system_helper._print_potential_enrrgy_contributions_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L539>`__

.. code-block:: python

   def _print_potential_enrrgy_contributions_(self)

.. rubric:: Docstring

.. code-block:: text

   Placeholder for reporting force-group energy contributions.


``MM_system_helper._U_GPU_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L571>`__

.. code-block:: python

   def _U_GPU_(self, r, b=None)

.. rubric:: Docstring

.. code-block:: text

   Evaluate dimensional potential energies for a coordinate batch.

   Returns ``(frames, 1)`` energies in kJ/mol and restores the original
   context positions and box before returning.


``MM_system_helper.u_GPU_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L598>`__

.. code-block:: python

   def u_GPU_(self, r, b=None)

.. rubric:: Docstring

.. code-block:: text

   Evaluate batched reduced potential energies ``beta * U``.


``MM_system_helper.F_GPU_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L605>`__

.. code-block:: python

   def F_GPU_(self, r, b=None)

.. rubric:: Docstring

.. code-block:: text

   Evaluate batched forces in kJ mol⁻¹ nm⁻¹, restoring context state.


``MM_system_helper.minimise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L628>`__

.. code-block:: python

   def minimise_(self, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Minimise the current OpenMM context positions at fixed box.


``MM_system_helper.minimise_xyz_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L636>`__

.. code-block:: python

   def minimise_xyz_(self, r, b=None, verbose=False)

.. rubric:: Docstring

.. code-block:: text

   energy minimising only the coordinates (r), box (b) is fixed
   Inputs:
       r : single frame (N,3) or trajectory (m,N,3)
       b : single frame (3,3) or trajectory (m,3,3)
   Output:
       r : single frame (N,3) or trajectory (m,N,3) after minimising in fixed box(es)


``MM_system_helper.minimise_xyz_.check_shape_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L644>`__

.. code-block:: python

   def check_shape_(x)

.. rubric:: Docstring

.. code-block:: text

   Normalise one frame or a trajectory to a three-dimensional array.


``MM_system_helper._Hessian_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L681>`__

.. code-block:: python

   def _Hessian_(self, r, b=None, dr=0.0001, fixed_atom_index=None, temperature_reduced=True)

.. rubric:: Docstring

.. code-block:: text

   Estimate the Cartesian Hessian by central differences of forces.

   ``dr`` is in nanometres. A fixed atom's three coordinates are omitted;
   by default the returned matrix is multiplied by ``beta`` and therefore
   has reduced units of nm⁻².


``MM_system_helper.harmonic_FE_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L716>`__

.. code-block:: python

   def harmonic_FE_(self, r, b, fixed_atom_index: int, dr=0.0001, n_steps_openmm=2, n_steps_adam=5000, alpha_adam=0.0001, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Classical Harmonic Approximation : configurational part only. Momentum not included here.
   Inputs:
       r : (N,3) array of positions
       b : (3,3) array of box 
       fixed_atom_index : index of atom to keep fixed; the only constraint in a crystal.
   Parameters:
       dr : finite difference parameter; can try a few small positive values for a stable output
       n_minimisations : number of times to run potential energy gradient descent (box is fixed).

   Output: dictionary = {
       'f0' = configurational Helmholtz per system FE in kT at the local minimum
       'u0' = potential energy per system in kT at the local minimum
       'r0' = minimised structure at the local minimum
       }

   Limitation: 
       Classical and requires finite temperature
       Box is not minimised, fixed as the original input (b) provided


``MM_system_helper.set_arrays_blank_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L798>`__

.. code-block:: python

   def set_arrays_blank_(self)

.. rubric:: Docstring

.. code-block:: text

   Reset trajectory buffers, frame count, and measured integration time.


``MM_system_helper.save_frame_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L810>`__

.. code-block:: python

   def save_frame_(self)

.. rubric:: Docstring

.. code-block:: text

   Append current coordinates, reduced energy, T, box, and COM.


``MM_system_helper.run_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L822>`__

.. code-block:: python

   def run_simulation_(self, n_saves, stride_save_frame: int=100, verbose_info: str='')

.. rubric:: Docstring

.. code-block:: text

   Advance OpenMM and save ``n_saves`` frames at the requested stride.


``MM_system_helper.run_simulation_w_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L833>`__

.. code-block:: python

   def run_simulation_w_(self, n_saves, stride_save_frame: int=100, verbose_info: str='')

.. rubric:: Docstring

.. code-block:: text

   w : wrapped ; for NVT in the presence of shearing (at higher T) or alchemical


``MM_system_helper.xyz`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L847>`__

.. code-block:: python

   def xyz(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray or None: Saved coordinates in nanometres.


``MM_system_helper.velicities`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L853>`__

.. code-block:: python

   def velicities(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray or None: Saved velocities; historical spelling retained.


``MM_system_helper.COMs`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L859>`__

.. code-block:: python

   def COMs(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray or None: Saved weighted centres of mass in nanometres.


``MM_system_helper.boxes`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L865>`__

.. code-block:: python

   def boxes(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray or None: Saved box matrices in nanometres.


``MM_system_helper.u`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L871>`__

.. code-block:: python

   def u(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray or None: Saved reduced energies shaped ``(frames, 1)``.


``MM_system_helper.temperature`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L877>`__

.. code-block:: python

   def temperature(self, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray or None: Saved instantaneous temperatures in kelvin.


``MM_system_helper.dt`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L883>`__

.. code-block:: python

   def dt(self)

.. rubric:: Docstring

.. code-block:: text

   float: Integration timestep in picoseconds.


``MM_system_helper.timescale`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L888>`__

.. code-block:: python

   def timescale(self)

.. rubric:: Docstring

.. code-block:: text

   float or None: Sampled trajectory duration in nanoseconds.


``MM_system_helper.average_temperature`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L901>`__

.. code-block:: python

   def average_temperature(self)

.. rubric:: Docstring

.. code-block:: text

   float: Mean saved temperature in kelvin.


``MM_system_helper.average_energy`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L905>`__

.. code-block:: python

   def average_energy(self)

.. rubric:: Docstring

.. code-block:: text

   float: Mean saved reduced potential energy.


``MM_system_helper.average_volume`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L909>`__

.. code-block:: python

   def average_volume(self)

.. rubric:: Docstring

.. code-block:: text

   float: Mean saved box volume in nm³.


``MM_system_helper.plot_simulation_info_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L915>`__

.. code-block:: python

   def plot_simulation_info_(self, figsize=(10, 10))

.. rubric:: Docstring

.. code-block:: text

   one plot with all information about the simulation


``MM_system_helper.plot_temperature_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L919>`__

.. code-block:: python

   def plot_temperature_(self, window: float=None)

.. rubric:: Docstring

.. code-block:: text

   Plot instantaneous and cumulative-average temperatures.


``MM_system_helper.temperature_plot`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L932>`__

.. code-block:: python

   def temperature_plot(self)

.. rubric:: Docstring

.. code-block:: text

   Print mean temperature and create the default temperature plot.


``MM_system_helper.plot_energy_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L937>`__

.. code-block:: python

   def plot_energy_(self, window: float=None)

.. rubric:: Docstring

.. code-block:: text

   Plot instantaneous and cumulative-average reduced energies.


``MM_system_helper.energy_plot`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L949>`__

.. code-block:: python

   def energy_plot(self)

.. rubric:: Docstring

.. code-block:: text

   Print mean energy and create the default energy plot.


``MM_system_helper.plot_volume_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L954>`__

.. code-block:: python

   def plot_volume_(self, window: float=None)

.. rubric:: Docstring

.. code-block:: text

   Plot instantaneous and cumulative-average volumes in nm³.


``MM_system_helper.volume_plot`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L966>`__

.. code-block:: python

   def volume_plot(self)

.. rubric:: Docstring

.. code-block:: text

   Print initial/mean volume and create the default volume plot.


``MM_system_helper.box_lengths_plot`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L973>`__

.. code-block:: python

   def box_lengths_plot(self)

.. rubric:: Docstring

.. code-block:: text

   Plot diagonal box lengths and the minimum cutoff-compatible length.


``MM_system_helper.index_frame_average_box_othorhombic_case_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L979>`__

.. code-block:: python

   def index_frame_average_box_othorhombic_case_(self)

.. rubric:: Docstring

.. code-block:: text

   Return frames closest to mean orthorhombic shape and volume.


``MM_system_helper.box_shapes`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L988>`__

.. code-block:: python

   def box_shapes(self)

.. rubric:: Docstring

.. code-block:: text

   Return initial and, when available, sampled cell lengths/angles.


``MM_system_helper.partial_charges_mol`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L996>`__

.. code-block:: python

   def partial_charges_mol(self)

.. rubric:: Docstring

.. code-block:: text

   numpy.ndarray: First molecule's partial charges in elementary charge.


``MM_system_helper.box_line`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1017>`__

.. code-block:: python

   def box_line(self, key='CRYST1')

.. rubric:: Docstring

.. code-block:: text

   only useful for the rough save_coordiantes_as_pdb_, not used otherwise.


``MM_system_helper.load_structures_with_mdtraj_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1033>`__

.. code-block:: python

   def load_structures_with_mdtraj_(self, r, b=None)

.. rubric:: Docstring

.. code-block:: text

   Create an MDTraj trajectory from coordinates and optional boxes.


``MM_system_helper.save_gro_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1061>`__

.. code-block:: python

   def save_gro_(self, r, name: str, b=None)

.. rubric:: Docstring

.. code-block:: text

   Save coordinates and per-frame boxes in GROMACS GRO format.


``MM_system_helper.save_pdb_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1069>`__

.. code-block:: python

   def save_pdb_(self, r, name: str, b=None)

.. rubric:: Docstring

.. code-block:: text

   Save coordinates as a PDB trajectory using MDTraj.


``MM_system_helper.save_xtc_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1077>`__

.. code-block:: python

   def save_xtc_(self, r, name: str, b=None, save_reference=True)

.. rubric:: Docstring

.. code-block:: text

   Save an XTC trajectory and optionally a first-frame reference PDB.


``plot_simulation_info_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1095>`__

.. code-block:: python

   def plot_simulation_info_(self: object, figsize=(10, 10))

.. rubric:: Docstring

.. code-block:: text

   after running an NVT or NPT simulation using SingleComponent, this plots some of the information about the simulation as a function of time
   the plots include:
       temperature : red
       potential energy : blue
       volume : green
       box vector length and angles
       diagonal lengths of the box matrix

   useful for checking consistency (e.g., in NPT, does the supercell always relax to the same state and is this converged or needs to run longer)


``cell_lengths_and_angles_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1205>`__

.. code-block:: python

   def cell_lengths_and_angles_(b, radians=False)

.. rubric:: Docstring

.. code-block:: text

   matching mdtraj, could be done there instead
   Inputs:
       b : (m,3,3) boxes (box-vectors are the rows; axis=-2)
       radians: default False
   Outputs:
       lengths_and_angles : m * [[a,b,c], [bc,ac,ab]]


``cell_lengths_and_angles_.get_angle_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1215>`__

.. code-block:: python

   def get_angle_(v1, v2, radians=True)

.. rubric:: Docstring

.. code-block:: text

   Return clipped vector angles in radians or degrees.


``save_gro_as_pdb_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1237>`__

.. code-block:: python

   def save_gro_as_pdb_(GRO: str, PDB: str=None)

.. rubric:: Docstring

.. code-block:: text

   Convert a GRO file to PDB using MDTraj, with MDAnalysis fallback.


``PDB_to_xyz_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1253>`__

.. code-block:: python

   def PDB_to_xyz_(PDB: str)

.. rubric:: Docstring

.. code-block:: text

   Return the first PDB frame as ``(N, 3)`` coordinates in nanometres.


``PDB_to_box_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1257>`__

.. code-block:: python

   def PDB_to_box_(PDB: str)

.. rubric:: Docstring

.. code-block:: text

   Return the first PDB frame's ``(3, 3)`` box in nanometres.


``box_to_lengths_angles_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1261>`__

.. code-block:: python

   def box_to_lengths_angles_(b)

.. rubric:: Docstring

.. code-block:: text

   b : (3,3) or (m,3,3) ; box or boxes


``lengths_angles_to_box_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1266>`__

.. code-block:: python

   def lengths_angles_to_box_(x)

.. rubric:: Docstring

.. code-block:: text

   x : (6) or (m,6) ; lengths and angles of one or more boxes


``get_index_average_box_automatic_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1271>`__

.. code-block:: python

   def get_index_average_box_automatic_(boxes, n_bins=30, rules=['av'] * 3 + ['max_prob'] * 3, verbose=False)

.. rubric:: Docstring

.. code-block:: text

   Select the sampled box closest to marginal representative values.

   Each length/angle marginal is summarised by its mean, mode, or minimum as
   specified by ``rules``. Standardised six-dimensional distance identifies the
   nearest actual frame; ``verbose`` also plots the marginal histograms.


``get_index_average_box_automatic_.peak_finder_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1300>`__

.. code-block:: python

   def peak_finder_(x, i)

.. rubric:: Docstring

.. code-block:: text

   Summarise marginal ``i`` using its configured histogram rule.


``get_index_average_box_automatic_.plot_box_lengths_angles_histograms_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1317>`__

.. code-block:: python

   def plot_box_lengths_angles_histograms_(boxes, b0=None, b1=None)

.. rubric:: Docstring

.. code-block:: text

   Plot sampled cell marginals with input and selected boxes.


``get_unitcell_stack_order_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1361>`__

.. code-block:: python

   def get_unitcell_stack_order_(b, n_mol_unitcell=1, top_n=None)

.. rubric:: Docstring

.. code-block:: text

   not sure if this is good

   tells how many times to stack the unitcell in each of the three directions
       while minimising surface area to volume ratio (more spherical ~~ more cubic)
       
   Inputs:
       b : (3,3) : box of the unit cell (NB: only the diagonal distances will be used)
       n_mol_unitcell : number of molecules in the unit cell
       top_n : how many results to include in output
   Output:
       res: dictionary {n_mol_supercell : number of times to stack unitcell in each of the three unitcell vector directions}
           res[n_mol_supercell of interest] --> input for the supercell_from_unitcell_ function


``supercell_from_unitcell_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1427>`__

.. code-block:: python

   def supercell_from_unitcell_(PDB_unitcell: str, cell: list=[1, 1, 1], save_output=True)

.. rubric:: Docstring

.. code-block:: text

   copy ideal unitcell along unitcell vectors to get ideal supercell of correct shape
   Inputs:
       PDB_unitcell : unitcell coordinates, must contain the header with unitcell lengths and angles
       cell : list of three integers (> 0) for how many copies to make along each unitcell vector direction
           [default : [1,1,1] : identity : no change]
       save_output : if True, saves the supercell
   Outputs:
       instance of mdtraj of the larger supercell for further modifications


``supercell_from_unitcell_.expand_mdtraj_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1440>`__

.. code-block:: python

   def expand_mdtraj_(input_instance, n_copies)

.. rubric:: Docstring

.. code-block:: text

   Stack ``n_copies`` identical MDTraj topologies.


``box_in_reduced_form_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1483>`__

.. code-block:: python

   def box_in_reduced_form_(box)

.. rubric:: Docstring

.. code-block:: text

   method copied from openmm sources
   Input:
       box : (3,3) box-vectors are rows
   Output:
       bool : True if the box is already in the reduced form


``reducePeriodicBoxVectors_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1507>`__

.. code-block:: python

   def reducePeriodicBoxVectors_(box)

.. rubric:: Docstring

.. code-block:: text

   method copied from openmm sources
   Input:
       box : (3,3) box-vectors are rows
   Output:
       the single box converted to format compatible with OpenMM


``reorder_atoms_mol_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1526>`__

.. code-block:: python

   def reorder_atoms_mol_(mol_pdb_fname, template_pdb_fname, output_pdb_fname)

.. rubric:: Docstring

.. code-block:: text

   REF: https://gist.github.com/fabian-paul/abba9172d394dffb93624a710acbab16


``validate_reorder_atoms_mol_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1575>`__

.. code-block:: python

   def validate_reorder_atoms_mol_(template_pdb_fname, output_pdb_fname)

.. rubric:: Docstring

.. code-block:: text

   REF: https://gist.github.com/fabian-paul/abba9172d394dffb93624a710acbab16


``reorder_atoms_unitcell_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1599>`__

.. code-block:: python

   def reorder_atoms_unitcell_(PDB: str, PDB_ref: str, n_atoms_mol: int)

.. rubric:: Docstring

.. code-block:: text

   Inputs:
       PDB     : supercell or unitcell of a molecule (e.g., to_reorder.pdb)
       PDB_ref : single molecule with intended order of atoms
       n_atoms_mol : number of atoms in the molecule, including all atoms
   Outputs:
       to_reorder_reordered.pdb : file with same xyz coordiantes as PDB, but atom order and names changed to match PDB_ref


``reorder_atoms_unitcell_.split_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1610>`__

.. code-block:: python

   def split_(PDB, n_atoms_mol, ref=False)

.. rubric:: Docstring

.. code-block:: text

   Split a structure into temporary one-molecule PDB files.


``reorder_atoms_unitcell_.expand_mdtraj_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1634>`__

.. code-block:: python

   def expand_mdtraj_(input_instance, n_copies)

.. rubric:: Docstring

.. code-block:: text

   Stack a molecular MDTraj object to rebuild the full cell.


``vectors_between_atoms_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1662>`__

.. code-block:: python

   def vectors_between_atoms_(r, b, n_images_search=1)

.. rubric:: Docstring

.. code-block:: text

   ALL vectors (not just those within the diagonal subspace of the box eye(3)*b)

   naive search for the shortest distanace vector between two atoms 

   Inputs:
       r : (N,3) single configuration (can be selection of any N atoms of interest)
       b : (3,3) box (box-vectors are the rows)
       n_images_search : number of periodic images to search for the shortest vector
           [if b very skewed can increase n_images_search; more expensive]
   Outputs:
       vs_out : (N,N,3) shortest vectors between all atoms (from rows to columns)


``change_box_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1706>`__

.. code-block:: python

   def change_box_(PDB, n_atoms_mol, make_orthorhombic=False, save_output=True, traj=None)

.. rubric:: Docstring

.. code-block:: text

   dont remember


``change_box_.wrap_points_1box_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1710>`__

.. code-block:: python

   def wrap_points_1box_(Ri, box)

.. rubric:: Docstring

.. code-block:: text

   Wrap points into one triclinic box through fractional coordinates.


``remove_clashes_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1750>`__

.. code-block:: python

   def remove_clashes_(PDB_unitcell: str, tol=0.001)

.. rubric:: Docstring

.. code-block:: text

   no needed in most cases?


``remove_clashes_.wrap_points_1box_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1763>`__

.. code-block:: python

   def wrap_points_1box_(Ri, box)

.. rubric:: Docstring

.. code-block:: text

   Wrap points into one box through fractional coordinates.


``remove_clashes_.minimum_image_othorhombic_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1770>`__

.. code-block:: python

   def minimum_image_othorhombic_(r, b)

.. rubric:: Docstring

.. code-block:: text

   Return orthorhombic minimum-image distances, masks, and vectors.


``rename_atoms_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1840>`__

.. code-block:: python

   def rename_atoms_(PDB, n_atoms_mol)

.. rubric:: Docstring

.. code-block:: text

   Compatibility placeholder for atom renaming; currently performs no edit.


``process_mercury_output_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1876>`__

.. code-block:: python

   def process_mercury_output_(PDB, n_atoms_mol: int, single_mol=False, custom_path_name=None, tol=0.001)

.. rubric:: Docstring

.. code-block:: text

   preparing initial structure


``extract_subcell_from_supercell_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1906>`__

.. code-block:: python

   def extract_subcell_from_supercell_(n_mol_in: int, n_atoms_mol: int, r_in, b_in, cell_in: list, cell_out: list, ind_rO: int)

.. rubric:: Docstring

.. code-block:: text

   rough method, was not used!

   cuts out a smaller cell from a bigger cell
   inputs (r_in) need to first go (successfully) through tidy_crystal_xyz_
   outputs (r_out, b_out); ouput needs to be energy-minimised before being used


``extract_subcell_from_supercell_.check_shape_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1916>`__

.. code-block:: python

   def check_shape_(x)

.. rubric:: Docstring

.. code-block:: text

   Normalise a matrix or matrix batch to three axes.


``extract_subcell_from_supercell_.dot_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1924>`__

.. code-block:: python

   def dot_(Ri, mat)

.. rubric:: Docstring

.. code-block:: text

   Apply one matrix per frame to batched atom vectors.
