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

.. warning:: Docstring pending.


``update_HarmonicBondForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L69>`__

.. code-block:: python

   def update_HarmonicBondForce_(_force, _lam, deepcopy=True)

.. warning:: Docstring pending.


``update_HarmonicAngleForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L80>`__

.. code-block:: python

   def update_HarmonicAngleForce_(_force, _lam, deepcopy=True)

.. warning:: Docstring pending.


``update_PeriodicTorsionForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L91>`__

.. code-block:: python

   def update_PeriodicTorsionForce_(_force, _lam, deepcopy=True)

.. warning:: Docstring pending.


``update_RBTorsionForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L103>`__

.. code-block:: python

   def update_RBTorsionForce_(_force, _lam, deepcopy=False)

.. rubric:: Docstring

.. code-block:: text

   only in OPLS, not in GAFF or TIP4P


``update_NonbondedForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L123>`__

.. code-block:: python

   def update_NonbondedForce_(_force, _lam, deepcopy=True)

.. warning:: Docstring pending.


``update_CustomNonbondedForce_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L152>`__

.. code-block:: python

   def update_CustomNonbondedForce_(_force, _lam, deepcopy=True)

.. warning:: Docstring pending.


``put_lambda_into_system_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L175>`__

.. code-block:: python

   def put_lambda_into_system_(system, lam, R0, k_EC=5000.0, verbose=True, inds_true_atoms=None)

.. rubric:: Docstring

.. code-block:: text

   REF : http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html

   force.getEnergyFunction()


``MM_system_helper`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L251>`__

.. code-block:: python

   class MM_system_helper()

.. warning:: Docstring pending.


``MM_system_helper.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L252>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``MM_system_helper.inject_methods_from_another_class_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L256>`__

.. code-block:: python

   def inject_methods_from_another_class_(self, class_to_inject_methods_from, **kwargs)

.. warning:: Docstring pending.


``MM_system_helper.corrections_to_ff_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L259>`__

.. code-block:: python

   def corrections_to_ff_(self, verbose)

.. warning:: Docstring pending.


``MM_system_helper._system_mass_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L264>`__

.. code-block:: python

   def _system_mass_(self)

.. warning:: Docstring pending.


``MM_system_helper.define_mu_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L267>`__

.. code-block:: python

   def define_mu_(self, index_atom=None)

.. warning:: Docstring pending.


``MM_system_helper._set_b_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L286>`__

.. code-block:: python

   def _set_b_(self, b)

.. warning:: Docstring pending.


``MM_system_helper._set_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L290>`__

.. code-block:: python

   def _set_r_(self, r)

.. warning:: Docstring pending.


``MM_system_helper._set_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L294>`__

.. code-block:: python

   def _set_v_(self, v)

.. warning:: Docstring pending.


``MM_system_helper.forward_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L298>`__

.. code-block:: python

   def forward_atom_index_(self, inds)

.. warning:: Docstring pending.


``MM_system_helper.inverse_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L301>`__

.. code-block:: python

   def inverse_atom_index_(self, inds)

.. warning:: Docstring pending.


``MM_system_helper._current_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L305>`__

.. code-block:: python

   def _current_r_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_COM_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L311>`__

.. code-block:: python

   def _current_COM_(self)

.. warning:: Docstring pending.


``MM_system_helper._recenter_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L315>`__

.. code-block:: python

   def _recenter_simulation_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L319>`__

.. code-block:: python

   def _current_v_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_p_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L325>`__

.. code-block:: python

   def _current_p_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_K_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L331>`__

.. code-block:: python

   def _current_K_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_T_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L337>`__

.. code-block:: python

   def _current_T_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_U_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L344>`__

.. code-block:: python

   def _current_U_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L350>`__

.. code-block:: python

   def _current_u_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_F_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L355>`__

.. code-block:: python

   def _current_F_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_b_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L361>`__

.. code-block:: python

   def _current_b_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_V_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L367>`__

.. code-block:: python

   def _current_V_(self)

.. warning:: Docstring pending.


``MM_system_helper._current_rho_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L373>`__

.. code-block:: python

   def _current_rho_(self)

.. warning:: Docstring pending.


``MM_system_helper._add_barostat_to_system_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L390>`__

.. code-block:: python

   def _add_barostat_to_system_(self)

.. warning:: Docstring pending.


``MM_system_helper._remove_barostat_from_system_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L413>`__

.. code-block:: python

   def _remove_barostat_from_system_(self)

.. warning:: Docstring pending.


``MM_system_helper.initialise_integrator_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L432>`__

.. code-block:: python

   def initialise_integrator_(self, integrator_class, collision_rate=1)

.. warning:: Docstring pending.


``MM_system_helper._list_forces_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L458>`__

.. code-block:: python

   def _list_forces_(self)

.. warning:: Docstring pending.


``MM_system_helper.turn_ON_nonbonded_SwitchingFunction`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L464>`__

.. code-block:: python

   def turn_ON_nonbonded_SwitchingFunction(self, factor=0.95)

.. warning:: Docstring pending.


``MM_system_helper.adjust_EwaldErrorTolerance`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L471>`__

.. code-block:: python

   def adjust_EwaldErrorTolerance(self, tol, verbose=True)

.. warning:: Docstring pending.


``MM_system_helper._reset_temperature_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L480>`__

.. code-block:: python

   def _reset_temperature_(self, T: float)

.. warning:: Docstring pending.


``MM_system_helper._print_potential_enrrgy_contributions_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L487>`__

.. code-block:: python

   def _print_potential_enrrgy_contributions_(self)

.. warning:: Docstring pending.


``MM_system_helper._U_GPU_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L518>`__

.. code-block:: python

   def _U_GPU_(self, r, b=None)

.. warning:: Docstring pending.


``MM_system_helper.u_GPU_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L540>`__

.. code-block:: python

   def u_GPU_(self, r, b=None)

.. warning:: Docstring pending.


``MM_system_helper.F_GPU_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L546>`__

.. code-block:: python

   def F_GPU_(self, r, b=None)

.. warning:: Docstring pending.


``MM_system_helper.minimise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L568>`__

.. code-block:: python

   def minimise_(self, verbose=True)

.. warning:: Docstring pending.


``MM_system_helper.minimise_xyz_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L575>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L583>`__

.. code-block:: python

   def check_shape_(x)

.. warning:: Docstring pending.


``MM_system_helper._Hessian_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L619>`__

.. code-block:: python

   def _Hessian_(self, r, b=None, dr=0.0001, fixed_atom_index=None, temperature_reduced=True)

.. warning:: Docstring pending.


``MM_system_helper.harmonic_FE_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L648>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L730>`__

.. code-block:: python

   def set_arrays_blank_(self)

.. warning:: Docstring pending.


``MM_system_helper.save_frame_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L741>`__

.. code-block:: python

   def save_frame_(self)

.. warning:: Docstring pending.


``MM_system_helper.run_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L752>`__

.. code-block:: python

   def run_simulation_(self, n_saves, stride_save_frame: int=100, verbose_info: str='')

.. warning:: Docstring pending.


``MM_system_helper.run_simulation_w_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L762>`__

.. code-block:: python

   def run_simulation_w_(self, n_saves, stride_save_frame: int=100, verbose_info: str='')

.. rubric:: Docstring

.. code-block:: text

   w : wrapped ; for NVT in the presence of shearing (at higher T) or alchemical


``MM_system_helper.xyz`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L776>`__

.. code-block:: python

   def xyz(self)

.. warning:: Docstring pending.


``MM_system_helper.velicities`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L781>`__

.. code-block:: python

   def velicities(self)

.. warning:: Docstring pending.


``MM_system_helper.COMs`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L786>`__

.. code-block:: python

   def COMs(self)

.. warning:: Docstring pending.


``MM_system_helper.boxes`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L791>`__

.. code-block:: python

   def boxes(self)

.. warning:: Docstring pending.


``MM_system_helper.u`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L796>`__

.. code-block:: python

   def u(self)

.. warning:: Docstring pending.


``MM_system_helper.temperature`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L801>`__

.. code-block:: python

   def temperature(self, verbose=True)

.. warning:: Docstring pending.


``MM_system_helper.dt`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L806>`__

.. code-block:: python

   def dt(self)

.. warning:: Docstring pending.


``MM_system_helper.timescale`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L810>`__

.. code-block:: python

   def timescale(self)

.. warning:: Docstring pending.


``MM_system_helper.average_temperature`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L822>`__

.. code-block:: python

   def average_temperature(self)

.. warning:: Docstring pending.


``MM_system_helper.average_energy`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L825>`__

.. code-block:: python

   def average_energy(self)

.. warning:: Docstring pending.


``MM_system_helper.average_volume`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L828>`__

.. code-block:: python

   def average_volume(self)

.. warning:: Docstring pending.


``MM_system_helper.plot_simulation_info_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L833>`__

.. code-block:: python

   def plot_simulation_info_(self, figsize=(10, 10))

.. rubric:: Docstring

.. code-block:: text

   one plot with all information about the simulation


``MM_system_helper.plot_temperature_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L837>`__

.. code-block:: python

   def plot_temperature_(self, window: float=None)

.. warning:: Docstring pending.


``MM_system_helper.temperature_plot`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L849>`__

.. code-block:: python

   def temperature_plot(self)

.. warning:: Docstring pending.


``MM_system_helper.plot_energy_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L853>`__

.. code-block:: python

   def plot_energy_(self, window: float=None)

.. warning:: Docstring pending.


``MM_system_helper.energy_plot`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L864>`__

.. code-block:: python

   def energy_plot(self)

.. warning:: Docstring pending.


``MM_system_helper.plot_volume_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L868>`__

.. code-block:: python

   def plot_volume_(self, window: float=None)

.. warning:: Docstring pending.


``MM_system_helper.volume_plot`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L879>`__

.. code-block:: python

   def volume_plot(self)

.. warning:: Docstring pending.


``MM_system_helper.box_lengths_plot`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L885>`__

.. code-block:: python

   def box_lengths_plot(self)

.. warning:: Docstring pending.


``MM_system_helper.index_frame_average_box_othorhombic_case_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L890>`__

.. code-block:: python

   def index_frame_average_box_othorhombic_case_(self)

.. warning:: Docstring pending.


``MM_system_helper.box_shapes`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L898>`__

.. code-block:: python

   def box_shapes(self)

.. warning:: Docstring pending.


``MM_system_helper.partial_charges_mol`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L905>`__

.. code-block:: python

   def partial_charges_mol(self)

.. warning:: Docstring pending.


``MM_system_helper.box_line`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L925>`__

.. code-block:: python

   def box_line(self, key='CRYST1')

.. rubric:: Docstring

.. code-block:: text

   only useful for the rough save_coordiantes_as_pdb_, not used otherwise.


``MM_system_helper.load_structures_with_mdtraj_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L941>`__

.. code-block:: python

   def load_structures_with_mdtraj_(self, r, b=None)

.. warning:: Docstring pending.


``MM_system_helper.save_gro_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L968>`__

.. code-block:: python

   def save_gro_(self, r, name: str, b=None)

.. warning:: Docstring pending.


``MM_system_helper.save_pdb_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L975>`__

.. code-block:: python

   def save_pdb_(self, r, name: str, b=None)

.. warning:: Docstring pending.


``MM_system_helper.save_xtc_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L982>`__

.. code-block:: python

   def save_xtc_(self, r, name: str, b=None, save_reference=True)

.. warning:: Docstring pending.


``plot_simulation_info_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L999>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1109>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1119>`__

.. code-block:: python

   def get_angle_(v1, v2, radians=True)

.. warning:: Docstring pending.


``save_gro_as_pdb_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1140>`__

.. code-block:: python

   def save_gro_as_pdb_(GRO: str, PDB: str=None)

.. warning:: Docstring pending.


``PDB_to_xyz_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1155>`__

.. code-block:: python

   def PDB_to_xyz_(PDB: str)

.. warning:: Docstring pending.


``PDB_to_box_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1158>`__

.. code-block:: python

   def PDB_to_box_(PDB: str)

.. warning:: Docstring pending.


``box_to_lengths_angles_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1161>`__

.. code-block:: python

   def box_to_lengths_angles_(b)

.. rubric:: Docstring

.. code-block:: text

   b : (3,3) or (m,3,3) ; box or boxes


``lengths_angles_to_box_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1166>`__

.. code-block:: python

   def lengths_angles_to_box_(x)

.. rubric:: Docstring

.. code-block:: text

   x : (6) or (m,6) ; lengths and angles of one or more boxes


``get_index_average_box_automatic_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1171>`__

.. code-block:: python

   def get_index_average_box_automatic_(boxes, n_bins=30, rules=['av'] * 3 + ['max_prob'] * 3, verbose=False)

.. warning:: Docstring pending.


``get_index_average_box_automatic_.peak_finder_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1194>`__

.. code-block:: python

   def peak_finder_(x, i)

.. warning:: Docstring pending.


``get_index_average_box_automatic_.plot_box_lengths_angles_histograms_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1210>`__

.. code-block:: python

   def plot_box_lengths_angles_histograms_(boxes, b0=None, b1=None)

.. warning:: Docstring pending.


``get_unitcell_stack_order_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1253>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1319>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1332>`__

.. code-block:: python

   def expand_mdtraj_(input_instance, n_copies)

.. warning:: Docstring pending.


``box_in_reduced_form_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1374>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1398>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1417>`__

.. code-block:: python

   def reorder_atoms_mol_(mol_pdb_fname, template_pdb_fname, output_pdb_fname)

.. rubric:: Docstring

.. code-block:: text

   REF: https://gist.github.com/fabian-paul/abba9172d394dffb93624a710acbab16


``validate_reorder_atoms_mol_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1466>`__

.. code-block:: python

   def validate_reorder_atoms_mol_(template_pdb_fname, output_pdb_fname)

.. rubric:: Docstring

.. code-block:: text

   REF: https://gist.github.com/fabian-paul/abba9172d394dffb93624a710acbab16


``reorder_atoms_unitcell_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1490>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1501>`__

.. code-block:: python

   def split_(PDB, n_atoms_mol, ref=False)

.. warning:: Docstring pending.


``reorder_atoms_unitcell_.expand_mdtraj_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1524>`__

.. code-block:: python

   def expand_mdtraj_(input_instance, n_copies)

.. warning:: Docstring pending.


``vectors_between_atoms_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1551>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1595>`__

.. code-block:: python

   def change_box_(PDB, n_atoms_mol, make_orthorhombic=False, save_output=True, traj=None)

.. rubric:: Docstring

.. code-block:: text

   dont remember


``change_box_.wrap_points_1box_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1599>`__

.. code-block:: python

   def wrap_points_1box_(Ri, box)

.. warning:: Docstring pending.


``remove_clashes_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1638>`__

.. code-block:: python

   def remove_clashes_(PDB_unitcell: str, tol=0.001)

.. rubric:: Docstring

.. code-block:: text

   no needed in most cases?


``remove_clashes_.wrap_points_1box_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1651>`__

.. code-block:: python

   def wrap_points_1box_(Ri, box)

.. warning:: Docstring pending.


``remove_clashes_.minimum_image_othorhombic_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1657>`__

.. code-block:: python

   def minimum_image_othorhombic_(r, b)

.. warning:: Docstring pending.


``rename_atoms_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1726>`__

.. code-block:: python

   def rename_atoms_(PDB, n_atoms_mol)

.. warning:: Docstring pending.


``process_mercury_output_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1761>`__

.. code-block:: python

   def process_mercury_output_(PDB, n_atoms_mol: int, single_mol=False, custom_path_name=None, tol=0.001)

.. rubric:: Docstring

.. code-block:: text

   preparing initial structure


``extract_subcell_from_supercell_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1791>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1801>`__

.. code-block:: python

   def check_shape_(x)

.. warning:: Docstring pending.


``extract_subcell_from_supercell_.dot_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_helper.py#L1808>`__

.. code-block:: python

   def dot_(Ri, mat)

.. warning:: Docstring pending.
