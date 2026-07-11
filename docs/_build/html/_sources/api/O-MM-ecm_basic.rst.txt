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

.. warning:: Docstring pending.


``LAMBDA_0.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L65>`__

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

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L102>`__

.. code-block:: python

   def energy0_(self, r)

.. rubric:: Docstring

.. code-block:: text

   also was not used, but can compare with openMM (self.lambda_systems[0.0].u_ should evalaute to the same values as this function;
   the input (r) should have the relevant COM already removed if checking this)


``LAMBDA_SYSTEM`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L111>`__

.. code-block:: python

   class LAMBDA_SYSTEM(args_initialise_object, args_initialise_system, args_initialise_simulation, COM_removal_by_fixing_one_atom_index_of_this_atom: int=None, lam=1.0, k_EC=6000.0, stride_save_frame=50, remove_warmup=200)

.. warning:: Docstring pending.


``LAMBDA_SYSTEM.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L112>`__

.. code-block:: python

   def __init__(self, args_initialise_object, args_initialise_system, args_initialise_simulation, COM_removal_by_fixing_one_atom_index_of_this_atom: int=None, lam=1.0, k_EC=6000.0, stride_save_frame=50, remove_warmup=200)

.. warning:: Docstring pending.


``LAMBDA_SYSTEM.plot_check_harmonic`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L226>`__

.. code-block:: python

   def plot_check_harmonic(self)

.. warning:: Docstring pending.


``LAMBDA_SYSTEM.reinitialise_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L235>`__

.. code-block:: python

   def reinitialise_simulation_(self)

.. warning:: Docstring pending.


``LAMBDA_SYSTEM.simulation_timescale`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L240>`__

.. code-block:: python

   def simulation_timescale(self)

.. rubric:: Docstring

.. code-block:: text

   not used here


``LAMBDA_SYSTEM.set_arrays_blank_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L246>`__

.. code-block:: python

   def set_arrays_blank_(self)

.. warning:: Docstring pending.


``LAMBDA_SYSTEM.run_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L249>`__

.. code-block:: python

   def run_simulation_(self, n_saves, verbose_info: str='')

.. warning:: Docstring pending.


``LAMBDA_SYSTEM.xyz`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L277>`__

.. code-block:: python

   def xyz(self)

.. warning:: Docstring pending.


``LAMBDA_SYSTEM.u`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L281>`__

.. code-block:: python

   def u(self)

.. warning:: Docstring pending.


``LAMBDA_SYSTEM.temperature`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L285>`__

.. code-block:: python

   def temperature(self)

.. warning:: Docstring pending.


``ECM_basic`` (class)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L290>`__

.. code-block:: python

   class ECM_basic(name, working_dir_folder_name: str, ARGS_oss: list, k_EC=6000.0, COM_removal_by_fixing_one_atom_index_of_this_atom=None, overwrite=False, path_lambda_1_dataset=None)

.. warning:: Docstring pending.


``ECM_basic.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L291>`__

.. code-block:: python

   def __init__(self, name, working_dir_folder_name: str, ARGS_oss: list, k_EC=6000.0, COM_removal_by_fixing_one_atom_index_of_this_atom=None, overwrite=False, path_lambda_1_dataset=None)

.. warning:: Docstring pending.


``ECM_basic.save_lambda_evaluations_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L354>`__

.. code-block:: python

   def save_lambda_evaluations_(self)

.. warning:: Docstring pending.


``ECM_basic.import_lambda_evaluations_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L358>`__

.. code-block:: python

   def import_lambda_evaluations_(self)

.. warning:: Docstring pending.


``ECM_basic.save_BAR_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L362>`__

.. code-block:: python

   def save_BAR_results_(self)

.. warning:: Docstring pending.


``ECM_basic.import_BAR_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L366>`__

.. code-block:: python

   def import_BAR_results_(self)

.. warning:: Docstring pending.


``ECM_basic.save_mBAR_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L370>`__

.. code-block:: python

   def save_mBAR_results_(self)

.. warning:: Docstring pending.


``ECM_basic.import_mBAR_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L374>`__

.. code-block:: python

   def import_mBAR_results_(self)

.. warning:: Docstring pending.


``ECM_basic.save_usupervised_sample_sizes_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L378>`__

.. code-block:: python

   def save_usupervised_sample_sizes_(self)

.. warning:: Docstring pending.


``ECM_basic.import_usupervised_sample_sizes_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L382>`__

.. code-block:: python

   def import_usupervised_sample_sizes_(self)

.. warning:: Docstring pending.


``ECM_basic.save_inds_rand_lambda1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L386>`__

.. code-block:: python

   def save_inds_rand_lambda1_(self, m)

.. warning:: Docstring pending.


``ECM_basic.import_inds_rand_lambda1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L390>`__

.. code-block:: python

   def import_inds_rand_lambda1_(self, m)

.. warning:: Docstring pending.


``ECM_basic.lambdas`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L397>`__

.. code-block:: python

   def lambdas(self)

.. warning:: Docstring pending.


``ECM_basic.n_lambdas`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L401>`__

.. code-block:: python

   def n_lambdas(self)

.. warning:: Docstring pending.


``ECM_basic.lambda_exists_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L404>`__

.. code-block:: python

   def lambda_exists_(self, lam)

.. warning:: Docstring pending.


``ECM_basic.add_lambda_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L407>`__

.. code-block:: python

   def add_lambda_(self, lam, verbose=True)

.. warning:: Docstring pending.


``ECM_basic.remove_lambda_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L433>`__

.. code-block:: python

   def remove_lambda_(self, lam, verbose=True)

.. warning:: Docstring pending.


``ECM_basic.is_converged_else_remove_all_unconverged_datasets_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L458>`__

.. code-block:: python

   def is_converged_else_remove_all_unconverged_datasets_(self, remove=True)

.. warning:: Docstring pending.


``ECM_basic.run_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L482>`__

.. code-block:: python

   def run_(self, lam: float, n_saves: int)

.. warning:: Docstring pending.


``ECM_basic.last_m_frames_converged_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L491>`__

.. code-block:: python

   def last_m_frames_converged_(self, lam, m, average_temperature_error_allowed=2.0)

.. warning:: Docstring pending.


``ECM_basic.run_to_get_m_coverged_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L506>`__

.. code-block:: python

   def run_to_get_m_coverged_(self, lam, m=5000, average_temperature_error_allowed=2.0)

.. warning:: Docstring pending.


``ECM_basic.lambda_sample_sizes_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L521>`__

.. code-block:: python

   def lambda_sample_sizes_(self, lam=None, m=None)

.. warning:: Docstring pending.


``ECM_basic.which_lambda_to_add_next_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L530>`__

.. code-block:: python

   def which_lambda_to_add_next_(self, max_dataset_size_per_lambda)

.. warning:: Docstring pending.


``ECM_basic.unsupervised_FE_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L571>`__

.. code-block:: python

   def unsupervised_FE_(self, batch_size_increments=10000, max_dataset_size_per_lambda=50000, max_n_lambdas=30, SE_tol_per_molecule=0.03125, re_evaluate=False, rerun_questionable_data=False)

.. warning:: Docstring pending.


``ECM_basic.unsupervised_FE_.main_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L602>`__

.. code-block:: python

   def main_()

.. warning:: Docstring pending.


``ECM_basic.save_simulations_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L652>`__

.. code-block:: python

   def save_simulations_(self, which_lambdas: list=None)

.. warning:: Docstring pending.


``ECM_basic.which_lambdas_exist_in_folder`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L678>`__

.. code-block:: python

   def which_lambdas_exist_in_folder(self)

.. warning:: Docstring pending.


``ECM_basic.load_dataset_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L693>`__

.. code-block:: python

   def load_dataset_(self, lam: float, verbose=False, custom_generic_name=None)

.. warning:: Docstring pending.


``ECM_basic.load_lambda1_dataset_seperately_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L723>`__

.. code-block:: python

   def load_lambda1_dataset_seperately_(self, m=None)

.. warning:: Docstring pending.


``ECM_basic.import_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L766>`__

.. code-block:: python

   def import_data_(self, which_lambdas: list, verbose=True, custom_generic_name=None)

.. warning:: Docstring pending.


``ECM_basic.stat_amount_of_data`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L788>`__

.. code-block:: python

   def stat_amount_of_data(self, verbose=True)

.. warning:: Docstring pending.


``ECM_basic.global_COM_remover_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L797>`__

.. code-block:: python

   def global_COM_remover_(self, r)

.. warning:: Docstring pending.


``ECM_basic.atom_based_COM_remover_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L801>`__

.. code-block:: python

   def atom_based_COM_remover_(self, r)

.. warning:: Docstring pending.


``ECM_basic.u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L805>`__

.. code-block:: python

   def u_(self, r, lam)

.. warning:: Docstring pending.


``ECM_basic.u_on_r_faster_helper_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L810>`__

.. code-block:: python

   def u_on_r_faster_helper_(self, lam_u, lam_r, _from)

.. warning:: Docstring pending.


``ECM_basic.u_on_r_faster_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L815>`__

.. code-block:: python

   def u_on_r_faster_(self, lam_u, lam_r, m=None)

.. warning:: Docstring pending.


``ECM_basic.plot_overlaps_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L838>`__

.. code-block:: python

   def plot_overlaps_(self, lam_i, lam_j, m=None, figsize=(10, 1.5), separate=False)

.. warning:: Docstring pending.


``ECM_basic.estimate_local_FE_difference_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L861>`__

.. code-block:: python

   def estimate_local_FE_difference_(self, lam_i, lam_j, m_i: int=None, m_j: int=None, verbose=True)

.. warning:: Docstring pending.


``ECM_basic.estimate_FE_using_mBAR_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L924>`__

.. code-block:: python

   def estimate_FE_using_mBAR_(self, re_evaluate=False)

.. warning:: Docstring pending.


``ECM_basic.estimate_FE_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L969>`__

.. code-block:: python

   def estimate_FE_(self, m: int=None, verbose=True)

.. warning:: Docstring pending.


``ECM_basic.gather_BAR_info_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1008>`__

.. code-block:: python

   def gather_BAR_info_(self, m_lambdas=None)

.. warning:: Docstring pending.


``ECM_basic.rerun_cumulative_BAR_result_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1040>`__

.. code-block:: python

   def rerun_cumulative_BAR_result_(self, n_windows: int=1, re_evaluate=False, reruning_logs=False, save_evaluations=True)

.. warning:: Docstring pending.


``succinic_acid_ARGS_oss`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1106>`__

.. code-block:: python

   def succinic_acid_ARGS_oss(form, cell, key='_')

.. warning:: Docstring pending.


``veliparib_ARGS_oss`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1134>`__

.. code-block:: python

   def veliparib_ARGS_oss(form, cell, key='_')

.. warning:: Docstring pending.


``mivebresib_ARGS_oss`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1162>`__

.. code-block:: python

   def mivebresib_ARGS_oss(form, cell, key='_')

.. warning:: Docstring pending.


``remove_lambda_from_lambda_evaluations_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ecm_basic.py#L1192>`__

.. code-block:: python

   def remove_lambda_from_lambda_evaluations_(name_old: str, lam, name_new=None)

.. rubric:: Docstring

.. code-block:: text

   1/2 parts of patch breakage
