.. _api-O-NN-pgm_rb:

O.NN.pgm_rb
===========

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py>`__

.. rubric:: Docstring

.. code-block:: text

   Variable-cell (NPT) coordinate maps, flow model, and training interfaces.

   An NPT microstate is carried as ``rb`` with shape ``(batch, N + 3, 3)``:
   particle coordinates followed by three lower-triangular box vectors. The model
   extends positional flow variables with six cell degrees of freedom. This path
   is experimental and should be validated carefully for each system.


Classes and functions
---------------------

``rb_to_r_b_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L66>`__

.. code-block:: python

   def rb_to_r_b_(rb)

.. rubric:: Docstring

.. code-block:: text

   Split combined ``rb`` states into coordinates and box vectors.


``r_b_to_rb_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L73>`__

.. code-block:: python

   def r_b_to_rb_tf_(r, b)

.. rubric:: Docstring

.. code-block:: text

   Concatenate TensorFlow coordinates and boxes along the atom axis.


``r_b_to_rb_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L77>`__

.. code-block:: python

   def r_b_to_rb_np_(r, b)

.. rubric:: Docstring

.. code-block:: text

   Concatenate NumPy coordinates and boxes along the atom axis.


``box_forward_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L83>`__

.. code-block:: python

   def box_forward_tf_(b)

.. rubric:: Docstring

.. code-block:: text

   placeholder


``box_inverse_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L89>`__

.. code-block:: python

   def box_inverse_tf_(h)

.. rubric:: Docstring

.. code-block:: text

   placeholder


``box_forward_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L98>`__

.. code-block:: python

   def box_forward_np_(b)

.. rubric:: Docstring

.. code-block:: text

   placeholder


``SingleComponent_map_rb`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L106>`__

.. code-block:: python

   class SingleComponent_map_rb(PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   !! : molecule must have >3 atoms to use this M_{IC} layer


``SingleComponent_map_rb.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L108>`__

.. code-block:: python

   def __init__(self, PDB_single_mol: str)

.. rubric:: Docstring

.. code-block:: text

   Create an uninitialised variable-cell map for one molecule type.


``SingleComponent_map_rb.remove_COM_from_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L117>`__

.. code-block:: python

   def remove_COM_from_data_(self, rb)

.. rubric:: Docstring

.. code-block:: text

   important step : fixing the first rO atoms (i.e., rO atom in the first molecule to zero)
   other rO atoms shifted into the [0,1) box, taking the molecules with them.
   Only the rO atoms are under PBCs, because molecules are whole and not transformed here in any way.


``SingleComponent_map_rb.initalise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L158>`__

.. code-block:: python

   def initalise_(self, rb_dataset, batch_size=10000, n_mol_unitcell=1, COM_remover='blank', focused='blank', whiten_setting=0)

.. rubric:: Docstring

.. code-block:: text

   Fit NPT position, box, internal, and rotational transformations.

   One molecular anchor removes translation. Remaining fractional
   positions are periodic; ``whiten_setting`` selects no whitening,
   positions-only whitening, or joint position/box whitening.


``SingleComponent_map_rb.sample_base_P_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L275>`__

.. code-block:: python

   def sample_base_P_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Sample reduced positions and six box variables uniformly on [-1, 1].


``SingleComponent_map_rb.white_setting_0_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L280>`__

.. code-block:: python

   def white_setting_0_(self, inputs, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Scale box variables without whitening positions.


``SingleComponent_map_rb.white_setting_1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L300>`__

.. code-block:: python

   def white_setting_1_(self, inputs, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Whiten positions separately while only scaling box variables.


``SingleComponent_map_rb.white_setting_2_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L329>`__

.. code-block:: python

   def white_setting_2_(self, inputs, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Jointly whiten and scale concatenated position and box variables.


``SingleComponent_map_rb.forward_rb_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L359>`__

.. code-block:: python

   def forward_rb_(self, rb)

.. rubric:: Docstring

.. code-block:: text

   Map combined coordinates/boxes to NPT flow variables and Jacobian.


``SingleComponent_map_rb.inverse_rb_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L421>`__

.. code-block:: python

   def inverse_rb_(self, variables_in)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct combined coordinates/boxes from NPT flow variables.


``SingleComponent_map_rb.xO_reshape_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L477>`__

.. code-block:: python

   def xO_reshape_(self, x, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Insert or remove the dummy anchor while preserving six box DOFs.


``SingleComponent_map_rb.flow_mask_xO`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L483>`__

.. code-block:: python

   def flow_mask_xO(self)

.. rubric:: Docstring

.. code-block:: text

   Mask the dummy anchor and allow all physical position/box variables.


``SingleComponent_map_rb.flow_mask_X`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L489>`__

.. code-block:: python

   def flow_mask_X(self)

.. rubric:: Docstring

.. code-block:: text

   All-ones mask for molecular internal and rotational variables.


``PGMcrys_v1_rb`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L498>`__

.. code-block:: python

   class PGMcrys_v1_rb(ic_maps: list, n_layers: int=4, optimiser_LR_decay=[0.001, 0.0], DIM_connection=10, n_att_heads=4, initialise=True)

.. rubric:: Docstring

.. code-block:: text

   !! : molecule should have >3 atoms (also true in ic_map)


``PGMcrys_v1_rb.load_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L501>`__

.. code-block:: python

   def load_model(path_and_name: str, VERSION='blank')

.. rubric:: Docstring

.. code-block:: text

   Load a variable-cell model from the current artifact format.


``PGMcrys_v1_rb.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L505>`__

.. code-block:: python

   def __init__(self, ic_maps: list, n_layers: int=4, optimiser_LR_decay=[0.001, 0.0], DIM_connection=10, n_att_heads=4, initialise=True)

.. rubric:: Docstring

.. code-block:: text

   Build an NPT PGM with alternating position-box/conformer flows.

   ``ic_maps`` provides one compatible variable-cell representation per
   state. Six box degrees of freedom extend the positional layer. Remaining
   architecture arguments mirror ``PGMcrys_v1``.


``PGMcrys_v1_rb.get_C2P_P2C_extensions_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L645>`__

.. code-block:: python

   def get_C2P_P2C_extensions_(self, m, crystal_index)

.. rubric:: Docstring

.. code-block:: text

   Broadcast the selected state encoding for both coupling directions.


``PGMcrys_v1_rb._forward_coupling_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L661>`__

.. code-block:: python

   def _forward_coupling_(self, X, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Apply alternating position-box and conformer couplings forward.


``PGMcrys_v1_rb._inverse_coupling_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L682>`__

.. code-block:: python

   def _inverse_coupling_(self, Z, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Invert NPT couplings in exact reverse order.


``PGMcrys_v1_rb._forward_represenation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L705>`__

.. code-block:: python

   def _forward_represenation_(self, rb, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Apply the selected state's combined coordinate-box representation.


``PGMcrys_v1_rb._inverse_represenation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L712>`__

.. code-block:: python

   def _inverse_represenation_(self, X, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Invert representation variables to the combined ``rb`` state.


``NN_interface_sc_rb`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L723>`__

.. code-block:: python

   class NN_interface_sc_rb(name: str, path_dataset: str, fraction_training: float=0.8, training: bool=True)

.. rubric:: Docstring

.. code-block:: text

   Prepare one NPT dataset and coordinate map for model training.


``NN_interface_sc_rb.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L726>`__

.. code-block:: python

   def __init__(self, name: str, path_dataset: str, fraction_training: float=0.8, training: bool=True)

.. rubric:: Docstring

.. code-block:: text

   Load metadata and optionally import one variable-cell MD dataset.


``NN_interface_sc_rb.reduced_enthalpy_from_reduced_energies_and_boxes_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L756>`__

.. code-block:: python

   def reduced_enthalpy_from_reduced_energies_and_boxes_(self, reduced_energies, boxes)

.. rubric:: Docstring

.. code-block:: text

   Convert reduced potential energies to NPT reduced enthalpies.

   Adds isotropic ``PV/(kT)`` and the variable-cell measure/Jacobian term
   used by this implementation. Inputs contain one energy and box per
   frame; output has shape ``(n_frames, 1)``.


``NN_interface_sc_rb.u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L781>`__

.. code-block:: python

   def u_(self, rb)

.. rubric:: Docstring

.. code-block:: text

   Evaluate the reduced NPT enthalpy of combined coordinate-box states.


``NN_interface_sc_rb.import_MD_dataset_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L799>`__

.. code-block:: python

   def import_MD_dataset_(self)

.. rubric:: Docstring

.. code-block:: text

   Load saved NPT simulation data and construct aligned ``rb``/enthalpy arrays.


``NN_interface_sc_rb.truncate_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L824>`__

.. code-block:: python

   def truncate_data_(self, m=None)

.. rubric:: Docstring

.. code-block:: text

   Keep ``m`` balanced samples, discard temporary split arrays, and reset indices.


``NN_interface_sc_rb.set_ic_map_step1`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L841>`__

.. code-block:: python

   def set_ic_map_step1(self, ind_root_atom=11, option=None)

.. rubric:: Docstring

.. code-block:: text

   self.sc.mol is available for this reason; to check the indices of atoms in the molecule
   once (ind_root_atom, option) pair is chosen, keep a fixed note of this for this molecule


``NN_interface_sc_rb.set_ic_map_step2`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L854>`__

.. code-block:: python

   def set_ic_map_step2(self, inds_rand=None, check_PES=True)

.. rubric:: Docstring

.. code-block:: text

   Remove translation, create a train/validation split, and check energies.


``NN_interface_sc_rb.set_ic_map_step3`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L864>`__

.. code-block:: python

   def set_ic_map_step3(self, n_mol_unitcell: int=1, whiten_setting=2)

.. rubric:: Docstring

.. code-block:: text

   Fit the NPT representation and print a small inversion diagnostic.


``adjust_ranges_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L898>`__

.. code-block:: python

   def adjust_ranges_(nn)

.. rubric:: Docstring

.. code-block:: text

   Expand bond/angle scaler ranges to common bounds across all states.


``NN_interface_sc_multimap_rb`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L913>`__

.. code-block:: python

   class NN_interface_sc_multimap_rb(name: str, paths_datasets: list, fraction_training: float=0.8, running_in_notebook: bool=False, training: bool=True)

.. rubric:: Docstring

.. code-block:: text

   Coordinate multi-state NPT preprocessing, training, and FE analysis.


``NN_interface_sc_multimap_rb.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L916>`__

.. code-block:: python

   def __init__(self, name: str, paths_datasets: list, fraction_training: float=0.8, running_in_notebook: bool=False, training: bool=True)

.. rubric:: Docstring

.. code-block:: text

   Create one single-state NPT interface per supplied dataset.


``NN_interface_sc_multimap_rb.save_inds_rand_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L943>`__

.. code-block:: python

   def save_inds_rand_(self)

.. rubric:: Docstring

.. code-block:: text

   Save each state's train/validation permutation under a distinct key.


``NN_interface_sc_multimap_rb.load_inds_rand_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L949>`__

.. code-block:: python

   def load_inds_rand_(self)

.. rubric:: Docstring

.. code-block:: text

   Restore every state's saved train/validation permutation.


``NN_interface_sc_multimap_rb.set_ic_map_step1`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L955>`__

.. code-block:: python

   def set_ic_map_step1(self, ind_root_atom=11, option=None)

.. rubric:: Docstring

.. code-block:: text

   Configure common internal-coordinate anchor choices for all states.


``NN_interface_sc_multimap_rb.set_ic_map_step2`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L961>`__

.. code-block:: python

   def set_ic_map_step2(self, check_PES=True)

.. rubric:: Docstring

.. code-block:: text

   Remove translation and split every state's dataset.


``NN_interface_sc_multimap_rb.set_ic_map_step3`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L965>`__

.. code-block:: python

   def set_ic_map_step3(self, n_mol_unitcells: list=None, whiten_setting=2)

.. rubric:: Docstring

.. code-block:: text

   Fit one NPT map per state using state-specific unit-cell sizes.


``NN_interface_sc_multimap_rb.set_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L992>`__

.. code-block:: python

   def set_model(self, learning_rate=0.001, evaluation_batch_size=5000, n_layers=4, DIM_connection=10, n_att_heads=4, initialise=True, test_inverse=True)

.. rubric:: Docstring

.. code-block:: text

   Construct the shared NPT model and optionally test each state map.


``NN_interface_sc_multimap_rb.set_trainer`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1020>`__

.. code-block:: python

   def set_trainer(self, n_batches_between_evaluations=50)

.. rubric:: Docstring

.. code-block:: text

   Create the shared training orchestrator and evaluation schedule.


``NN_interface_sc_multimap_rb.u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1028>`__

.. code-block:: python

   def u_(self, r, k)

.. rubric:: Docstring

.. code-block:: text

   Evaluate combined-state reduced enthalpy with state ``k``'s system.


``NN_interface_sc_multimap_rb.train`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1032>`__

.. code-block:: python

   def train(self, n_batches=2000, save_BAR=True, save_mBAR=False, save_misc=True, verbose=True, verbose_divided_by_n_mol=True, f_halfwindow_visualisation=1.0, test_inverse=False, evaluate_on_training_data=False, evaluate_main=True, training_batch_size=1000)

.. rubric:: Docstring

.. code-block:: text

   Train all NPT states and optionally save BAR/MBAR diagnostics.


``NN_interface_sc_multimap_rb.load_misc_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1116>`__

.. code-block:: python

   def load_misc_(self)

.. rubric:: Docstring

.. code-block:: text

   Restore training histories and propagate state-specific result views.


``NN_interface_sc_multimap_rb.load_energies_during_training_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1131>`__

.. code-block:: python

   def load_energies_during_training_(self)

.. rubric:: Docstring

.. code-block:: text

   Load saved generated/MD energy histories for every state.


``NN_interface_sc_multimap_rb.plot_energies_during_training_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1135>`__

.. code-block:: python

   def plot_energies_during_training_(self, crystal_index=0, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Plot energy-overlap diagnostics for one state.


``NN_interface_sc_multimap_rb.solve_BAR_using_pymbar_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1139>`__

.. code-block:: python

   def solve_BAR_using_pymbar_(self, rerun=False)

.. rubric:: Docstring

.. code-block:: text

   Solve saved two-state BAR inputs independently for every state.


``NN_interface_sc_multimap_rb.plot_result_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1144>`__

.. code-block:: python

   def plot_result_(self, crystal_index=0, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Plot final free-energy estimates for one state.


``NN_interface_sc_multimap_rb.solve_mBAR_using_pymbar_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1149>`__

.. code-block:: python

   def solve_mBAR_using_pymbar_(self, rerun=False, n_bootstraps=0, uncertainty_method=None, save_output=True)

.. rubric:: Docstring

.. code-block:: text

   Solve saved cross-state MBAR inputs over all evaluation batches.


``NN_interface_sc_multimap_rb.sample_model_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1212>`__

.. code-block:: python

   def sample_model_(self, m, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Generate full evaluation-sized batches up to requested count ``m``.


``NN_interface_sc_multimap_rb.save_samples_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1217>`__

.. code-block:: python

   def save_samples_(self, m: int=20000)

.. rubric:: Docstring

.. code-block:: text

   Generate and pickle samples for every state.


``NN_interface_sc_multimap_rb.load_samples_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm_rb.py#L1222>`__

.. code-block:: python

   def load_samples_(self, crystal_index=None)

.. rubric:: Docstring

.. code-block:: text

   Load samples for one state or all states into ``samples_from_model``.
