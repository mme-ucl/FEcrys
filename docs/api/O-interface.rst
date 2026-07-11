.. _api-O-interface:

O.interface
===========

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py>`__

.. rubric:: Docstring

.. code-block:: text

   High-level workflows for training and evaluating FEcrys flow models.

   The classes in this module connect molecular-dynamics datasets, internal-
   coordinate maps, probabilistic generative models, and BAR/MBAR free-energy
   estimators.  They also centralise persistence of training results and generated
   samples.  Methods whose names end in an underscore follow the naming convention
   used throughout the original FEcrys research code; the suffix does not imply a
   private method.


Classes and functions
---------------------

``NN_interface_helper`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L18>`__

.. code-block:: python

   class NN_interface_helper()

.. rubric:: Docstring

.. code-block:: text

   Provide persistence and free-energy analysis shared by the interfaces.

   Subclasses are expected to set ``name``, construct ``model`` and ``trainer``,
   and populate the energy and evaluation arrays used by the analysis methods.


``NN_interface_helper.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L25>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Initialise standard result directories and the split-index cache.


``NN_interface_helper.name_save_misc`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L34>`__

.. code-block:: python

   def name_save_misc(self)

.. rubric:: Docstring

.. code-block:: text

   str: Prefix used for miscellaneous training-result files.


``NN_interface_helper.save_misc_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L37>`__

.. code-block:: python

   def save_misc_(self)

.. rubric:: Docstring

.. code-block:: text

   Save the trainer's accumulated diagnostics and free-energy curves.


``NN_interface_helper.name_save_inds_rand`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L43>`__

.. code-block:: python

   def name_save_inds_rand(self)

.. rubric:: Docstring

.. code-block:: text

   str: Prefix used to persist the shuffled dataset indices.


``NN_interface_helper.save_inds_rand_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L46>`__

.. code-block:: python

   def save_inds_rand_(self, key='')

.. rubric:: Docstring

.. code-block:: text

   Save the indices defining the training/validation split.

   Parameters
   ----------
   key : str, optional
       Suffix that distinguishes states sharing the same interface name.


``NN_interface_helper.load_inds_rand_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L55>`__

.. code-block:: python

   def load_inds_rand_(self, key='')

.. rubric:: Docstring

.. code-block:: text

   Restore previously saved training/validation split indices.

   Parameters
   ----------
   key : str, optional
       State-specific suffix used when the indices were saved.


``NN_interface_helper.name_save_inv_test`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L66>`__

.. code-block:: python

   def name_save_inv_test(self)

.. rubric:: Docstring

.. code-block:: text

   str: Prefix used for model-inversion diagnostic results.


``NN_interface_helper.save_inv_test_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L69>`__

.. code-block:: python

   def save_inv_test_results_(self)

.. rubric:: Docstring

.. code-block:: text

   Save inversion diagnostics collected by the trainer.


``NN_interface_helper.load_inv_test_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L72>`__

.. code-block:: python

   def load_inv_test_results_(self)

.. rubric:: Docstring

.. code-block:: text

   Load inversion diagnostics into ``inv_test_results``.


``NN_interface_helper.name_save_BAR_inputs`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L77>`__

.. code-block:: python

   def name_save_BAR_inputs(self)

.. rubric:: Docstring

.. code-block:: text

   str: Prefix for two-state BAR input and output files.


``NN_interface_helper.name_save_mBAR_inputs`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L84>`__

.. code-block:: python

   def name_save_mBAR_inputs(self)

.. rubric:: Docstring

.. code-block:: text

   str: Prefix for multistate BAR input and output files.


``NN_interface_helper.name_save_samples`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L91>`__

.. code-block:: python

   def name_save_samples(self)

.. rubric:: Docstring

.. code-block:: text

   str: Prefix for configurations sampled from the trained model.


``NN_interface_helper.name_save_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L97>`__

.. code-block:: python

   def name_save_model(self)

.. rubric:: Docstring

.. code-block:: text

   str: Prefix for the serialised generative model.


``NN_interface_helper.save_model_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L100>`__

.. code-block:: python

   def save_model_(self)

.. rubric:: Docstring

.. code-block:: text

   Serialise the current model using its native save method.


``NN_interface_helper.load_model_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L103>`__

.. code-block:: python

   def load_model_(self, VERSION='NEW')

.. rubric:: Docstring

.. code-block:: text

   Load a saved model, optionally enabling legacy import aliases.

   Parameters
   ----------
   VERSION : {'NEW', 'OLD'}, optional
       ``'NEW'`` uses the current package layout.  Any other value installs
       aliases for the former top-level ``NN`` package before loading an
       old pickle.


``NN_interface_helper.set_training_validation_split_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L130>`__

.. code-block:: python

   def set_training_validation_split_(self, n_training, inds_rand=None)

.. rubric:: Docstring

.. code-block:: text

   Shuffle a dataset and construct its training and validation subsets.

   The same permutation is applied to coordinates ``r`` and reduced
   potential energies ``u``.  If no permutation is supplied or cached,
   :func:`find_split_indices_` searches for a split whose energy
   distributions agree within the hard-coded tolerance.

   Parameters
   ----------
   n_training : int
       Number of shuffled observations assigned to the training subset.
   inds_rand : array_like of int, optional
       Complete permutation of dataset indices.  It is cached in
       ``self.inds_rand`` for reproducible reuse.


``NN_interface_helper.check_PES_matching_dataset_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L167>`__

.. code-block:: python

   def check_PES_matching_dataset_(self, m=1000)

.. rubric:: Docstring

.. code-block:: text

   Compare stored energies with the configured potential-energy function.

   Parameters
   ----------
   m : int, optional
       Maximum number of configurations checked in each data subset.

   Notes
   -----
   The method prints the mean, minimum, and maximum residual for both the
   training and validation data; it does not impose an acceptance threshold.


``NN_interface_helper.load_energies_during_training_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L188>`__

.. code-block:: python

   def load_energies_during_training_(self, index_of_state=0)

.. rubric:: Docstring

.. code-block:: text

   Load saved MD and model energy samples from every evaluation batch.

   Parameters
   ----------
   index_of_state : int, optional
       Thermodynamic-state index embedded in the BAR filenames.

   Notes
   -----
   The loaded values populate ``MD_energies_T``, ``MD_energies_V``, and
   ``BG_energies`` with axes ``(state, evaluation, sample)``.


``NN_interface_helper.plot_energies_during_training_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L235>`__

.. code-block:: python

   def plot_energies_during_training_(self, dpi=300, n_bins=80, _from=0, _range=None)

.. rubric:: Docstring

.. code-block:: text

   Plot energy-overlap histograms over the course of training.

   Parameters
   ----------
   dpi : int, optional
       Resolution of the histogram figure.
   n_bins : int, optional
       Number of common energy bins.
   _from : int, optional
       First model-evaluation index to display.
   _range : tuple of float, optional
       Explicit lower and upper histogram limits.  By default they are
       inferred from validation energies and extended toward higher energy.

   Returns
   -------
   None
       The colour scale and histogram figures are created with Matplotlib.


``NN_interface_helper.solve_BAR_using_pymbar_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L304>`__

.. code-block:: python

   def solve_BAR_using_pymbar_(self, rerun=False, index_of_state=0, key='', n_bootstraps=0, uncertainty_method=None, save_output=True, method_for_selective_evalaution_=None)

.. rubric:: Docstring

.. code-block:: text

   Estimate absolute free energies from saved two-state BAR inputs.

   Parameters
   ----------
   rerun : bool, optional
       Recompute even when a saved output is available.
   index_of_state : int, optional
       State whose training and validation BAR inputs should be loaded.
   key : str, optional
       Suffix distinguishing the saved result for this state.
   n_bootstraps : int, optional
       Number of bootstrap samples passed to :class:`pymbar.MBAR`.
   uncertainty_method : str, optional
       PyMBAR uncertainty method, for example ``'bootstrap'``.
   save_output : bool, optional
       Persist the resulting ``estimates_BAR`` array.
   method_for_selective_evalaution_ : callable, optional
       Alternative evaluator used when model-sample energies should be
       computed only for selected training batches.

   Notes
   -----
   ``estimates_BAR`` has leading rows for training free energy/error and
   validation free energy/error.  Failed estimates are marked with ``1e20``.


``NN_interface_helper.set_final_result_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L418>`__

.. code-block:: python

   def set_final_result_(self)

.. rubric:: Docstring

.. code-block:: text

   Aggregate raw validation BAR estimates into the reported result.

   Running free-energy averages and their dispersion are computed with
   :func:`FE_of_model_curve_`.  The final values are exposed as
   ``BAR_V_FE``, ``BAR_V_SD``, and ``BAR_V_SE``.


``NN_interface_helper.plot_result_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L476>`__

.. code-block:: python

   def plot_result_(self, window=1, entropy_only=False, plot_red=True, n_mol=1, colors=['green', 'blue', 'm', 'red'], ax=None, plot_raw_errors=True, alpha_raw_errors=0.4)

.. rubric:: Docstring

.. code-block:: text

   Plot raw and running free-energy estimates against training batch.

   Parameters
   ----------
   window : float, optional
       Half-width of the vertical plotting window around the final estimate.
   entropy_only : bool, optional
       Retained for API compatibility; the current implementation does not
       transform the plotted values.
   plot_red : bool, optional
       Show running grid-search and BAR estimates with uncertainty bounds.
   n_mol : int, optional
       Number of molecules used to convert crystal to per-molecule values.
   colors : sequence of str, optional
       Colours for raw BAR, auxiliary, grid-search, and averaged BAR curves.
   ax : matplotlib.axes.Axes, optional
       Existing axes.  When omitted, the pyplot state is used.
   plot_raw_errors : bool, optional
       Shade the raw PyMBAR standard-error interval.
   alpha_raw_errors : float, optional
       Opacity of the raw-error band.

   Returns
   -------
   None
       The plot is modified in place and final numerical estimates printed.


``NN_interface_sc`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L556>`__

.. code-block:: python

   class NN_interface_sc(name: str, path_dataset: str, fraction_training: float=0.8, training: bool=True, ic_map_class=SingleComponent_map)

.. rubric:: Docstring

.. code-block:: text

   Represent one metastable state described by one MD dataset.

   The dataset is split into training and validation observations, while one
   internal-coordinate map is fitted to the full coordinate ensemble.


``NN_interface_sc.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L563>`__

.. code-block:: python

   def __init__(self, name: str, path_dataset: str, fraction_training: float=0.8, training: bool=True, ic_map_class=SingleComponent_map)

.. rubric:: Docstring

.. code-block:: text

   Configure a single-state interface and optionally load its dataset.

   Parameters
   ----------
   name : str
       Experiment identifier used to construct result filenames.
   path_dataset : str or array_like
       Pickled simulation-data path when ``training`` is true.  In
       evaluation-only mode it may instead supply energy data directly.
   fraction_training : float, optional
       Fraction of observations assigned to the training subset.
   training : bool, optional
       Whether coordinate data and simulation metadata are required.
   ic_map_class : type, optional
       Internal-coordinate map class constructed by :meth:`set_ic_map_step1`.


``NN_interface_sc.import_MD_dataset_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L604>`__

.. code-block:: python

   def import_MD_dataset_(self)

.. rubric:: Docstring

.. code-block:: text

   Load an MD dataset and reconstruct its molecular system.

   The method restores the :class:`SingleComponent` setup recorded with
   the simulation, exposes its reduced potential as ``u_``, and imports
   coordinates, box vectors, temperatures, and reduced energies.  The
   saved box is required to be constant across the NVT trajectory.


``NN_interface_sc.set_ic_map_step1`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L634>`__

.. code-block:: python

   def set_ic_map_step1(self, ind_root_atom=11, option=None)

.. rubric:: Docstring

.. code-block:: text

   self.sc.mol is available for this reason; to check the indices of atoms in the molecule
   once (ind_root_atom, option) pair is chosen, keep a fixed note of this for this molecule


``NN_interface_sc.set_ic_map_step2`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L647>`__

.. code-block:: python

   def set_ic_map_step2(self, inds_rand=None, check_PES=True)

.. rubric:: Docstring

.. code-block:: text

   Remove centre-of-mass motion and split the coordinate dataset.

   Parameters
   ----------
   inds_rand : array_like of int, optional
       Reusable permutation defining the training/validation split.
   check_PES : bool, optional
       Compare stored energies with the reconstructed potential after
       splitting.


``NN_interface_sc.truncate_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L666>`__

.. code-block:: python

   def truncate_data_(self, m=None)

.. rubric:: Docstring

.. code-block:: text

   Restrict the interface to ``m`` observations and rebuild its split.

   Parameters
   ----------
   m : int
       Number of observations retained.  A split of this size is selected
       first, then divided again according to ``fraction_training``.


``NN_interface_sc.set_ic_map_step3`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L685>`__

.. code-block:: python

   def set_ic_map_step3(self, n_mol_unitcell: int=1, COM_remover=WhitenFlow)

.. rubric:: Docstring

.. code-block:: text

   Fit the internal-coordinate map and verify its numerical inverse.

   Parameters
   ----------
   n_mol_unitcell : int, optional
       Number of molecules in the crystallographic unit cell underlying the
       simulated supercell.
   COM_remover : type, optional
       Flow layer used to whiten or remove centre-of-mass coordinates.

   Notes
   -----
   Inversion errors are reported on at most 1000 validation structures, and
   transformed-coordinate magnitudes are checked against the expected unit
   interval.


``NN_interface_sc_multimap`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L734>`__

.. code-block:: python

   class NN_interface_sc_multimap(name: str, paths_datasets: list, fraction_training: float=0.8, running_in_notebook: bool=False, training: bool=True, model_class=PGMcrys_v1, identity_init=False, ic_map_class=SingleComponent_map)

.. rubric:: Docstring

.. code-block:: text

   Coordinate a shared generative model across several metastable states.

   Each state owns an :class:`NN_interface_sc` instance and its own coordinate
   map.  A single model and trainer then learn across all states and can produce
   state-specific BAR estimates or joint MBAR differences.


``NN_interface_sc_multimap.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L742>`__

.. code-block:: python

   def __init__(self, name: str, paths_datasets: list, fraction_training: float=0.8, running_in_notebook: bool=False, training: bool=True, model_class=PGMcrys_v1, identity_init=False, ic_map_class=SingleComponent_map)

.. rubric:: Docstring

.. code-block:: text

   Construct one single-state interface for every supplied dataset.

   Parameters
   ----------
   name : str
       Shared experiment identifier.
   paths_datasets : list
       Dataset path or evaluation-only energy input for each state.
   fraction_training : float, optional
       Fraction of each dataset assigned to training.
   running_in_notebook : bool, optional
       Allow the trainer to use notebook-oriented progress displays.
   training : bool, optional
       Whether full coordinates and simulation systems will be loaded.
   model_class : type, optional
       Multi-map probabilistic generative model implementation.
   identity_init : bool, optional
       Initialise trainable transformations near the identity.
   ic_map_class : type, optional
       Internal-coordinate map used for each state.


``NN_interface_sc_multimap.save_inds_rand_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L794>`__

.. code-block:: python

   def save_inds_rand_(self)

.. rubric:: Docstring

.. code-block:: text

   Save each state's training/validation permutation separately.


``NN_interface_sc_multimap.load_inds_rand_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L800>`__

.. code-block:: python

   def load_inds_rand_(self)

.. rubric:: Docstring

.. code-block:: text

   Load the saved training/validation permutation for every state.


``NN_interface_sc_multimap.set_ic_map_step1`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L806>`__

.. code-block:: python

   def set_ic_map_step1(self, ind_root_atom=11, option=None)

.. rubric:: Docstring

.. code-block:: text

   Define atom ordering for every state's internal-coordinate map.

   Parameters
   ----------
   ind_root_atom : int, optional
       Root atom used to build the molecular coordinate tree.
   option : optional
       Alternative map-construction option forwarded unchanged to each map.


``NN_interface_sc_multimap.set_ic_map_step2`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L820>`__

.. code-block:: python

   def set_ic_map_step2(self, check_PES=True)

.. rubric:: Docstring

.. code-block:: text

   Remove centre-of-mass motion and split all state datasets.

   Parameters
   ----------
   check_PES : bool, optional
       Verify each dataset against its reconstructed potential.


``NN_interface_sc_multimap.set_ic_map_step3`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L830>`__

.. code-block:: python

   def set_ic_map_step3(self, n_mol_unitcells: list=None, COM_remover=WhitenFlow)

.. rubric:: Docstring

.. code-block:: text

   Initialise and test the internal-coordinate map for every state.

   Parameters
   ----------
   n_mol_unitcells : list of int or int, optional
       Molecule count of the underlying unit cell for each state.  A scalar
       is accepted only for a single-state interface.
   COM_remover : type, optional
       Centre-of-mass transformation used by each coordinate map.


``NN_interface_sc_multimap.set_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L866>`__

.. code-block:: python

   def set_model(self, learning_rate=0.001, evaluation_batch_size=5000, n_layers=4, DIM_connection=10, n_att_heads=4, identity_init=None, initialise=True, test_inverse=True)

.. rubric:: Docstring

.. code-block:: text

   Build the shared normalising-flow model and run an inverse check.

   Parameters
   ----------
   learning_rate : float, optional
       Initial optimiser learning rate.
   evaluation_batch_size : int, optional
       Number of structures used for model evaluation and sampling batches.
   n_layers : int, optional
       Number of coupling transformations in the model.
   DIM_connection : int, optional
       Width of the inter-component connection representation.
   n_att_heads : int, optional
       Number of attention heads used by the model.
   identity_init : bool, optional
       Override the interface-wide identity-initialisation setting.
   initialise : bool, optional
       Force model variable creation, useful for eager-mode debugging.
   test_inverse : bool, optional
       Test forward/inverse consistency on validation structures.


``NN_interface_sc_multimap.set_trainer`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L918>`__

.. code-block:: python

   def set_trainer(self, n_batches_between_evaluations=50)

.. rubric:: Docstring

.. code-block:: text

   Attach a trainer to the current model.

   Parameters
   ----------
   n_batches_between_evaluations : int, optional
       Training updates between diagnostic and free-energy evaluations.


``NN_interface_sc_multimap.u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L932>`__

.. code-block:: python

   def u_(self, r, k)

.. rubric:: Docstring

.. code-block:: text

   Evaluate the reduced potential of state ``k`` at coordinates ``r``.

   Parameters
   ----------
   r : array_like
       Batched Cartesian coordinates.
   k : int
       State index.

   Returns
   -------
   array_like
       Reduced potential energies returned by the state's energy function.


``NN_interface_sc_multimap.train`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L949>`__

.. code-block:: python

   def train(self, n_batches=2000, save_BAR=True, save_mBAR=False, save_misc=True, verbose=True, verbose_divided_by_n_mol=True, f_halfwindow_visualisation=1.0, test_inverse=False, evaluate_on_training_data=False, evaluate_main=True, training_batch_size=1000)

.. rubric:: Docstring

.. code-block:: text

   Train the shared model and optionally save evaluation artefacts.

   Parameters
   ----------
   n_batches : int, optional
       Number of optimisation batches.
   save_BAR, save_mBAR : bool, optional
       Save inputs required for later two-state BAR or multistate MBAR.
   save_misc : bool, optional
       Persist learning curves and trainer diagnostics after training.
   verbose : bool, optional
       Display live training plots.
   verbose_divided_by_n_mol : bool, optional
       Report per-molecule lattice rather than whole-crystal free energies.
   f_halfwindow_visualisation : float or list, optional
       Half-width of the displayed free-energy window.
   test_inverse : bool, optional
       Collect forward/inverse consistency diagnostics during training.
   evaluate_on_training_data : bool, optional
       Evaluate additional free-energy diagnostics on training observations.
   evaluate_main : bool, optional
       Enable the trainer's main periodic evaluation.
   training_batch_size : int, optional
       Number of configurations in each gradient update.


``NN_interface_sc_multimap.load_misc_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1057>`__

.. code-block:: python

   def load_misc_(self)

.. rubric:: Docstring

.. code-block:: text

   Restore trainer outputs and distribute state-specific curves.

   The saved arrays are copied into the corresponding single-state
   interfaces, where running free-energy means and standard deviations are
   reconstructed from validation estimates.


``NN_interface_sc_multimap.load_energies_during_training_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1077>`__

.. code-block:: python

   def load_energies_during_training_(self)

.. rubric:: Docstring

.. code-block:: text

   Load saved energy-overlap samples for every state.


``NN_interface_sc_multimap.plot_energies_during_training_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1081>`__

.. code-block:: python

   def plot_energies_during_training_(self, crystal_index=0, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Plot training-time energy overlap for one state.

   Parameters
   ----------
   crystal_index : int, optional
       State to plot.
   **kwargs
       Forwarded to the single-state plotting method.


``NN_interface_sc_multimap.solve_BAR_using_pymbar_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1093>`__

.. code-block:: python

   def solve_BAR_using_pymbar_(self, rerun=False)

.. rubric:: Docstring

.. code-block:: text

   Solve and aggregate two-state BAR estimates for every state.

   Parameters
   ----------
   rerun : bool, optional
       Recompute results even if saved outputs exist.


``NN_interface_sc_multimap.plot_result_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1104>`__

.. code-block:: python

   def plot_result_(self, crystal_index=0, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Plot the free-energy convergence result for one state.

   Parameters
   ----------
   crystal_index : int, optional
       State to plot.
   **kwargs
       Forwarded to :meth:`NN_interface_helper.plot_result_`.


``NN_interface_sc_multimap.solve_mBAR_using_pymbar_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1118>`__

.. code-block:: python

   def solve_mBAR_using_pymbar_(self, rerun=False, n_bootstraps=0, uncertainty_method=None, save_output=True)

.. rubric:: Docstring

.. code-block:: text

   Compute pairwise free-energy differences from saved multistate inputs.

   Parameters
   ----------
   rerun : bool, optional
       Ignore an existing saved MBAR output and recompute all evaluations.
   n_bootstraps : int, optional
       Number of bootstrap samples passed to :class:`pymbar.MBAR`.
   uncertainty_method : str, optional
       PyMBAR uncertainty method, such as ``'bootstrap'``.
   save_output : bool, optional
       Save ``estimates_mBAR`` after computation.

   Notes
   -----
   The result stores free-energy and standard-error matrices for training
   and validation data at every model-evaluation point.


``NN_interface_sc_multimap.sample_model_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1198>`__

.. code-block:: python

   def sample_model_(self, m, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Draw full evaluation-sized batches from one model state.

   Parameters
   ----------
   m : int
       Requested sample count.  Only
       ``floor(m / evaluation_batch_size) * evaluation_batch_size`` samples
       are returned.
   crystal_index : int, optional
       State from which to sample.

   Returns
   -------
   numpy.ndarray
       Concatenated Cartesian configurations generated by the model.


``NN_interface_sc_multimap.save_samples_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1218>`__

.. code-block:: python

   def save_samples_(self, m: int=20000)

.. rubric:: Docstring

.. code-block:: text

   Generate and save state-specific model samples.

   Parameters
   ----------
   m : int, optional
       Requested number of samples per state; see :meth:`sample_model_` for
       batch-size rounding behaviour.


``NN_interface_sc_multimap.load_samples_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1230>`__

.. code-block:: python

   def load_samples_(self, crystal_index=None)

.. rubric:: Docstring

.. code-block:: text

   Load saved model samples for one state or for all states.

   Parameters
   ----------
   crystal_index : int, optional
       State to load.  If omitted, ``samples_from_model`` becomes a list in
       state order; otherwise it is the selected state's array.


``NN_interface_sc_multimap_selective_evaluation`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1248>`__

.. code-block:: python

   class NN_interface_sc_multimap_selective_evaluation(*args, parent_class=default_parent, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   this should allow the interface_T.py things to also work with this if needed, selecting parent_class = one of them


``NN_interface_sc_multimap_selective_evaluation.__new__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1253>`__

.. code-block:: python

   def __new__(cls, *args, parent_class=default_parent, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Create a runtime subclass combining this mixin with ``parent_class``.

   Parameters
   ----------
   parent_class : type, optional
       Interface implementation that supplies dataset, model, and analysis
       behaviour.


``NN_interface_sc_multimap_selective_evaluation.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1265>`__

.. code-block:: python

   def __init__(self, *args, parent_class=default_parent, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Initialise the dynamically selected parent interface.

   Parameters
   ----------
   *args, **kwargs
       Forwarded to ``parent_class.__init__``.
   parent_class : type, optional
       Consumed by :meth:`__new__`; documented here for a consistent public
       constructor signature.


``NN_interface_sc_multimap_selective_evaluation.train`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1282>`__

.. code-block:: python

   def train(self, n_batches=2000, verbose=True, verbose_divided_by_n_mol=True, f_halfwindow_visualisation=1.0, test_inverse=False, training_batch_size=1000)

.. rubric:: Docstring

.. code-block:: text

   Train while postponing potential evaluation of generated samples.

   Parameters
   ----------
   n_batches : int, optional
       Number of optimisation batches.
   verbose : bool, optional
       Display live diagnostic plots.
   verbose_divided_by_n_mol : bool, optional
       Display per-molecule rather than whole-crystal free energies.
   f_halfwindow_visualisation : float or list, optional
       Vertical half-width used by training plots.
   test_inverse : bool, optional
       Collect model inversion diagnostics.
   training_batch_size : int, optional
       Number of configurations in each gradient update.

   Notes
   -----
   BAR metadata and generated configurations are saved, but state potential
   functions are deliberately withheld from the trainer.  Expensive model-
   sample energies are evaluated later only for selected batches.


``NN_interface_sc_multimap_selective_evaluation.solve_BAR_using_pymbar_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1357>`__

.. code-block:: python

   def solve_BAR_using_pymbar_(self, rerun=False, n_selective_evalautions=5)

.. rubric:: Docstring

.. code-block:: text

   Evaluate BAR only at batches with the best validation performance.

   Parameters
   ----------
   rerun : bool, optional
       Retained for compatibility; selective state solves currently force
       recomputation so newly evaluated batches are included.
   n_selective_evalautions : int, optional
       Number of lowest-validation-loss batches whose model samples receive
       an explicit potential-energy evaluation.


``NN_interface_sc_multimap_selective_evaluation.reset_final_result_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1378>`__

.. code-block:: python

   def reset_final_result_(self, obj)

.. rubric:: Docstring

.. code-block:: text

   Rebuild final BAR curves using only successfully evaluated batches.

   Parameters
   ----------
   obj : NN_interface_sc
       State interface whose raw estimates and reported final values are
       updated.  Unevaluated curve positions are filled with ``NaN`` for
       plotting.


``NN_interface_sc_multimap_selective_evaluation.method_for_selective_evalaution_v1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1419>`__

.. code-block:: python

   def method_for_selective_evalaution_v1_(self, obj, index_of_state, AVMD_V)

.. rubric:: Docstring

.. code-block:: text

   Compute validation BAR estimates for the best saved model batches.

   Parameters
   ----------
   obj : NN_interface_sc
       State interface holding saved BAR inputs and receiving the estimates.
   index_of_state : int
       State whose potential function evaluates generated configurations.
   AVMD_V : array_like
       Validation free-energy diagnostic by evaluation batch.  Its negative
       is used as the selection loss.

   Notes
   -----
   Completed batch results are cached individually.  ``inds_solved_current``
   records all unmasked validation estimates available after the operation.


``NN_interface_sc_multimap_selective_evaluation_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/interface.py#L1496>`__

.. code-block:: python

   def NN_interface_sc_multimap_selective_evaluation_(name, n_states=1, list_r=None, list_b0=None, list_u=None, list_u_=None, single_mol_pdb_file=None, training=False, fraction_training=0.8, running_in_notebook=True, parent_class=NN_interface_sc_multimap, model_class=PGMcrys_v1, ic_map_class=SingleComponent_map)

.. rubric:: Docstring

.. code-block:: text

   Build a selective-evaluation interface from in-memory state data.

   Parameters
   ----------
   name : str
       Experiment identifier used in saved-result filenames.
   n_states : int, optional
       Number of metastable states.
   list_r : list of array_like, optional
       Cartesian MD coordinates for each state, shaped ``(frames, atoms, 3)``
       in nanometres.
   list_b0 : list of array_like, optional
       Static ``(3, 3)`` simulation box for each state, in nanometres.
   list_u : list of array_like, optional
       Reduced whole-crystal potential energies for each MD trajectory.
   list_u_ : list of callable, optional
       Batched reduced-potential functions, one per state.
   single_mol_pdb_file : str, optional
       PDB file describing one molecule for internal-coordinate construction.
   training : bool, optional
       Populate the complete in-memory datasets needed for training.  If false,
       only supplied potential functions are attached.
   fraction_training : float, optional
       Fraction of each in-memory trajectory assigned to training.
   running_in_notebook : bool, optional
       Configure notebook-oriented trainer output.
   parent_class : type, optional
       Base interface dynamically combined with selective evaluation.
   model_class : type, optional
       Shared generative-model implementation.
   ic_map_class : type, optional
       Internal-coordinate map implementation.

   Returns
   -------
   NN_interface_sc_multimap_selective_evaluation
       Configured dynamic interface.  Coordinate maps, model, and trainer still
       need to be initialised through their normal setup methods.
