.. _api-O-NN-model_helper:

O.NN.model_helper
=================

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py>`__

.. rubric:: Docstring

.. code-block:: text

   Common model, persistence, training, and free-energy utilities.

   The model mixin wraps TensorFlow forward/inverse maps with NumPy-facing APIs,
   tracks Jacobians, and implements a versioned single-file persistence format.
   The estimator and trainer routines compare molecular-dynamics samples with
   generated samples using likelihood, exponential averaging, BAR, and MBAR
   quantities. Energies and free energies are reduced and expressed in ``kT``.


Classes and functions
---------------------

``no_nans_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L21>`__

.. code-block:: python

   def no_nans_(grads: list)

.. rubric:: Docstring

.. code-block:: text

   Return whether every supplied gradient tensor contains finite values.


``_BoundMethodRef`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L29>`__

.. code-block:: python

   class _BoundMethodRef(name: str)

.. rubric:: Docstring

.. code-block:: text

   Serializable placeholder for an allow-listed bound method name.


``_BoundMethodRef.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L32>`__

.. code-block:: python

   def __init__(self, name: str)

.. rubric:: Docstring

.. code-block:: text

   Store the method name for restoration after deserialization.


``model_helper`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L47>`__

.. code-block:: python

   class model_helper()

.. rubric:: Docstring

.. code-block:: text

   Mixin providing training and persistence around an invertible model.

   Subclasses must implement ``sample_base_``, ``ln_base_``, ``forward_``, and
   ``inverse_``. Forward/inverse functions return transformed values and
   per-sample log-absolute Jacobian determinants.


``model_helper.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L54>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   the class which inherits these methods should have the following methods:

   sample_base_ : m   -> z         ; z : list or tensors
   ln_base_     : z   -> ln_p0     ; tensor which shape (m,1)

   forward_     : xyz -> z, ladJ   ; z : list, ladJ : tensor which shape (m,1)
   inverse_     : z   -> xyz, ladJ ; zyz : list, ladJ : tensor which shape (m,1)


``model_helper._serialize_obj_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L66>`__

.. code-block:: python

   def _serialize_obj_(x)

.. rubric:: Docstring

.. code-block:: text

   GitHub Copilot written


``model_helper._deserialize_obj_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L127>`__

.. code-block:: python

   def _deserialize_obj_(x)

.. rubric:: Docstring

.. code-block:: text

   GitHub Copilot written


``model_helper._restore_bound_methods_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L174>`__

.. code-block:: python

   def _restore_bound_methods_(obj)

.. rubric:: Docstring

.. code-block:: text

   Resolve serialized method placeholders and legacy strategy aliases.


``model_helper.initialise_weights_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L197>`__

.. code-block:: python

   def initialise_weights_(self)

.. rubric:: Docstring

.. code-block:: text

   Build Keras weights by sending one base sample through the inverse map.


``model_helper.reset_optimiser_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L204>`__

.. code-block:: python

   def reset_optimiser_(self, optimiser_LR_decay: list)

.. rubric:: Docstring

.. code-block:: text

   Replace the optimiser with a fresh Adam instance.

   The two-element input is stored as learning rate and decay rate; the
   current TensorFlow optimiser uses only the learning rate.


``model_helper.ln_model_for_step_ML_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L214>`__

.. code-block:: python

   def ln_model_for_step_ML_(self, inputs: list)

.. rubric:: Docstring

.. code-block:: text

   Return latent variables and model log density for one tensor batch.


``model_helper.step_ML_graph_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L221>`__

.. code-block:: python

   def step_ML_graph_(self, inputs_batch: list)

.. rubric:: Docstring

.. code-block:: text

   Perform one maximum-likelihood gradient update.

   Non-finite gradients are detected and skipped. Returns negative mean
   log likelihood and a Boolean indicating whether the update was applied.


``model_helper.forward_graph_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L243>`__

.. code-block:: python

   def forward_graph_(self, inputs: list)

.. rubric:: Docstring

.. code-block:: text

   TensorFlow-graph-compatible wrapper around the subclass forward map.


``model_helper.inverse_graph_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L248>`__

.. code-block:: python

   def inverse_graph_(self, inputs: list)

.. rubric:: Docstring

.. code-block:: text

   TensorFlow-graph-compatible wrapper around the subclass inverse map.


``model_helper.initialise_graphs_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L253>`__

.. code-block:: python

   def initialise_graphs_(self, re_initialise: bool=False)

.. rubric:: Docstring

.. code-block:: text

   Compile forward, inverse, and training-step methods with ``tf.function``.

   With ``re_initialise=True``, existing instance wrappers are removed
   before tracing again. Stable input shapes reduce expensive retracing.


``model_helper.forward`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L275>`__

.. code-block:: python

   def forward(self, r)

.. rubric:: Docstring

.. code-block:: text

   Convert NumPy-like Cartesian input and evaluate the compiled forward map.


``model_helper.inverse`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L279>`__

.. code-block:: python

   def inverse(self, inputs: list)

.. rubric:: Docstring

.. code-block:: text

   Evaluate the compiled inverse map on base/model variables.


``model_helper.step_ML`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L283>`__

.. code-block:: python

   def step_ML(self, r, u: np.ndarray=None, batch_size: int=1000)

.. rubric:: Docstring

.. code-block:: text

   Train on a random coordinate minibatch and report loss estimates.

   Sampling is without replacement. Returns ``(AVMD_T_f, AVMD_T_s)`` when
   energies ``u`` are provided; without energies both entries are the
   negative-log-likelihood loss.


``model_helper.ln_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L302>`__

.. code-block:: python

   def ln_model(self, r: list)

.. rubric:: Docstring

.. code-block:: text

   Evaluate the normalised model log density ``ln q(r)``.


``model_helper.sample_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L309>`__

.. code-block:: python

   def sample_model(self, m: int)

.. rubric:: Docstring

.. code-block:: text

   Generate ``m`` Cartesian samples and their model log densities.


``model_helper.print_model_size`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L317>`__

.. code-block:: python

   def print_model_size(self)

.. rubric:: Docstring

.. code-block:: text

   Calculate, store, and print trainable parameter counts and shapes.


``model_helper.save_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L328>`__

.. code-block:: python

   def save_model(self, path_and_name: str)

.. rubric:: Docstring

.. code-block:: text

   Modified with GitHub Copilot for TF 2.19


``model_helper._is_new_model_artifact_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L351>`__

.. code-block:: python

   def _is_new_model_artifact_(path_and_name: str)

.. rubric:: Docstring

.. code-block:: text

   Written with GitHub Copilot for TF 2.19


``model_helper._load_model_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L366>`__

.. code-block:: python

   def _load_model_(path_and_name: str, class_of_the_model)

.. rubric:: Docstring

.. code-block:: text

   in each class_of_the_model have a method:
   @staticmethod
   def load_model(path_and_name : str):
       return class_of_the_model.load_model_(path_and_name, class_of_the_model)

   Modified with GitHub Copilot for TF 2.19


``model_helper.test_inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L395>`__

.. code-block:: python

   def test_inverse_(self, r, graph=True)

.. rubric:: Docstring

.. code-block:: text

   Measure coordinate and Jacobian round-trip errors in both directions.

   The forward test maps validation coordinates ``r -> z -> r``; the
   backward test maps fresh base samples ``z -> r -> z``. Returned nested
   summaries contain coordinate mean/max and Jacobian mean/min/max errors.


``model_helper.index_transferable_parameters_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L448>`__

.. code-block:: python

   def index_transferable_parameters_(self)

.. rubric:: Docstring

.. code-block:: text

   Locate trainable tensors whose TensorFlow name contains ``TRANSFERABLE``.


``model_helper.save_transferable_parameters_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L457>`__

.. code-block:: python

   def save_transferable_parameters_(self, path_and_name)

.. rubric:: Docstring

.. code-block:: text

   Serialize transferable-model metadata and selected weights.


``model_helper.load_transferable_parameters_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L462>`__

.. code-block:: python

   def load_transferable_parameters_(self, path_and_name)

.. rubric:: Docstring

.. code-block:: text

   Load compatible transferable weights and rebuild compiled graphs.


``model_helper.set_transferable_parameters_fixed_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L474>`__

.. code-block:: python

   def set_transferable_parameters_fixed_(self)

.. rubric:: Docstring

.. code-block:: text

   Exclude indexed transferable tensors from subsequent optimisation.


``model_helper.set_transferable_parameters_trainable_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L483>`__

.. code-block:: python

   def set_transferable_parameters_trainable_(self)

.. rubric:: Docstring

.. code-block:: text

   Restore optimisation over every model trainable variable.


``model_helper.initialise`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L495>`__

.. code-block:: python

   def initialise(self)

.. rubric:: Docstring

.. code-block:: text

   Build weights, optimiser, compiled graphs, and transferability index.


``pool_`` (function)
^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L504>`__

.. code-block:: python

   def pool_(x, ws=None)

.. rubric:: Docstring

.. code-block:: text

   Output: weighted average of x


``tolBAR_`` (function)
^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L509>`__

.. code-block:: python

   def tolBAR_(incuA, incuB, wsA=None, wsB=None, f_window_grain=[-180000.0, 60000.0, 30000], tol=0.0001, return_errs=False)

.. rubric:: Docstring

.. code-block:: text

   This is only used as a double check. Final FE results are always solved using pymbar

   local implementation of 2-state BAR, finding f (via grid search) that best fits the BAR equality A = B below.
       'tol' is tolerance of the best fit (absolute error |A-B|)

   No standard error method is included here, therefore cannot be taken as a final result.

   Inputs as used in current work:
       f is absolute FE

       incuA : (m,1) shaped array = \phi(r) = u(r) + ln(p(r)) ; r ~ \mu
           \mu = \mu(r) = exp(-u(r)) / Z, is the MD distribution of NVT data with 
               unknown normalisation constant Z, but we have m ergodic samples r ~ \mu
               Want output f to be as close as possible to underlying FE -ln(Z).

       incuB : (m,1) shaped array = \phi(r) = u(r) + ln(p(r)) ; r ~ p
           p is any normalised distribution that is similar to \mu,
           Should be able to evaluate ln(p) exactly, and sample m ergodic samples from it r ~ p

       wsA : was the sampling from \mu (no bias)? YES: wsA = None, NO : wsA = weights to reweight the bias.
       wsB : is the sampling from p ergodic? Yes by default: wsB = None.
       f_window_grain : list [float, float, int] = [a ,b, grain]
           a = minimum f that may be valid in this system
           b = maximum f that may be valid in this system
           grain = how many values on a grid to try between a and b (probably does not make much difference given tol)
   Output:
       f : scalar : estimate of -ln(Z)


``tolBAR_._BAR_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L546>`__

.. code-block:: python

   def _BAR_(grid_f, incuA, incuB, wsA=None, wsB=None)

.. rubric:: Docstring

.. code-block:: text

   Evaluate BAR equality error on a free-energy grid and select its minimum.


``get_FE_estimates_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L615>`__

.. code-block:: python

   def get_FE_estimates_(model, r_training, r_validation, name_save_BAR_inputs: str, u_training, u_validation, u_function_=None, w_training=None, w_validation=None, crystal_index: int=0, evaluation_batch_size=5000, shuffle=True, test_inverse=False, evaluate_on_training_data=False, save_generated_configurations_anyway=False)

.. rubric:: Docstring

.. code-block:: text

   FE evalaution from the model:

   During training every so often this function is ran on each macrostate being trained.

   The overall cost of this function therefore can be high, but 
   can this be reduced to just a validation loss evalaution (does not require a FF).
   If the FF is cheap to evaluate, and want FEs as effeciently as possible:
       use only the top 7 to 9 arguments, with other arguments left as default.

   Inputs (arguments):
       model         : instace of PGM model with methods: ln_model, sample_model, and test_inverse_
       r_training    : (m_T,N,3) : Cartesian coordiantes of the training data (dataset contains m_T configurations)
       r_validation  : (m_V,N,3) : Cartesian coordiantes of the validation data (dataset contains m_V configurations)
       name_save_BAR_inputs : path and name where to save pymbar inputs for BAR 
           (saved with suffix added at the end of the name: '_V' ; validation, '_T' ; training)
       u_training    : (m_T,1)   : potential energies of the training data
       u_validation  : (m_T,1)   : potential energies of the validation data
       u_function_   : potential energy function of the current macrostate
       w_training    : (m,1)     : weights of training data (only if the MD data sampled from a biased ensemble)
       w_validation  : (m,1)     : weights of validation data (only if the MD data sampled from a biased ensemble)
       crystal_index : in multimap model, index of the current macrostate. (! placeholder is 0, no warnings)
       evaluation_batch_size : number of samples to include for all types of FE estiamates
       shuffle       : default True, otherwise evaluating only on the first evaluation_batch_size configurations
       test_inverse  : extra cost if True, but useful when testing a model (allows plottin inversion accuracy during training)
       evaluate_on_training_data : extra cost if True, default False (not necesary for general use)
       save_generated_configurations_anyway : optional, default is False (saving only the small energy arrays, saving disk space)
   Outputs:
       running_estimates : abbreviations defined in doi/10.1021/acs.jctc.4c00520 (*P1)
           all elements of this output can be undestood by refering to the short lines they were defined
       inv_test_result   : [None] if test_inverse False, otherwise depends the outputs from model.test_inverse_


``FE_of_model_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L811>`__

.. code-block:: python

   def FE_of_model_(AVMD_V, BAR_V)

.. rubric:: Docstring

.. code-block:: text

   weighted average of raw BAR_V FE estimates 
   the weights = np.exp(AVMD_V) ; higher is better (lower validation error)
       motivation for heuristic : since AVMD_V is a type of FE estimate, exponentiating it a type of weight >= 0
   REF: doi/10.1021/acs.jctc.4c01612 (P2, defined also in P3)

   Inputs:
       AVMD_V : (n_evaluation_batches, 1) array of raw FE estimates based on AVMD_V
       BAR_V  : (n_evaluation_batches, 1) array of raw estimates based on BAR_V
   Output:
       av_BAR_V : scalar value : averaged BAR_V FE from the training run


``FE_of_model_curve_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L829>`__

.. code-block:: python

   def FE_of_model_curve_(AVMD_V, BAR_V)

.. rubric:: Docstring

.. code-block:: text

   cumulative weighted average using FE_of_model_
   Inputs: same as in FE_of_model_
   Output: (n_evaluation_batches, ) array of *averaged BAR_V FEs from the training run
       *(averaged up to each evaluation batch)


``get_phi_ij_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L842>`__

.. code-block:: python

   def get_phi_ij_(model, list_r, list_potential_energy_functions, evalation_sample_size=5000, shuffle=True)

.. rubric:: Docstring

.. code-block:: text

   composition of maps r^{[i]} -> z -> r^{[j]} ; states i,j = 0,...,n_states

       n_states = len(list_r) = len(list_potential_energy_functions)

   Inputs:
       model : instace of PGM model with methods: forward, inverse
           both methods need to have crystal_index : int as argument
       list_r : list of MD datasets ergodically sampled in n_states different states
       list_potential_energy_functions : list of corresponding potential energy functions
       evalation_sample_size : number of samples to evalaute (here this is the same number in each direction)
       shuffle : default True, otherwise evaluating only on the first evaluation_batch_size configurations
   Outputs:
       phi_ij : MBAR input for pymbar to compute FE differences between all pairs of states


``TRAINER`` (class)
^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L886>`__

.. code-block:: python

   class TRAINER(model, max_training_batches: int=50000, n_batches_between_evaluations: int=50, running_in_notebook: bool=False)

.. rubric:: Docstring

.. code-block:: text

   Orchestrate multi-state likelihood training and periodic FE evaluation.

   The trainer owns progress arrays, evaluation schedules, BAR/MBAR inputs,
   convergence plots, and inversion diagnostics. It delegates actual gradient
   updates and transformations to a compatible PGM model.


``TRAINER.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L893>`__

.. code-block:: python

   def __init__(self, model, max_training_batches: int=50000, n_batches_between_evaluations: int=50, running_in_notebook: bool=False)

.. rubric:: Docstring

.. code-block:: text

   takes the model + MD data, to train, and evaluate the model during training
   Inputs:
       model : instace of PGM model with methods: 
           step_ML, forward, inverse, ln_model, sample_model, and test_inverse_
       max_training_batches : any large number 
           to allocate long enough arrays into which output numbers are stored during training
       n_batches_between_evaluations : evaluation stride
           number of training batches between which model is evaluated for FE estimate(s)
       running_in_notebook : True is better if running the training in a jupyter notebook
   Output:
       None
           NB: if running_in_notebook, the cell in which TRAINER is initialised (here)
           is the cell where progress of training, and FE estiamte(s) printed using text


``TRAINER.print_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L949>`__

.. code-block:: python

   def print_(self, text)

.. rubric:: Docstring

.. code-block:: text

   Update notebook display output or overwrite terminal progress text.


``TRAINER.train`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L954>`__

.. code-block:: python

   def train(self, n_batches: int, list_r_training: list, list_r_validation: list, list_u_training: list, list_u_validation: list, list_w_training: list=None, list_w_validation: list=None, list_potential_energy_functions: list=None, training_batch_size=1000, evaluation_batch_size=5000, evaluate_main=True, name_save_BAR_inputs=None, name_save_mBAR_inputs=None, shuffle=True, f_halfwindow_visualisation=0.5, verbose=True, verbose_divided_by_n_mol=True, evaluate_on_training_data=False, test_inverse=False, save_generated_configurations_anyway=False)

.. rubric:: Docstring

.. code-block:: text

   Run model-training batches with scheduled free-energy diagnostics.

   One coordinate/energy dataset pair is required per model state. Optional
   weights support biased MD data, and potential-energy callables enable
   generated-sample BAR/MBAR estimates. The method mutates training state,
   may save estimator inputs and plots progress, and returns ``None``.
   ``n_batches`` must fit within the arrays allocated at construction.


``TRAINER.estimates`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L1225>`__

.. code-block:: python

   def estimates(self)

.. rubric:: Docstring

.. code-block:: text

   Free-energy diagnostic tensor truncated to completed evaluations.


``TRAINER.evaluation_grid`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L1230>`__

.. code-block:: python

   def evaluation_grid(self)

.. rubric:: Docstring

.. code-block:: text

   Training-batch indices at which completed evaluations occurred.


``TRAINER.AVMD_T_f`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L1235>`__

.. code-block:: python

   def AVMD_T_f(self)

.. rubric:: Docstring

.. code-block:: text

   Training-set likelihood-derived estimates through the current batch.


``TRAINER.save_the_above_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L1239>`__

.. code-block:: python

   def save_the_above_(self, name: str)

.. rubric:: Docstring

.. code-block:: text

   Serialize accumulated FE curves, diagnostics, and training time.


``TRAINER.save_inv_test_results_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L1250>`__

.. code-block:: python

   def save_inv_test_results_(self, name: str)

.. rubric:: Docstring

.. code-block:: text

   Serialize inversion-test histories for every state.


``plot_inv_test_res_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/model_helper.py#L1256>`__

.. code-block:: python

   def plot_inv_test_res_(inv_test_result, mean_range=[True, True], forward_inverse=[True, True], plot_during_training=True)

.. rubric:: Docstring

.. code-block:: text

   Print and optionally plot Jacobian round-trip error histories.

   ``mean_range`` selects mean and min/max summaries; ``forward_inverse``
   selects Cartesian-to-base and base-to-Cartesian round trips. Reported
   numbers compare averages over the first and last ten recorded evaluations.
