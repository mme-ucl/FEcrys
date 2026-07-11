.. _api-O-MM-Tx:

O.MM.Tx
=======

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py>`__

.. rubric:: Docstring

.. code-block:: text

   Propagate polymorph free energies across temperature with NPT MBAR data.

   The active implementation loads temperature-replica trajectories for one
   polymorph, constructs reduced enthalpies at every sampled temperature, and uses
   MBAR to interpolate Gibbs free energy and enthalpy.  Two ``g_of_T`` instances
   can subsequently be compared to locate a polymorph transition temperature.


Classes and functions
---------------------

``SingleComponent_lite`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L14>`__

.. code-block:: python

   class SingleComponent_lite(name, check_energies=True)

.. rubric:: Docstring

.. code-block:: text

   Read only the trajectory fields needed for temperature reweighting.


``SingleComponent_lite.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L17>`__

.. code-block:: python

   def __init__(self, name, check_energies=True)

.. rubric:: Docstring

.. code-block:: text

   Load a saved NPT dataset without retaining a full simulation object.

   Parameters
   ----------
   name : str
       Path or pickle prefix of the saved simulation dataset.
   check_energies : bool, optional
       Reconstruct the molecular system and compare the first 50 stored
       reduced energies with fresh evaluations.


``all_lower_triangular_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L55>`__

.. code-block:: python

   def all_lower_triangular_(boxes)

.. rubric:: Docstring

.. code-block:: text

   Return whether one or more box matrices are lower triangular.

   Parameters
   ----------
   boxes : array_like
       One ``(3, 3)`` box or a batch with box axes in the final two dimensions.

   Returns
   -------
   bool
       True when every strictly upper-triangular component is below ``1e-5``.


``g_of_T`` (class)
^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L73>`__

.. code-block:: python

   class g_of_T(Tref: float=300, Tref_FE: float=0.0, Tref_SE: float=0.0, Tref_box: np.ndarray=None, paths_datasets_NPT: list=[], check_energies=True, f2g_correction_params: dict={'version': 1, 'bins': 40})

.. rubric:: Docstring

.. code-block:: text

   Interpolate one polymorph's Gibbs free energy over temperature.

   The reference free energy anchors the absolute curve.  NPT trajectories at
   several temperatures provide the reduced enthalpy matrix used by MBAR.
   Reported crystal quantities are dimensionless in units of ``kT`` unless a
   method explicitly divides them by ``n_mol`` to give lattice values.


``g_of_T.set_Tref_g_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L85>`__

.. code-block:: python

   def set_Tref_g_(self, Tref_FE=None, version=1, bins: int=40)

.. rubric:: Docstring

.. code-block:: text

   Convert a reference Helmholtz free energy to Gibbs free energy.

   Parameters
   ----------
   Tref_FE : float, optional
       Replacement whole-crystal Helmholtz free energy at ``Tref``, in
       reduced units.
   version : int, optional
       Box-density model: ``0`` omits the density term, ``1`` models volume,
       ``2`` models the diagonal box elements, and ``3`` additionally models
       the lower off-diagonal elements.
   bins : int, optional
       Number of bins in each one-dimensional density histogram.

   Notes
   -----
   The correction is ``beta * P * V_ref + log p(box_ref)``.  It is stored in
   ``Tref_f_to_g_correction`` and added to ``Tref_FE`` to form ``Tref_g``.


``g_of_T.set_Tref_g_.log_histogram_1D_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L117>`__

.. code-block:: python

   def log_histogram_1D_(x, data, bins=40)

.. rubric:: Docstring

.. code-block:: text

   Estimate log density at ``x`` from a one-dimensional histogram.


``g_of_T.set_Tref_g_.log_1D_model_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L127>`__

.. code-block:: python

   def log_1D_model_(x, data)

.. rubric:: Docstring

.. code-block:: text

   Evaluate the configured histogram density model.


``g_of_T.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L157>`__

.. code-block:: python

   def __init__(self, Tref: float=300, Tref_FE: float=0.0, Tref_SE: float=0.0, Tref_box: np.ndarray=None, paths_datasets_NPT: list=[], check_energies=True, f2g_correction_params: dict={'version': 1, 'bins': 40})

.. rubric:: Docstring

.. code-block:: text

   This whole class deals with only one polymorph. To get Tx, need at least two instances.


``g_of_T.__init__.gR_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L203>`__

.. code-block:: python

   def gR_(_bool)

.. rubric:: Docstring

.. code-block:: text

   Choose the status colour associated with a convergence flag.


``g_of_T.set_enthalpies_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L235>`__

.. code-block:: python

   def set_enthalpies_(self)

.. rubric:: Docstring

.. code-block:: text

   Assemble sampled and cross-temperature reduced enthalpies.

   ``evaluations[(Ti, Tj)]`` is the reduced enthalpy evaluated at target
   temperature ``Ti`` on configurations sampled at ``Tj``.  The separate
   ``evaluations_parts`` dictionaries retain potential, pressure-volume,
   and anisotropic-box Jacobian contributions.  Sampled kinetic energies
   are also prepared for optional heat-capacity estimates.


``g_of_T.average_sampled_enthalpy_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L320>`__

.. code-block:: python

   def average_sampled_enthalpy_(self, T, m=None)

.. rubric:: Docstring

.. code-block:: text

   Return the mean reduced enthalpy at a sampled temperature.

   Parameters
   ----------
   T : float
       Sampled temperature in kelvin.
   m : int, optional
       Use only the final ``m`` observations.  ``None`` uses all data.


``g_of_T.maximum_batch_size`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L335>`__

.. code-block:: python

   def maximum_batch_size(self)

.. rubric:: Docstring

.. code-block:: text

   int: Smallest available trajectory length across temperatures.


``g_of_T.ANI`` (method)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L343>`__

.. code-block:: python

   def ANI(self)

.. rubric:: Docstring

.. code-block:: text

   str: Filename marker used for anisotropic-box calculations.


``g_of_T.save_mbar_instance_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L348>`__

.. code-block:: python

   def save_mbar_instance_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Save the MBAR estimator and free-energy-difference result.

   Parameters
   ----------
   m : int
       Number of observations used per temperature state.


``g_of_T.load_mbar_instance_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L359>`__

.. code-block:: python

   def load_mbar_instance_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Load a previously saved sequential-subset MBAR calculation.

   Parameters
   ----------
   m : int
       Per-state sample count encoded in the filename.


``g_of_T.save_mbar_instance_shuffled_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L370>`__

.. code-block:: python

   def save_mbar_instance_shuffled_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Save a representative-subset MBAR calculation and its indices.

   Parameters
   ----------
   m : int
       Number of representative observations selected per state.


``g_of_T.load_mbar_instance_shuffled_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L381>`__

.. code-block:: python

   def load_mbar_instance_shuffled_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Load representative-subset MBAR state and selection indices.

   Parameters
   ----------
   m : int
       Per-state sample count encoded in the filename.


``g_of_T.get_inds_select_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L395>`__

.. code-block:: python

   def get_inds_select_(self, uii, m)

.. rubric:: Docstring

.. code-block:: text

   Select an energy-representative subset of trajectory indices.

   Parameters
   ----------
   uii : array_like
       Reduced enthalpies evaluated and sampled at the same temperature.
   m : int
       Required number of indices.

   Returns
   -------
   numpy.ndarray
       The first ``m`` indices of a split accepted by
       :func:`find_split_indices_`, or every available index when ``m`` is
       the maximum batch size.


``g_of_T._set_evaluations_subsample_selection_indices_SHUFFLED_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L424>`__

.. code-block:: python

   def _set_evaluations_subsample_selection_indices_SHUFFLED_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Choose a representative ``m``-sample subset for every temperature.


``g_of_T._set_evaluations_subsample_selection_indices_basic_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L431>`__

.. code-block:: python

   def _set_evaluations_subsample_selection_indices_basic_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Choose the final ``m`` trajectory observations at every temperature.


``g_of_T.set_evaluations_subsampled_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L440>`__

.. code-block:: python

   def set_evaluations_subsampled_(self)

.. rubric:: Docstring

.. code-block:: text

   Apply each source state's selected indices to all target evaluations.


``g_of_T.compute_MBAR_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L451>`__

.. code-block:: python

   def compute_MBAR_(self, m=None, rerun=False, save=True, use_representative_subsets=False)

.. rubric:: Docstring

.. code-block:: text

   Fit or restore the multitemperature MBAR estimator.

   Parameters
   ----------
   m : int, optional
       Observations per temperature.  If omitted, all observations up to
       the shortest trajectory length are used.
   rerun : bool, optional
       Fit MBAR even when a matching saved estimator exists.
   save : bool, optional
       Persist newly fitted MBAR state.
   use_representative_subsets : bool, optional
       Select distribution-matched observations across each full trajectory;
       otherwise use its final ``m`` observations.

   Notes
   -----
   ``Q[i]`` concatenates the reduced enthalpy at target temperature ``i``
   over samples from every source temperature.  ``Ns`` records the number
   of samples contributed by each source state.


``g_of_T.finalise_patch_related_to_ladJ_should_not_be_rescaled_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L555>`__

.. code-block:: python

   def finalise_patch_related_to_ladJ_should_not_be_rescaled_(self)

.. rubric:: Docstring

.. code-block:: text

   Prepare anisotropic box-Jacobian terms that must not scale with beta.

   The selected Jacobian data are concatenated into ``not_Q`` so
   :meth:`g_` and :meth:`av_u_` can rescale energetic terms with temperature
   while leaving the dimensionless measure correction unchanged.


``g_of_T.mbar_sample_size`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L575>`__

.. code-block:: python

   def mbar_sample_size(self)

.. rubric:: Docstring

.. code-block:: text

   int: Number of samples assigned to each MBAR temperature state.


``g_of_T.n_energy_evalautions`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L580>`__

.. code-block:: python

   def n_energy_evalautions(self)

.. rubric:: Docstring

.. code-block:: text

   int: Number of off-diagonal cross-temperature energy evaluations.


``g_of_T._mbar`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L587>`__

.. code-block:: python

   def _mbar(self)

.. rubric:: Docstring

.. code-block:: text

   pymbar.MBAR: Deep copy of the fitted estimator for isolated queries.


``g_of_T.g_`` (method)
^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L594>`__

.. code-block:: python

   def g_(self, T)

.. rubric:: Docstring

.. code-block:: text

   Interpolate absolute whole-crystal Gibbs free energy at ``T``.

   Parameters
   ----------
   T : float
       Target temperature in kelvin.

   Returns
   -------
   FE : float
       Gibbs free energy in reduced whole-crystal units.
   SE : float
       Corresponding propagated standard error.


``g_of_T.g`` (method)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L627>`__

.. code-block:: python

   def g(self)

.. rubric:: Docstring

.. code-block:: text

   g_{crys}(T \in self.Ts) : discrete gibbs FE estimates as a function of temperature


``g_of_T.av_u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L636>`__

.. code-block:: python

   def av_u_(self, T)

.. rubric:: Docstring

.. code-block:: text

   Interpolate the mean whole-crystal reduced enthalpy at ``T``.

   Parameters
   ----------
   T : float
       Target temperature in kelvin.

   Returns
   -------
   float
       MBAR expectation of enthalpy at the target temperature.


``g_of_T._test_average_enthalpy_interpolator_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L686>`__

.. code-block:: python

   def _test_average_enthalpy_interpolator_(self, m=None)

.. rubric:: Docstring

.. code-block:: text

   Measure interpolation error at the sampled temperatures.

   Parameters
   ----------
   m : int, optional
       Number of final observations used in each direct sampled mean.

   Returns
   -------
   float
       Maximum absolute difference per molecule between MBAR-interpolated
       and directly sampled reduced enthalpy.


``g_of_T.curve_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L708>`__

.. code-block:: python

   def curve_(self, Tmin, Tmax, Tstride=100, what='g')

.. rubric:: Docstring

.. code-block:: text

   Evaluate a per-molecule thermodynamic curve on a temperature grid.

   Parameters
   ----------
   Tmin, Tmax : float
       Inclusive endpoints of the temperature grid in kelvin.
   Tstride : int, optional
       Number of evenly spaced grid points, despite the historical name.
   what : {'g', 'u', 's'}, optional
       Quantity to evaluate: Gibbs free energy, mean enthalpy, or the
       entropy-like difference ``u - g``.

   Returns
   -------
   values : numpy.ndarray
       Selected thermodynamic quantity divided by ``n_mol``.
   errors : numpy.ndarray
       Standard errors divided by ``n_mol``; enthalpy errors are set to zero.
   temperatures : numpy.ndarray
       Evaluation grid in kelvin.


``g_of_T.path_RES`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L765>`__

.. code-block:: python

   def path_RES(self)

.. rubric:: Docstring

.. code-block:: text

   str: Default prefix for the final temperature-dependent result.


``g_of_T.load_RES_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L770>`__

.. code-block:: python

   def load_RES_(self, name_RES)

.. rubric:: Docstring

.. code-block:: text

   Load a final result, accounting for representative subsampling.

   Parameters
   ----------
   name_RES : str
       Base result filename or pickle prefix.

   Returns
   -------
   dict
       Saved reference metadata and continuous/discrete curves.


``g_of_T.save_RES_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L787>`__

.. code-block:: python

   def save_RES_(self, name_RES)

.. rubric:: Docstring

.. code-block:: text

   Save ``RES`` with the appropriate subsampling filename marker.

   Parameters
   ----------
   name_RES : str
       Base result filename or pickle prefix.


``g_of_T.get_result_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L799>`__

.. code-block:: python

   def get_result_(self, Tmin=50, Tmax=800, Tstride=500, save=True)

.. rubric:: Docstring

.. code-block:: text

   Load or construct the complete temperature-dependent result bundle.

   Parameters
   ----------
   Tmin, Tmax : float, optional
       Bounds of the continuous temperature grid in kelvin.
   Tstride : int, optional
       Number of points in that grid.
   save : bool, optional
       Persist a newly constructed result.

   Notes
   -----
   ``RES`` contains reference free-energy metadata, smooth per-molecule
   Gibbs/enthalpy curves, and discrete values at sampled temperatures with
   convergence flags.


``g_of_T.cP_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L878>`__

.. code-block:: python

   def cP_(self, Tmin=50, Tmax=800, Tstride=500, include_KE=False)

.. rubric:: Docstring

.. code-block:: text

   Estimate constant-pressure heat capacity from a linear energy fit.

   Parameters
   ----------
   Tmin, Tmax : float, optional
       Bounds of the fine plotting grid in kelvin.
   Tstride : int, optional
       Number of points in the fine grid.
   include_KE : bool, optional
       Fit total energy when true; otherwise fit configurational enthalpy.

   Returns
   -------
   sampled : list
       Sampled temperatures and per-molecule energies in kJ/mol.
   fitted : list
       Fine temperature grid and fitted energies.
   cP : float
       Linear slope in kJ mol\ :sup:`-1` K\ :sup:`-1` molecule\ :sup:`-1`.


``LineFit`` (class)
^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L1875>`__

.. code-block:: python

   class LineFit(X, Y)

.. rubric:: Docstring

.. code-block:: text

   Multivariate least-squares line with explicit centring.


``LineFit.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L1878>`__

.. code-block:: python

   def __init__(self, X, Y)

.. rubric:: Docstring

.. code-block:: text

   Fit a linear map from rows of ``X`` to rows of ``Y``.

   Parameters
   ----------
   X : numpy.ndarray
       Input matrix shaped ``(samples, input_dimensions)``.
   Y : numpy.ndarray
       Target matrix shaped ``(samples, output_dimensions)``.


``LineFit.__call__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/Tx.py#L1897>`__

.. code-block:: python

   def __call__(self, X)

.. rubric:: Docstring

.. code-block:: text

   Predict centred linear responses for input rows ``X``.
