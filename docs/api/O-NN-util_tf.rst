.. _api-O-NN-util_tf:

O.NN.util_tf
============

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py>`__

.. rubric:: Docstring

.. code-block:: text

   TensorFlow numerical and coordinate-transformation utilities.

   This module supplies whitening, range scaling, focused bond/angle/torsion and
   rotation maps, molecular internal-coordinate transformations, quaternion
   geometry, and box representations. Transformation functions return values and
   log-absolute Jacobian determinants in the direction requested. Unless stated
   otherwise, tensors use float32 and Cartesian components occupy the final axis.


Classes and functions
---------------------

``np2tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L21>`__

.. code-block:: python

   def np2tf_(x)

.. rubric:: Docstring

.. code-block:: text

   Convert an array-like value to the configured TensorFlow float dtype.


``tf2np_`` (function)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L26>`__

.. code-block:: python

   def tf2np_(x)

.. rubric:: Docstring

.. code-block:: text

   Return a TensorFlow eager tensor as a NumPy array.


``clip_positive_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L37>`__

.. code-block:: python

   def clip_positive_(x)

.. rubric:: Docstring

.. code-block:: text

   Clip tensor values to the positive numerical-stability interval.


``norm_`` (function)
^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L42>`__

.. code-block:: python

   def norm_(x)

.. rubric:: Docstring

.. code-block:: text

   Euclidean norm along the final axis, retaining that axis.


``unit_`` (function)
^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L47>`__

.. code-block:: python

   def unit_(x)

.. rubric:: Docstring

.. code-block:: text

   Normalise vectors along the final axis without zero-norm protection.


``norm_clipped_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L52>`__

.. code-block:: python

   def norm_clipped_(x)

.. rubric:: Docstring

.. code-block:: text

   Final-axis Euclidean norm clipped below for numerical stability.


``unit_clipped_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L57>`__

.. code-block:: python

   def unit_clipped_(x)

.. rubric:: Docstring

.. code-block:: text

   Normalise final-axis vectors using a positive clipped norm.


``det_3x3_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L61>`__

.. code-block:: python

   def det_3x3_(M, keepdims=False)

.. rubric:: Docstring

.. code-block:: text

   Compute determinants of tensors shaped ``(..., 3, 3)``.


``make_COM_removal_matrix_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L66>`__

.. code-block:: python

   def make_COM_removal_matrix_(n_particles, dim=3)

.. rubric:: Docstring

.. code-block:: text

   Output:
   M : (n_particles*dim, n_particles*dim) matrix
       (n_particles - 1)*dim eigenvalues are ones, and dim eigenvalues are zeros.
       using the top (n_particles - 1)*dim eigenvectors removes COM from data 
       similarly to whitening, but is volume preserving.


``PCA_`` (function)
^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L88>`__

.. code-block:: python

   def PCA_(X0, removedims=3, diagonal=False, isotropic=False, not_whiten=False)

.. rubric:: Docstring

.. code-block:: text

   REF: https://github.com/noegroup/bgflow


``NotWhitenFlow`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L127>`__

.. code-block:: python

   class NotWhitenFlow(r_flat, removedims=3, whiten_anyway=False)

.. rubric:: Docstring

.. code-block:: text

   Remove global translation by fixing the first particle at the origin.

   The map reduces ``3*n_mol`` coordinates to ``3*(n_mol-1)``. It is volume
   preserving in this reduced representation; optional ``whiten_anyway`` then
   fits a conventional zero-dimension-removal whitening transform.


``NotWhitenFlow.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L135>`__

.. code-block:: python

   def __init__(self, r_flat, removedims=3, whiten_anyway=False)

.. rubric:: Docstring

.. code-block:: text

   Infer particle count and optionally fit a secondary whitening map.


``NotWhitenFlow.forward`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L151>`__

.. code-block:: python

   def forward(self, x)

.. rubric:: Docstring

.. code-block:: text

   Subtract the first particle, drop it, and return the reduced array.


``NotWhitenFlow.inverse`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L165>`__

.. code-block:: python

   def inverse(self, x)

.. rubric:: Docstring

.. code-block:: text

   Restore a zero first particle and expand to the original dimension.


``NotWhitenFlow._forward_np`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L178>`__

.. code-block:: python

   def _forward_np(self, x)

.. rubric:: Docstring

.. code-block:: text

   NumPy implementation of the translation-removing forward map.


``WhitenFlow`` (class)
^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L191>`__

.. code-block:: python

   class WhitenFlow(r, removedims=3, diagonal=False, isotropic=False)

.. rubric:: Docstring

.. code-block:: text

   REF: https://github.com/noegroup/bgflow


``WhitenFlow.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L193>`__

.. code-block:: python

   def __init__(self, r, removedims=3, diagonal=False, isotropic=False)

.. rubric:: Docstring

.. code-block:: text

   Fit PCA mean, whitening, and inverse matrices to ``r``.


``WhitenFlow._whiten`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L213>`__

.. code-block:: python

   def _whiten(self, x)

.. rubric:: Docstring

.. code-block:: text

   Apply mean removal/PCA whitening and return its constant log-Jacobian.


``WhitenFlow._blacken`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L219>`__

.. code-block:: python

   def _blacken(self, x)

.. rubric:: Docstring

.. code-block:: text

   Apply inverse whitening and return its constant log-Jacobian.


``WhitenFlow.forward`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L225>`__

.. code-block:: python

   def forward(self, x)

.. rubric:: Docstring

.. code-block:: text

   Whiten a tensor batch.


``WhitenFlow.inverse`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L230>`__

.. code-block:: python

   def inverse(self, x)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct a tensor batch from whitened coordinates.


``WhitenFlow._forward_np`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L235>`__

.. code-block:: python

   def _forward_np(self, x)

.. rubric:: Docstring

.. code-block:: text

   NumPy forward whitening without returning the Jacobian.


``pad_ranges_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L240>`__

.. code-block:: python

   def pad_ranges_(Min, Max, factor)

.. rubric:: Docstring

.. code-block:: text

   Return symmetric padding needed to enlarge a range by ``factor``.


``get_ranges_centres_MIN_MAX_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L246>`__

.. code-block:: python

   def get_ranges_centres_MIN_MAX_(x, axis: list, percentage_pad=0.0, range_limits: list=None, keepdims=False)

.. rubric:: Docstring

.. code-block:: text

   Estimate marginal ranges and centres from extrema.

   Optional limits are asserted to contain the observed data. Returns float32
   tensors whose reduced axes follow ``keepdims``.


``get_ranges_centres_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L307>`__

.. code-block:: python

   def get_ranges_centres_(x, axis: list, range_limits=None, keepdims=False)

.. rubric:: Docstring

.. code-block:: text

   Return data ranges and midpoints using the current min/max estimator.


``scale_shift_x_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L312>`__

.. code-block:: python

   def scale_shift_x_(x, physical_ranges_x, physical_centres_x, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Map each final-axis marginal between its physical range and [-1, 1].


``scale_shift_x_general_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L339>`__

.. code-block:: python

   def scale_shift_x_general_(x, physical_ranges_x, physical_centres_x, model_range=2.0, model_centre=0.0, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Map marginals between fitted physical ranges and a chosen model interval.


``scale_shift_individual_x_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L368>`__

.. code-block:: python

   def scale_shift_individual_x_(x, physical_ranges_x, physical_centres_x, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Vectorised per-entry scaling to/from [-1, 1] with batch Jacobians.


``FocusedBonds`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L409>`__

.. code-block:: python

   class FocusedBonds(X, axis=[0, 1], focused=True)

.. rubric:: Docstring

.. code-block:: text

   Affine map focusing observed bond lengths into [-1, 1].


``FocusedBonds.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L412>`__

.. code-block:: python

   def __init__(self, X, axis=[0, 1], focused=True)

.. rubric:: Docstring

.. code-block:: text

   Fit marginal ranges within the physical 0.05–0.22 nm limits.


``FocusedBonds.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L450>`__

.. code-block:: python

   def forward_(self, X)

.. rubric:: Docstring

.. code-block:: text

   Scale physical bond lengths to model coordinates.


``FocusedBonds.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L453>`__

.. code-block:: python

   def inverse_(self, X)

.. rubric:: Docstring

.. code-block:: text

   Restore physical bond lengths from model coordinates.


``FocusedBonds.__call__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L456>`__

.. code-block:: python

   def __call__(self, X, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Dispatch to the forward or inverse bond transform.


``FocusedAngles`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L461>`__

.. code-block:: python

   class FocusedAngles(X, axis=[0, 1], range_limits=[0.0, PI], focused=True)

.. rubric:: Docstring

.. code-block:: text

   Affine map focusing observed angles into [-1, 1].


``FocusedAngles.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L464>`__

.. code-block:: python

   def __init__(self, X, axis=[0, 1], range_limits=[0.0, PI], focused=True)

.. rubric:: Docstring

.. code-block:: text

   Fit marginal ranges inside the supplied angular limits in radians.


``FocusedAngles.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L502>`__

.. code-block:: python

   def forward_(self, X)

.. rubric:: Docstring

.. code-block:: text

   Scale physical angles to model coordinates.


``FocusedAngles.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L505>`__

.. code-block:: python

   def inverse_(self, X)

.. rubric:: Docstring

.. code-block:: text

   Restore physical angles from model coordinates.


``FocusedAngles.__call__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L508>`__

.. code-block:: python

   def __call__(self, X, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Dispatch to the forward or inverse angle transform.


``average_torsion_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L517>`__

.. code-block:: python

   def average_torsion_np_(x, axis=0, keepdims=True, pooling_method_=np.mean)

.. rubric:: Docstring

.. code-block:: text

   Compute a circular mean/median-like torsion centre in radians.


``centre_torsion_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L527>`__

.. code-block:: python

   def centre_torsion_tf_(x, x_mean, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Shift periodic angles by a centre while wrapping to [-pi, pi).


``FocusedTorsions`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L534>`__

.. code-block:: python

   class FocusedTorsions(X, axis=[0, 1], focused=True, verbose=True, mask_periodic=None)

.. rubric:: Docstring

.. code-block:: text

   Centre and scale torsions while preserving fully periodic marginals.


``FocusedTorsions.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L537>`__

.. code-block:: python

   def __init__(self, X, axis=[0, 1], focused=True, verbose=True, mask_periodic=None)

.. rubric:: Docstring

.. code-block:: text

   Fit circular centres/ranges and infer or accept periodic flags.


``FocusedTorsions.set_ranges_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L574>`__

.. code-block:: python

   def set_ranges_(self, mask_periodic)

.. rubric:: Docstring

.. code-block:: text

   Apply shared periodic flags and update ranges/secondary centres.


``FocusedTorsions.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L598>`__

.. code-block:: python

   def forward_(self, X)

.. rubric:: Docstring

.. code-block:: text

   Centre torsions and scale them to model coordinates.


``FocusedTorsions.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L604>`__

.. code-block:: python

   def inverse_(self, Z)

.. rubric:: Docstring

.. code-block:: text

   Undo torsion scaling and circular centring.


``FocusedTorsions.__call__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L610>`__

.. code-block:: python

   def __call__(self, X, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Dispatch to the forward or inverse torsion transform.


``merge_periodic_masks_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L615>`__

.. code-block:: python

   def merge_periodic_masks_(list_periodic_masks)

.. rubric:: Docstring

.. code-block:: text

   inputs arrays with integer elements 0 and 1 only


``Static_Rotations_Layer`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L749>`__

.. code-block:: python

   class Static_Rotations_Layer(q, indices: list=None)

.. rubric:: Docstring

.. code-block:: text

   Choose fixed quaternion rotations that avoid hemisphere singularities.


``Static_Rotations_Layer.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L752>`__

.. code-block:: python

   def __init__(self, q, indices: list=None)

.. rubric:: Docstring

.. code-block:: text

   Select grid rotations for each molecule or restore known indices.


``Static_Rotations_Layer.find_best_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L770>`__

.. code-block:: python

   def find_best_(self, q)

.. rubric:: Docstring

.. code-block:: text

   Grid-search rotations minimising hemisphere-coordinate pathologies.


``Static_Rotations_Layer.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L817>`__

.. code-block:: python

   def forward_(self, q)

.. rubric:: Docstring

.. code-block:: text

   Apply the selected per-molecule quaternion rotations.


``Static_Rotations_Layer.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L822>`__

.. code-block:: python

   def inverse_(self, q)

.. rubric:: Docstring

.. code-block:: text

   Apply transposed selected rotations to recover original quaternions.


``sample_phi_in_limits_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L827>`__

.. code-block:: python

   def sample_phi_in_limits_(m, E, F)

.. rubric:: Docstring

.. code-block:: text

   Sample ``m`` uniform azimuths independently between bounds ``E`` and ``F``.


``sample_theta1_in_limits_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L836>`__

.. code-block:: python

   def sample_theta1_in_limits_(m, C, D)

.. rubric:: Docstring

.. code-block:: text

   Sample polar angles with the correct sine surface-area density.


``sample_theta0_fastest_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L846>`__

.. code-block:: python

   def sample_theta0_fastest_(m, A, B, test=False)

.. rubric:: Docstring

.. code-block:: text

   Sample first hyperspherical angles with their ``sin(theta)^2`` density.

   A fixed rational-quadratic approximation inverts the analytic cumulative
   map. Returns samples shaped ``(m, n_mol)`` and an optional approximation
   error diagnostic.


``sample_theta0_fastest_.rqs_here_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L854>`__

.. code-block:: python

   def rqs_here_(x, w, h, s, interval=[0.0, 0.5 * PI], forward=False)

.. rubric:: Docstring

.. code-block:: text

   Evaluate the fixed spline used to invert the theta0 cumulative map.


``identity_shift`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L939>`__

.. code-block:: python

   class identity_shift()

.. rubric:: Docstring

.. code-block:: text

   Identity rotation strategy used when rotations are configured externally.


``identity_shift.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L942>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Create a stateless identity transform.


``identity_shift.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L945>`__

.. code-block:: python

   def forward_(self, x)

.. rubric:: Docstring

.. code-block:: text

   Return ``x`` unchanged.


``identity_shift.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L948>`__

.. code-block:: python

   def inverse_(self, x)

.. rubric:: Docstring

.. code-block:: text

   Return ``x`` unchanged.


``FocusedHemisphere`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L952>`__

.. code-block:: python

   class FocusedHemisphere(q, srl_indices_known=None, focused=True, static_rotations_defined_externally=False, mask_periodic_Phi=None)

.. rubric:: Docstring

.. code-block:: text

   Map quaternion rotations to a focused three-coordinate hemisphere patch.


``FocusedHemisphere.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L955>`__

.. code-block:: python

   def __init__(self, q, srl_indices_known=None, focused=True, static_rotations_defined_externally=False, mask_periodic_Phi=None)

.. rubric:: Docstring

.. code-block:: text

   Fit static rotations and focused hyperspherical marginal ranges.


``FocusedHemisphere.set_ranges_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L999>`__

.. code-block:: python

   def set_ranges_(self, mask_periodic_Phi)

.. rubric:: Docstring

.. code-block:: text

   Update periodic azimuth flags and compute patch bounds/area.


``FocusedHemisphere.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1046>`__

.. code-block:: python

   def forward_(self, q)

.. rubric:: Docstring

.. code-block:: text

   Rotate and map quaternions into three scaled patch coordinates.


``FocusedHemisphere.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1057>`__

.. code-block:: python

   def inverse_(self, s_scaled)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct quaternions from scaled hemisphere coordinates.


``FocusedHemisphere.__call__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1068>`__

.. code-block:: python

   def __call__(self, X, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Dispatch to the forward or inverse quaternion-patch map.


``FocusedHemisphere.sample_quaternion_patch_v1_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1073>`__

.. code-block:: python

   def sample_quaternion_patch_v1_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Rejection-sample ``m`` uniform quaternions inside every fitted patch.


``FocusedHemisphere.sample_quaternion_patch_v2_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1094>`__

.. code-block:: python

   def sample_quaternion_patch_v2_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Directly sample ``m`` uniform quaternions from separable patch marginals.


``quat2axisangle_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1113>`__

.. code-block:: python

   def quat2axisangle_(q)

.. rubric:: Docstring

.. code-block:: text

   can be used for visualising rotational distributions in 3D


``get_coupling_masks_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1124>`__

.. code-block:: python

   def get_coupling_masks_(dim_flow: int)

.. rubric:: Docstring

.. code-block:: text

   REF: TABLE 1 in arXiv:2001.05486v2 (i-flow)
   Input:
       dim_flow : number of DOFs being transformed using coupling flow
   Output:
       cond_masks : (n_layers, dim_flow) array of zeros and ones
           'conditionaling masks' or 'coupling masks'
   Usage:
       In a NF model that is based on coupling:
           each coupling layer splits the total number (dim_flow) of 1D margianl DOFs into two sets A and B
           setting one coupling layer for each row of cond_masks allows all DOFs to be coupled
               in practice it is not necesary to use all of the rows (just first 4 were generally used; n_layers = 4)


``reshape_to_molecules_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1149>`__

.. code-block:: python

   def reshape_to_molecules_tf_(r, n_molecules, n_atoms_in_molecule)

.. rubric:: Docstring

.. code-block:: text

   Reshape batched coordinates to ``(batch, molecules, atoms, 3)``.


``reshape_to_atoms_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1153>`__

.. code-block:: python

   def reshape_to_atoms_tf_(r, n_molecules, n_atoms_in_molecule)

.. rubric:: Docstring

.. code-block:: text

   Reshape batched coordinates to ``(batch, total_atoms, 3)``.


``reshape_to_flat_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1157>`__

.. code-block:: python

   def reshape_to_flat_tf_(r, n_molecules, n_atoms_in_molecule)

.. rubric:: Docstring

.. code-block:: text

   Flatten all molecular Cartesian coordinates within each batch item.


``get_distance_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1164>`__

.. code-block:: python

   def get_distance_tf_(R, inds_2_atoms)

.. rubric:: Docstring

.. code-block:: text

   bond distance


``get_angle_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1174>`__

.. code-block:: python

   def get_angle_tf_(R, inds_3_atoms)

.. rubric:: Docstring

.. code-block:: text

   bond angle


``get_torsion_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1195>`__

.. code-block:: python

   def get_torsion_tf_(R, inds_4_atoms)

.. rubric:: Docstring

.. code-block:: text

   REF: https://github.com/noegroup/bgflow


``r_to_x_atom_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1231>`__

.. code-block:: python

   def r_to_x_atom_(R, row_ABCD_IC)

.. rubric:: Docstring

.. code-block:: text

   transforms Cartesian coordiante of one atom into internal coordinate (IC) [i.e., a Z-matrix coordinate]
       r_to_x_atom_ : r_{A} -> x_{A}
   Inputs:
       R            : (..., n_atoms_mol, 3) ; n_atoms_mol = number of atoms in a single molecule
       row_ABCD_IC  : (4,) ; four indices of atoms in the molecule [A,B,C,D]
           the indices corespond to atoms that are covalently bonded A-B-C-D
   Output:
       x_IC         : (..., 3) ; x_IC = x_{A}


``IC_ladJ_inv_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1247>`__

.. code-block:: python

   def IC_ladJ_inv_(X_IC)

.. rubric:: Docstring

.. code-block:: text

   log volume change of the internal coordinate representation
   Input: 
       X_IC : (..., n_atoms, 3) where the last axis must be [bond, angle, torsion]
   Output:
       ladJ_inv : (..., 1) log volume change in the inverse direction (x -> r)
           = - log(det(Jacobian)) of the forward Jacobian = dx(r)/dr ; x(r) = r_to_x_atom_
   this is d*S2 coordinate ; d > 0
       size the sphere (bond distance) and the longitude (bond angle) contribute to local volume


``IC_forward_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1263>`__

.. code-block:: python

   def IC_forward_(r, ABCD_IC)

.. rubric:: Docstring

.. code-block:: text

   Inputs:
       r       : (m, n_mol, n_atoms_mol, 3) Cartesian coordiantes of the (single component) system
       ABCD_IC : (n_atoms_IC, 4) matrix of indices for all bonded atoms that are represented using ICs
   Ouputs:
       X_IC    : (m, n_mol, n_atoms_IC, 3) internal coordinates of the (single component) system
       ladJ    : (m, n_mol, 1) log volume change of the IC representation. Sum over n_mol later.


``NeRF_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1279>`__

.. code-block:: python

   def NeRF_tf_(d, theta, phi, rB, rC, rD, no_ladJ=True)

.. rubric:: Docstring

.. code-block:: text

   REF: DOI 10.1002/jcc.20237

   xA -> rA (inverse of r_to_x_atom_) ; xA = [d, theta, phi]

   Inputs:
       internal coordinate of atom A (relative to atoms B, C and D):
           d     : (..., ) ; A-B bond distance
           theta : (..., ) ; ABC bond angle
           phi   : (..., ) ; AB-CD torsional angle
       constants:
           rB    : (..., 3) Cartesian coordinate of atom B (bonded to A)
           rC    : (..., 3) Cartesian coordinate of atom C (bonded to B)
           rD    : (..., 3) Cartesian coordinate of atom D (bonded to C)
       no_ladJ   : bool ; True because IC_ladJ_inv_ is a cheaper way to get the same number.

   Outputs:
       rA        : (..., 3) Cartesian coordinate of atom A


``IC_inverse_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1326>`__

.. code-block:: python

   def IC_inverse_(X_IC, r_CB, ABCD_IC_inverse: list, inds_unpermute_atoms)

.. rubric:: Docstring

.. code-block:: text

   reconstruct Cartesian coordinates of a molecule from internal coordinates
       Cartesian coordinates of the first 3 atoms are specified already (r_CB)
       All other atoms of the molecule reconstructed one-by-one using NeRF_tf_
   Inputs:
       X_IC : (..., n_atoms_IC, 3) ; internal coordinates of atoms that are 
       r_CB : (..., 3, 3) ; the Cartesian coordinates of the 'Cartesian block' (CB)
       ABCD_IC_inverse : (n_atoms_IC, 4) ; n_atoms_IC == n_atoms_mol - 3
           same as ABCD_IC used for the forward transfromation but the order of the rows is now different
           the order of the rows is set such that NeRF_tf_ is 'walking' along the molecule, (i.e.,
           any new atom (rA) being reconstructed has [rB, rC, rD] already reconstructed earlier).
       inds_unpermute_atoms : (n_atoms_mol,) indices to permute the atoms in the molecule back to original order
   Outputs:
       R    : (..., n_atoms_mol, 3) ; Cartesian coordinates of the entire system
       ladJ : (..., 1) ; log volume change of the current reconstruction (sum over molecules later)


``cond_0_true_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1367>`__

.. code-block:: python

   def cond_0_true_(R)

.. rubric:: Docstring

.. code-block:: text

   for mat_to_quat_tf_


``cond_1_true_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1377>`__

.. code-block:: python

   def cond_1_true_(R)

.. rubric:: Docstring

.. code-block:: text

   for mat_to_quat_tf_


``cond_2_true_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1386>`__

.. code-block:: python

   def cond_2_true_(R)

.. rubric:: Docstring

.. code-block:: text

   for mat_to_quat_tf_


``all_conds_false_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1395>`__

.. code-block:: python

   def all_conds_false_(R)

.. rubric:: Docstring

.. code-block:: text

   for mat_to_quat_tf_


``mat_to_quat_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1404>`__

.. code-block:: python

   def mat_to_quat_tf_(R)

.. rubric:: Docstring

.. code-block:: text

   REF: arXiv:2301.11355 (rigid body flows)

   transformation: rotation matrix (R) -> quaternion (q)


``CB_ladJ_inv_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1446>`__

.. code-block:: python

   def CB_ladJ_inv_(a, d0, d1)

.. rubric:: Docstring

.. code-block:: text

   *REF: arXiv:2301.11355 (rigid body flows) 
   log volume change of transforming between 'Cartesian block' (CB) and the representation of CB from *REF

   Cartesian block (e.g,. a water molecule H0 - O - H1) is : r_{CB} = [r_{O}, r_{H0}, r_{H1}]
   The representation is: x_{CB} = [r_{O}, a, d0, d1, q]

   Inputs:
       a  : angle between bonds H0 - O and H1 - O
       d0 : bond distance between H0 and O
       d1 : bond distance between H1 and O
   Output:
       ladJ_inv : log volume change of transformation x_{CB} -> r_{CB}


``CB_forward_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1468>`__

.. code-block:: python

   def CB_forward_(xyz_CB)

.. rubric:: Docstring

.. code-block:: text

   REF: arXiv:2301.11355 (rigid body flows)

   Input:
       xyz_CB : (..., 3, 3) Cartesian coordinates of the 'Cartesian block' (CB) = r_{CB} = [rA, rB, rC]

   Outputs:
       X       : list of 5 arrays [rA, q, a, dAB, dAC] = x_{CB}
           rA  : (..., 3) ; Cartesian coordinate of atom A unchanged
           q   : (..., 4) ; unit-quaternion describing the rotation of the CB
           a   : (..., 1) ; angle between bonds rB - rA and rC - rA
           dAB : (..., 1) ; bond distance between rB and rA
           dAC : (..., 1) ; bond distance between rC and rA
       ladJ    : (..., 1) ; log volume change of the transformation r_{CB} -> x_{CB}


``quat_to_mat_tf_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1519>`__

.. code-block:: python

   def quat_to_mat_tf_(q)

.. rubric:: Docstring

.. code-block:: text

   REF: INTRODUCTION TO ROBOTICS MECHANICS, PLANNING, AND CONTROL F. C. Park and K. M. Lynch

   transformation: quaternion (q) -> rotation matrix (R)


``CB_inverse_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1532>`__

.. code-block:: python

   def CB_inverse_(X)

.. rubric:: Docstring

.. code-block:: text

   REF: arXiv:2301.11355 (rigid body flows)

   inverse of CB_forward_

   Inputs:
       X      : list of 5 arrays [rA, q, a, dAB, dAC] = x_{CB}
           explained in CB_forward_
   Outputs:
       xyz_CB : (..., 3, 3) Cartesian coordinates of the 'Cartesian block' (CB) = r_{CB}
       ladJ   : (..., 1) ; log volume change of the transformation x_{CB} -> r_{CB}


``test_CB_transformation_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1561>`__

.. code-block:: python

   def test_CB_transformation_(n_frames=1000, n_molecules=40)

.. rubric:: Docstring

.. code-block:: text

   comparing the analytical expression of log volume change in CB_ladJ_inv_ to a numerical analogue (using full jacobian)
   comparison on random Cartesian blocks
   comparisons show that the closed-form expression is correct and very efficient


``CB_single_molecule_forward_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1597>`__

.. code-block:: python

   def CB_single_molecule_forward_(xyz_CB)

.. rubric:: Docstring

.. code-block:: text

   for single molecule in vaccum
   TODO: was not tested properly yet, but seems to work


``CB_single_molecule_inverse_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1637>`__

.. code-block:: python

   def CB_single_molecule_inverse_(X)

.. rubric:: Docstring

.. code-block:: text

   Reconstruct a canonical isolated three-atom Cartesian block.

   ``X`` contains angle and two bond distances. Translation is fixed at zero
   and rotation at the identity because isolated-molecule energy is invariant
   to both. Returns coordinates and the inverse polar-coordinate Jacobian.


``hemisphere_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1659>`__

.. code-block:: python

   def hemisphere_(x)

.. rubric:: Docstring

.. code-block:: text

   Choose the quaternion representative whose scalar component is nonnegative.


``hemisphere_forward_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1663>`__

.. code-block:: python

   def hemisphere_forward_(q, rescale_marginals=True)

.. rubric:: Docstring

.. code-block:: text

   REF: https://doi.org/10.1021/acs.jctc.4c01612


``hemisphere_inverse_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1711>`__

.. code-block:: python

   def hemisphere_inverse_(xq, rescale_marginals=True)

.. rubric:: Docstring

.. code-block:: text

   REF: https://doi.org/10.1021/acs.jctc.4c01612


``sample_q_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1749>`__

.. code-block:: python

   def sample_q_(shape: list)

.. rubric:: Docstring

.. code-block:: text

   samples random 4D unit-vector


``quat_metrix_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1767>`__

.. code-block:: python

   def quat_metrix_(q, inverse=False)

.. rubric:: Docstring

.. code-block:: text

   for quaternion product done as a matrix multiplication (less efficent)
   Inputs:
       q       : (..., 4)    ; unitvector in R^4
   Ouput:
       R4      : (..., 4, 4) ; rotation matrix in R^4
       inverse : bool        ; if True, R4.transpose() = inv(R4) returned
   Usage: in FocusedHemisphere
       Used in Static_Rotations_Layer to rotate MD data away from problematic regions 
       of the hyperspherical representation of 'Cartesian block' rotations (s = xq).


``quat_product_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1790>`__

.. code-block:: python

   def quat_product_(q, p)

.. rubric:: Docstring

.. code-block:: text

   closed-from quaternion product (more efficent)


``quat_inverse_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1799>`__

.. code-block:: python

   def quat_inverse_(q)

.. rubric:: Docstring

.. code-block:: text

   Return the conjugate/inverse of a unit quaternion.


``box_forward_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1808>`__

.. code-block:: python

   def box_forward_(b)

.. rubric:: Docstring

.. code-block:: text

   placeholder


``box_inverse_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/util_tf.py#L1814>`__

.. code-block:: python

   def box_inverse_(h)

.. rubric:: Docstring

.. code-block:: text

   placeholder
