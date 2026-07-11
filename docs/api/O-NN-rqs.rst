.. _api-O-NN-rqs:

O.NN.rqs
========

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py>`__

.. rubric:: Docstring

.. code-block:: text

   TensorFlow rational-quadratic spline transformations.

   The module implements monotonic elementwise bijections used by the FECrys
   normalising flows. Transformations return both transformed values and
   elementwise log-absolute Jacobian determinants. The implementation follows
   Durkan et al., arXiv:1906.04032, and uses float64 internally near spline knots
   before returning float32 tensors.


Classes and functions
---------------------

``cast_64_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L38>`__

.. code-block:: python

   def cast_64_(x)

.. rubric:: Docstring

.. code-block:: text

   Convert ``x`` to a TensorFlow ``float64`` tensor.


``cast_32_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L41>`__

.. code-block:: python

   def cast_32_(x)

.. rubric:: Docstring

.. code-block:: text

   Convert ``x`` to a TensorFlow ``float32`` tensor.


``clip_by_value_preserve_gradient`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L45>`__

.. code-block:: python

   def clip_by_value_preserve_gradient(t, clip_value_min, clip_value_max, name=None)

.. rubric:: Docstring

.. code-block:: text

   Clip forward values while preserving the identity gradient.

   This matches ``tensorflow_probability.math.clip_by_value_preserve_gradient``:
   values are limited to the supplied interval, but differentiation with
   respect to ``t`` sees a derivative of one.


``get_grid_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L58>`__

.. code-block:: python

   def get_grid_(MLP_output, dim, n_bins, min_bin_w, interval, shape_parallel)

.. rubric:: Docstring

.. code-block:: text

   Convert unconstrained network outputs into ordered spline knots.

   Parameters
   ----------
   MLP_output : tensor
       Flattened bin logits with final size ``dim * n_bins``.
   dim, n_bins : int
       Number of transformed variables and bins per variable.
   min_bin_w : float
       Minimum bin width in normalised interval units. It must satisfy
       ``min_bin_w * n_bins <= 1``.
   interval : sequence of two float
       Lower and upper spline bounds.
   shape_parallel : list of int
       Additional batch-like dimensions between the leading batch and
       transformed-variable axes.

   Returns
   -------
   tensorflow.Tensor
       Ordered knot positions with shape ``(-1, *shape_parallel, dim,
       n_bins + 1)``.


``softplus_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L97>`__

.. code-block:: python

   def softplus_(x)

.. rubric:: Docstring

.. code-block:: text

   Apply the elementwise softplus function ``log(1 + exp(x))``.


``soft_cap_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L101>`__

.. code-block:: python

   def soft_cap_(x, a, s)

.. rubric:: Docstring

.. code-block:: text

   Continue softplus above ``a`` with a more slowly growing logarithm.

   ``s`` controls the slope/softness of the continuation. Inputs and
   parameters are evaluated in float64.


``softplus_with_a_softcap_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L113>`__

.. code-block:: python

   def softplus_with_a_softcap_(x, a=9.0, s=0.2)

.. rubric:: Docstring

.. code-block:: text

   Apply softplus up to ``a`` and its soft-capped continuation above it.


``normalize_knot_slopes_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L117>`__

.. code-block:: python

   def normalize_knot_slopes_(x, knot_slope_range)

.. rubric:: Docstring

.. code-block:: text

   Map unconstrained logits to positive, softly capped knot slopes.

   ``knot_slope_range`` supplies a hard positive offset and the location at
   which the upper soft cap begins. The upper value is not a strict maximum.


``rqs_bin_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L131>`__

.. code-block:: python

   def rqs_bin_(x, xA, xB, yA, yB, sA=1.0, sB=1.0, forward=True)

.. rubric:: Docstring

.. code-block:: text

   Evaluate or invert one rational-quadratic spline bin.

   ``xA``/``xB`` and ``yA``/``yB`` are the bin's input and output knots;
   ``sA`` and ``sB`` are positive boundary slopes. All arguments broadcast.
   With ``forward=False``, ``x`` denotes an output-space value. Returns the
   transformed value and log absolute derivative in the chosen direction.
   Small negative inverse discriminants caused by round-off are clipped to
   zero.


``rqs_`` (function)
^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L176>`__

.. code-block:: python

   def rqs_(x, w, h, s, interval=[-1.0, 1.0], identity_BCs=False, periodic_BCs=False, forward=True, min_bin_width=0.0001, knot_slope_range=[0.0001, 50.0], eps=1e-06)

.. rubric:: Docstring

.. code-block:: text

   Apply an elementwise monotonic rational-quadratic spline.

   Parameters
   ----------
   x : tensor, shape (..., dim)
       Values inside ``interval``.
   w, h : tensor, shape (..., dim * n_bins)
       Unconstrained logits for input-bin widths and output-bin heights.
   s : tensor, shape (..., dim * (n_bins + 1))
       Unconstrained knot-slope logits.
   interval : sequence of two float, default=(-1, 1)
       Common input and output domain.
   identity_BCs : bool, default=False
       Fix both outer slopes to one.
   periodic_BCs : bool, default=False
       Tie the two outer slopes when identity boundaries are disabled.
   forward : bool, default=True
       Evaluate input-to-output when true and the inverse otherwise.
   min_bin_width : float
       Smallest allowed bin width.
   knot_slope_range : sequence of two float
       Positive slope offset and soft-cap location.
   eps : float
       Boundary clipping margin. The implementation currently fixes this to
       ``1e-6`` internally for historical float32 compatibility.

   Returns
   -------
   y, log_abs_det_jacobian : tensorflow.Tensor
       Float32 tensors, each with shape ``(..., dim)``.


``shift_`` (function)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L316>`__

.. code-block:: python

   def shift_(x, shifts, interval=[-1.0, 1.0], forward=True)

.. rubric:: Docstring

.. code-block:: text

   Apply an invertible periodic translation within an interval.

   ``shifts`` may be scalar or broadcastable to ``x``. Values wrap into the
   half-open interval ``[a, b)``. Setting ``forward=False`` applies the
   inverse translation.


``periodic_rqs_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L331>`__

.. code-block:: python

   def periodic_rqs_(x, list_w, list_h, list_s, list_shifts, interval=[-1.0, 1.0], forward=True, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], eps=1e-06)

.. rubric:: Docstring

.. code-block:: text

   Compose shifted splines to form a periodic bijection.

   The leading axis of ``list_w``, ``list_h``, ``list_s``, and
   ``list_shifts`` enumerates transformations. Each stage shifts into a local
   periodic coordinate system, applies :func:`rqs_` with tied boundary
   slopes, then reverses the shift. Inverse evaluation traverses stages in
   reverse order. Returns transformed values and the sum of elementwise log
   Jacobians, both shaped like ``x``.


``test_periodic_rqs_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L383>`__

.. code-block:: python

   def test_periodic_rqs_(n_bins=8, n_transforms=2, min_bin_width=0.001, knot_slope_range=[0.001, 50.0])

.. rubric:: Docstring

.. code-block:: text

   Visually test scalar periodic-spline inversion and Jacobian cancellation.

   Random spline parameters are generated for 1,000 points. The function
   prints mean/max Jacobian cancellation error, displays diagnostic plots,
   and returns ``None``. It is an interactive diagnostic, not a unit test.


``test_periodic_rqs_parallel_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/rqs.py#L431>`__

.. code-block:: python

   def test_periodic_rqs_parallel_(n_bins=8, n_transforms=2, min_bin_width=0.001, knot_slope_range=[0.001, 50.0])

.. rubric:: Docstring

.. code-block:: text

   Run the periodic-spline diagnostic with an extra parallel axis.

   Two copies of a 1D grid are transformed together to exercise broadcasting.
   Inversion and log-Jacobian errors are printed and plotted; no value is
   returned.
