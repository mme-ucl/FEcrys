.. _api-O-NN-spline_layer:

O.NN.spline_layer
=================

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py>`__

.. rubric:: Docstring

.. code-block:: text

   Keras layers for spline-based normalising-flow couplings.

   Conditioner networks generate rational-quadratic spline parameters for
   periodic and ordinary variables. Half layers transform one mask partition;
   full coupling layers compose complementary half layers so every non-auxiliary
   variable can be transformed. Forward and inverse methods return the transformed
   tensor and a per-sample summed log-absolute Jacobian determinant.


Classes and functions
---------------------

``MLP`` (class)
^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L18>`__

.. code-block:: python

   class MLP(dims_outputs: list, dims_hidden: list, hidden_activation=tf.nn.silu, outputs_activations: list=None, output_kernel_initializer='glorot_uniform', output_bias_initializer='zeros', **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Dense multilayer perceptron with one or more output heads.

   The layer always returns a list—one tensor for every requested output
   dimension—even when there is only one output head.


``MLP.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L25>`__

.. code-block:: python

   def __init__(self, dims_outputs: list, dims_hidden: list, hidden_activation=tf.nn.silu, outputs_activations: list=None, output_kernel_initializer='glorot_uniform', output_bias_initializer='zeros', **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Construct shared hidden layers and independent dense output heads.

   ``dims_hidden`` and ``dims_outputs`` specify layer widths. Output
   activations and kernel/bias initializers may be single values applied
   to every head or one value per head.


``MLP.call`` (method)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L67>`__

.. code-block:: python

   def call(self, x)

.. rubric:: Docstring

.. code-block:: text

   Input:   x : (..., d) ; d = dimensionality of input
   Outputs: ys : list of outputs with shapes (m, dims_outputs[i]) for every output layer i.


``get_pos_encoding_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L83>`__

.. code-block:: python

   def get_pos_encoding_(pos, C=6, dim_embedding=3)

.. rubric:: Docstring

.. code-block:: text

   Return sinusoidal position features for one molecule index.

   The output contains ``dim_embedding`` sine values followed by the same
   number of cosine values, with frequencies scaled geometrically by ``C``.


``AT_MLP`` (class)
^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L94>`__

.. code-block:: python

   class AT_MLP(n_mol: int, n_heads: int, embedding_dim: int, output_dim: int, n_hidden_kqv: list=[1, 1, 1], hidden_activation=tf.nn.silu, one_hot_kqv=[True] * 3, name='AT_MLP', mask_self=True)

.. rubric:: Docstring

.. code-block:: text

   self-attention block


``AT_MLP.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L98>`__

.. code-block:: python

   def __init__(self, n_mol: int, n_heads: int, embedding_dim: int, output_dim: int, n_hidden_kqv: list=[1, 1, 1], hidden_activation=tf.nn.silu, one_hot_kqv=[True] * 3, name='AT_MLP', mask_self=True)

.. rubric:: Docstring

.. code-block:: text

   Construct multi-head key, query, and value conditioner networks.

   Inputs are expected as ``(batch, n_mol, input_dim)``. ``one_hot_kqv``
   controls addition of sinusoidal molecule-index features to each of the
   key/query/value networks. ``mask_self`` removes diagonal attention for
   multi-molecule systems.


``AT_MLP.one_hot_extend_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L174>`__

.. code-block:: python

   def one_hot_extend_(self, x)

.. rubric:: Docstring

.. code-block:: text

   Append the fixed position encoding to each molecule's features.


``AT_MLP.call`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L178>`__

.. code-block:: python

   def call(self, x, flow_mask=None)

.. rubric:: Docstring

.. code-block:: text

   Apply positive normalised attention across molecules.

   Parameters
   ----------
   x : tensorflow.Tensor, shape (batch, n_mol, input_dim)
       Per-molecule input features.
   flow_mask : tensor, shape (n_mol,), optional
       Multiplicative mask for participating molecules.

   Returns
   -------
   tensorflow.Tensor, shape (batch, n_mol, output_dim)
       Multi-head values summed over source molecules and heads.


``SPLINE_COUPLING_helper`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L219>`__

.. code-block:: python

   class SPLINE_COUPLING_helper()

.. rubric:: Docstring

.. code-block:: text

   Shared masking and transformation logic for spline half layers.

   Subclasses call :meth:`init_` and provide ``MLP_``, a conditioner mapping
   the conditioning variables to periodic and ordinary spline parameters.


``SPLINE_COUPLING_helper.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L226>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Create an unconfigured helper; subclasses must call :meth:`init_`.


``SPLINE_COUPLING_helper.init_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L229>`__

.. code-block:: python

   def init_(self, periodic_mask: list, cond_mask: list, flow_mask: tf.constant=None, use_tfp: bool=False, n_bins: int=4, min_bin_width: float=0.001, knot_slope_range: list=[0.001, 50.0], nk_for_periodic_encoding: int=1)

.. rubric:: Docstring

.. code-block:: text

   Inputs:
   periodic_mask  : (n_variables,)
       1 if variable is periodic,
       0 otherwise.
   cond_mask      : (n_variables,) ; conditioning mask
       1 if variable should be transformed, 
       0 if variable should help condition transformations of the former.
   flow_mask      : (1, n_mol, n_variables) ; default is None (no masking)
       1.0 if variable should be transformed,
       0.0 if variable should be kept constant.
           This is mostly useful if the variables that should not be transformed
           are different depending on the axis that is between the last axis, 
           and the batch axis (i.e., first axis) of the input.
               NB: the computations of the splines are still done on the 0.0 variables, but
               these variables are returned back to original, with 0.0 set for log volume change.
               This is why this mask is flaot, as it is actually multipled to the variables.
   use_tfp        : whether to the use tensorflow_probability version of rational quadratic spline (both are in rqs.py)
   n_bins         : number of bins for each spline (a 'bins' is an area between two neighboring knots)
   min_bin_width  : mimimum vertical/horizontal distance between any pair of neighboring knots
   knot_slope_range : minimum and maximum slopes allowed at any knot
   nk_for_periodic_encoding : number of orthogonal sines and cosines for encoding each periodic marginal variable
       Such encoding used before forwarding any periodic variable into a DNN (DNN can only deal with Euclidean variables)


``SPLINE_COUPLING_helper.init_.cos_sin_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L360>`__

.. code-block:: python

   def cos_sin_(x, nk: int=1)

.. rubric:: Docstring

.. code-block:: text

   Encode scaled periodic values with ``nk`` Fourier harmonics.


``SPLINE_COUPLING_helper.init_.cos_sin_1_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L369>`__

.. code-block:: python

   def cos_sin_1_(x)

.. rubric:: Docstring

.. code-block:: text

   Encode scaled periodic values with one cosine/sine pair.


``SPLINE_COUPLING_helper._identity_output_bias_initializer_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L379>`__

.. code-block:: python

   def _identity_output_bias_initializer_(self)

.. rubric:: Docstring

.. code-block:: text

   Compute output-layer bias initializer(s) so that, when combined with a zero-valued
   output kernel, the spline parameters produce the identity transformation.

   For a rational quadratic spline to be the identity:
     - width (w) and height (h) parameters equal zero -> softmax gives uniform bins ->
       the x-grid and y-grid are identical, so every knot lies on the diagonal y = x.
     - slope (s) parameters must produce a slope of 1.0 everywhere:
           normalize_knot_slopes_(s_init) == 1.0
           => s_init = log( exp(1 - min_slope) - 1 )
     - shift parameters for periodic splines equal zero -> no horizontal shift.

   Returns a single initializer when dims_MLP_outputs contains one non-zero entry,
   or a list of two initializers [bias_P, bias_O] when both entries are non-zero.


``SPLINE_COUPLING_helper._get_output_initializers_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L423>`__

.. code-block:: python

   def _get_output_initializers_(self, identity_init)

.. rubric:: Docstring

.. code-block:: text

   Return (output_kernel_initializer, output_bias_initializer) for the output MLP layer(s).

   When identity_init is True, the kernel is zeroed and the bias is set so that the
   untrained spline produces the identity transformation.
   When identity_init is False, the standard Glorot-uniform kernel and zero bias are used.


``SPLINE_COUPLING_helper.transform_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L434>`__

.. code-block:: python

   def transform_(self, x, forward: bool, dont_mask=False)

.. rubric:: Docstring

.. code-block:: text

   Inputs:
       x         : (m, n_mol, n_variables) input variables
                     ; where n_mol >= 1 (i.e., cannot have a missing axis for molecules)
       forward   : if True transfromation is in the forward direction, False for inverse
       dont_mask : if True, self.flow_mask is ignored as if it was None (default)
   Outputs:
       y         : (m, n_mol, n_variables) transformed variables
       ladJ      : (m, 1) elementwise sum_{i,j} log(dy[...,i,j]/dx[...,i,j])


``SPLINE_COUPLING_helper.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L551>`__

.. code-block:: python

   def forward_(self, x, dont_mask=False)

.. rubric:: Docstring

.. code-block:: text

   Transform ``x`` forward and return values plus summed log-Jacobian.


``SPLINE_COUPLING_helper.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L555>`__

.. code-block:: python

   def inverse_(self, x, dont_mask=False)

.. rubric:: Docstring

.. code-block:: text

   Invert the half-layer and return values plus inverse log-Jacobian.


``SPLINE_COUPLING_HALF_LAYER`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L559>`__

.. code-block:: python

   class SPLINE_COUPLING_HALF_LAYER(periodic_mask, cond_mask, use_tfp=False, n_bins=4, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], dims_hidden=None, n_hidden=2, hidden_activation=tf.nn.silu, identity_init=False, name=None)

.. rubric:: Docstring

.. code-block:: text

   SPLINE_COUPLING_helper with standard DNN used for parametrising the splines


``SPLINE_COUPLING_HALF_LAYER.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L575>`__

.. code-block:: python

   def __init__(self, periodic_mask, cond_mask, use_tfp=False, n_bins=4, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], dims_hidden=None, n_hidden=2, hidden_activation=tf.nn.silu, identity_init=False, name=None)

.. rubric:: Docstring

.. code-block:: text

   Build a half coupling layer with a standard dense conditioner.

   ``periodic_mask`` identifies circular variables and ``cond_mask`` uses
   one for transformed variables and zero for conditioner inputs.
   ``identity_init`` zeroes the final kernels and chooses biases that make
   the initial splines identities. Hidden widths default to the total
   spline-parameter output size.


``SPLINE_COUPLING_HALF_LAYER_AT`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L656>`__

.. code-block:: python

   class SPLINE_COUPLING_HALF_LAYER_AT(periodic_mask, cond_mask, use_tfp=False, n_bins=4, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], flow_mask=None, n_mol=1, n_heads=2, embedding_dim=20, n_hidden_kqv=[1, 1, 1], hidden_activation=tf.nn.silu, one_hot_kqv=[False] * 3, n_hidden_decode=2, add_residual=False, identity_init=False, name=None)

.. rubric:: Docstring

.. code-block:: text

   SPLINE_COUPLING_helper with AT_MLP involved for parametrising the splines


``SPLINE_COUPLING_HALF_LAYER_AT.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L664>`__

.. code-block:: python

   def __init__(self, periodic_mask, cond_mask, use_tfp=False, n_bins=4, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], flow_mask=None, n_mol=1, n_heads=2, embedding_dim=20, n_hidden_kqv=[1, 1, 1], hidden_activation=tf.nn.silu, one_hot_kqv=[False] * 3, n_hidden_decode=2, add_residual=False, identity_init=False, name=None)

.. rubric:: Docstring

.. code-block:: text

   Build a spline half layer with inter-molecular self-attention.

   ``flow_mask`` can freeze selected molecule/variable entries. Attention
   is used only when ``n_mol > 1``; otherwise the decoder receives the
   conditioning variables directly. ``add_residual`` chooses addition or
   concatenation of attention features before decoding spline parameters.


``SPLINE_COUPLING_HALF_LAYER_AT.set_MLPs_default_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L725>`__

.. code-block:: python

   def set_MLPs_default_(self)

.. rubric:: Docstring

.. code-block:: text

   Anything that works is allowed in this part of the model.
   Finding a combinatoon that works better is always useful.

   The settings that were used so far, are:

       y_flowing = splines(x_flowing, params = MLP( \psi_conditioning )

       Since the output dimensionality of the MLP is very high (spline params are many),
       if does not matter too much if:
           \psi_conditioning = [AT_MLP(x_conditioning), x_conditioning] # add_residual False
       OR; 
           \psi_conditioning = AT_MLP(x_conditioning) + x_conditioning  # add_residual True


``SPLINE_COUPLING_LAYER`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L788>`__

.. code-block:: python

   class SPLINE_COUPLING_LAYER(periodic_mask, cond_mask, use_tfp=False, n_bins=4, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], name=None, half_layer_class=SPLINE_COUPLING_HALF_LAYER, kwargs_for_given_half_layer_class={'n_hidden': 2, 'dims_hidden': None, 'hidden_activation': tf.nn.silu})

.. rubric:: Docstring

.. code-block:: text

   General coupling layer wrapper, where can choose half layer type (half_layer_class arg).

   There are currenty two options for the half_layer_class:
       - SPLINE_COUPLING_HALF_LAYER
       - SPLINE_COUPLING_HALF_LAYER_AT

   kwargs_for_given_half_layer_class:
       dictionary of settings that are relevant to the chosen half_layer_class

   Template examples for choosing half_layer_class and kwargs_for_given_half_layer_class:

   """ standard coupling layer:
   half_layer_class = SPLINE_COUPLING_HALF_LAYER,
   kwargs_for_given_half_layer_class = {
                                       'n_hidden' : 2,
                                       'dims_hidden':None,
                                       'hidden_activation':tf.nn.silu,
                                       }
   """

   """ coupling layer involving self-attention:
   half_layer_class = SPLINE_COUPLING_HALF_LAYER_AT,
   kwargs_for_given_half_layer_class = {
                                       'flow_mask' : None,
                                       'n_mol' : 1,
                                       'n_heads' : 2,
                                       'embedding_dim' : 20,
                                       'n_hidden_kqv' : [1,1,1],
                                       'hidden_activation' : tf.nn.silu,
                                       'one_hot_kqv' : [False]*3,
                                       'n_hidden_decode' : 2,
                                       }
   """


``SPLINE_COUPLING_LAYER.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L825>`__

.. code-block:: python

   def __init__(self, periodic_mask, cond_mask, use_tfp=False, n_bins=4, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], name=None, half_layer_class=SPLINE_COUPLING_HALF_LAYER, kwargs_for_given_half_layer_class={'n_hidden': 2, 'dims_hidden': None, 'hidden_activation': tf.nn.silu})

.. rubric:: Docstring

.. code-block:: text

   Compose complementary spline half layers into one bijection.

   ``cond_mask`` values mean: 1 transforms in layer A, 0 transforms in
   layer B, and 2 remains auxiliary/conditioning-only in both. The chosen
   ``half_layer_class`` receives shared spline settings plus
   ``kwargs_for_given_half_layer_class``.


``SPLINE_COUPLING_LAYER.forward`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L894>`__

.. code-block:: python

   def forward(self, x, dont_mask=False)

.. rubric:: Docstring

.. code-block:: text

   Apply half layers A then B and sum their log-Jacobians.


``SPLINE_COUPLING_LAYER.inverse`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/spline_layer.py#L901>`__

.. code-block:: python

   def inverse(self, x, dont_mask=False)

.. rubric:: Docstring

.. code-block:: text

   Invert half layers in reverse order and sum inverse log-Jacobians.
