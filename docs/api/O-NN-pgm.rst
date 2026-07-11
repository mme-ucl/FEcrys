.. _api-O-NN-pgm:

O.NN.pgm
========

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py>`__

.. rubric:: Docstring

.. code-block:: text

   Probabilistic generative models for molecular crystals and molecules.

   The active models compose a physical coordinate representation with alternating
   rational-quadratic spline couplings. Position-to-conformation and
   conformation-to-position connector networks exchange conditioning information.
   Multiple thermodynamic or metastable states can share one model through a
   scalar state encoding selected by ``crystal_index``.


Classes and functions
---------------------

``POSITIONS_FLOW_LAYER`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L17>`__

.. code-block:: python

   class POSITIONS_FLOW_LAYER(n_mol, layer_index, DIM_P2C_connection: int, DIM_C2P_connection: int, name='POSITIONS_FLOW_LAYER', n_hidden_main=2, n_hidden_connection=1, hidden_activation=tf.nn.leaky_relu, identity_init=False, use_tfp=False, n_bins=5, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], n_P2C=None)

.. rubric:: Docstring

.. code-block:: text

   PGMcrys_v1 only


``POSITIONS_FLOW_LAYER.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L21>`__

.. code-block:: python

   def __init__(self, n_mol, layer_index, DIM_P2C_connection: int, DIM_C2P_connection: int, name='POSITIONS_FLOW_LAYER', n_hidden_main=2, n_hidden_connection=1, hidden_activation=tf.nn.leaky_relu, identity_init=False, use_tfp=False, n_bins=5, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], n_P2C=None)

.. rubric:: Docstring

.. code-block:: text

   Construct one positional spline layer and its P-to-C connector.

   The position flow has ``3*(n_mol-1)`` translation-invariant variables.
   ``aux`` features condition its spline without themselves being
   transformed. ``layer_index`` selects a coupling mask.


``POSITIONS_FLOW_LAYER.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L89>`__

.. code-block:: python

   def forward_(self, input, aux)

.. rubric:: Docstring

.. code-block:: text

   Transform positional variables conditioned on auxiliary features.


``POSITIONS_FLOW_LAYER.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L99>`__

.. code-block:: python

   def inverse_(self, input, aux)

.. rubric:: Docstring

.. code-block:: text

   Invert positional variables with the same auxiliary features.


``POSITIONS_FLOW_LAYER.convert_to_aux_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L109>`__

.. code-block:: python

   def convert_to_aux_(self, input)

.. rubric:: Docstring

.. code-block:: text

   Encode flat positional variables into per-conformer auxiliary vectors.


``POSITIONS_FLOW_LAYER.convert_to_flow_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L119>`__

.. code-block:: python

   def convert_to_flow_(self, pos)

.. rubric:: Docstring

.. code-block:: text

   Insert the singleton molecule-like axis expected by the position flow.


``POSITIONS_FLOW_LAYER.convert_from_flow_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L125>`__

.. code-block:: python

   def convert_from_flow_(self, pos)

.. rubric:: Docstring

.. code-block:: text

   Remove the singleton axis from positional flow variables.


``C2P_connector_v1`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L131>`__

.. code-block:: python

   class C2P_connector_v1(**kwargs)

.. rubric:: Docstring

.. code-block:: text

   Encode conformational variables into one flat C-to-P feature vector.


``C2P_connector_v1.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L134>`__

.. code-block:: python

   def __init__(self, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Store connector metadata and construct its dense network.


``C2P_connector_v1.__call__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L147>`__

.. code-block:: python

   def __call__(self, input)

.. rubric:: Docstring

.. code-block:: text

   Fourier-encode periodic inputs, flatten molecules, and return features.


``CONFORMER_FLOW_LAYER`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L162>`__

.. code-block:: python

   class CONFORMER_FLOW_LAYER(periodic_mask, layer_index, n_mol, DIM_P2C_connection, DIM_C2P_connection, name='CONFORMER_FLOW_LAYER', half_layer_class=SPLINE_COUPLING_HALF_LAYER, kwargs_for_given_half_layer_class={'n_hidden': 2, 'dims_hidden': None, 'hidden_activation': tf.nn.leaky_relu}, use_tfp=False, n_bins=5, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], custom_coupling_mask=None, n_hidden_connection=1, connector_type=C2P_connector_v1)

.. rubric:: Docstring

.. code-block:: text

   Notes


``CONFORMER_FLOW_LAYER.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L165>`__

.. code-block:: python

   def __init__(self, periodic_mask, layer_index, n_mol, DIM_P2C_connection, DIM_C2P_connection, name='CONFORMER_FLOW_LAYER', half_layer_class=SPLINE_COUPLING_HALF_LAYER, kwargs_for_given_half_layer_class={'n_hidden': 2, 'dims_hidden': None, 'hidden_activation': tf.nn.leaky_relu}, use_tfp=False, n_bins=5, min_bin_width=0.001, knot_slope_range=[0.001, 50.0], custom_coupling_mask=None, n_hidden_connection=1, connector_type=C2P_connector_v1)

.. rubric:: Docstring

.. code-block:: text

   Construct one conformational spline and optional C-to-P connector.

   ``periodic_mask`` defines circular coordinates, ``layer_index`` or
   ``custom_coupling_mask`` selects transformed marginals, and auxiliary
   P-to-C features are conditioning-only dimensions.


``CONFORMER_FLOW_LAYER.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L253>`__

.. code-block:: python

   def forward_(self, input, aux)

.. rubric:: Docstring

.. code-block:: text

   Transform conformation variables conditioned on ``aux``.


``CONFORMER_FLOW_LAYER.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L264>`__

.. code-block:: python

   def inverse_(self, input, aux)

.. rubric:: Docstring

.. code-block:: text

   Invert conformation variables conditioned on ``aux``.


``CONFORMER_FLOW_LAYER.convert_to_aux_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L285>`__

.. code-block:: python

   def convert_to_aux_(self, input)

.. rubric:: Docstring

.. code-block:: text

   Encode conformational variables for conditioning a position layer.


``P3_version_SingleComponent_map_LITE`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L291>`__

.. code-block:: python

   class P3_version_SingleComponent_map_LITE(ic_map_OLD_version)

.. rubric:: Docstring

.. code-block:: text

   Allows models from the paper to be loaded,
   Only the ic_map = SingleComponent_map part of the model was changed slightly since the paper (P3)
   To be able to load all of the saved models from before,
   this class fixes the errors that would otherwise appear if loading those models with current code.


``P3_version_SingleComponent_map_LITE.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L300>`__

.. code-block:: python

   def __init__(self, ic_map_OLD_version)

.. rubric:: Docstring

.. code-block:: text

   Adapt a saved paper-era coordinate map to the current model API.


``P3_version_SingleComponent_map_LITE.permute_unitcell_tf_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L351>`__

.. code-block:: python

   def permute_unitcell_tf_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Apply the legacy unit-cell molecule permutation to coordinates.


``P3_version_SingleComponent_map_LITE.unpermute_unitcell_tf_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L359>`__

.. code-block:: python

   def unpermute_unitcell_tf_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Undo the legacy unit-cell molecule permutation.


``P3_version_SingleComponent_map_LITE.forward_reshape_cells_tf_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L368>`__

.. code-block:: python

   def forward_reshape_cells_tf_(self, x)

.. rubric:: Docstring

.. code-block:: text

   Fold legacy unit cells into the batch axis.


``P3_version_SingleComponent_map_LITE.inverse_reshape_cells_tf_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L373>`__

.. code-block:: python

   def inverse_reshape_cells_tf_(self, x)

.. rubric:: Docstring

.. code-block:: text

   Restore a separate legacy unit-cell axis from the batch axis.


``P3_version_SingleComponent_map_LITE.inverse_reshape_cells_tf_cat_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L378>`__

.. code-block:: python

   def inverse_reshape_cells_tf_cat_(self, x)

.. rubric:: Docstring

.. code-block:: text

   Restore legacy unit cells by concatenating them on the molecule axis.


``P3_version_SingleComponent_map_LITE.ln_base_C_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L384>`__

.. code-block:: python

   def ln_base_C_(self, z)

.. rubric:: Docstring

.. code-block:: text

   Return the saved legacy conformational base log density.


``P3_version_SingleComponent_map_LITE.ln_base_P_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L389>`__

.. code-block:: python

   def ln_base_P_(self, z)

.. rubric:: Docstring

.. code-block:: text

   Return the saved legacy positional base log density.


``P3_version_SingleComponent_map_LITE.sample_base_C_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L394>`__

.. code-block:: python

   def sample_base_C_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Sample legacy conformational variables uniformly on [-1, 1].


``P3_version_SingleComponent_map_LITE.sample_base_P_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L401>`__

.. code-block:: python

   def sample_base_P_(self, m)

.. rubric:: Docstring

.. code-block:: text

   Sample legacy positional variables uniformly on [-1, 1].


``P3_version_SingleComponent_map_LITE.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L408>`__

.. code-block:: python

   def forward_(self, r)

.. rubric:: Docstring

.. code-block:: text

   Apply the paper-era coordinate representation and Jacobian bookkeeping.


``P3_version_SingleComponent_map_LITE.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L454>`__

.. code-block:: python

   def inverse_(self, variables_in)

.. rubric:: Docstring

.. code-block:: text

   Invert paper-era representation variables to Cartesian coordinates.


``load_P3_PGMcrys`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L512>`__

.. code-block:: python

   def load_P3_PGMcrys(path_and_name, class_of_the_model)

.. rubric:: Docstring

.. code-block:: text

   Load a paper-era PGMcrys artifact through its compatibility map.

   The legacy pickle contains initialisation arguments and weights. Historical
   architecture constants are validated, attention-head count is inferred
   from the filename convention, and weights are assigned to a current model.


``model_helper_PGMcrys_v1`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L549>`__

.. code-block:: python

   class model_helper_PGMcrys_v1()

.. rubric:: Docstring

.. code-block:: text

   State-indexed representation, training, and sampling mixin for PGM models.


``model_helper_PGMcrys_v1.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L552>`__

.. code-block:: python

   def __init__(self)

.. rubric:: Docstring

.. code-block:: text

   Create a stateless mixin instance.


``model_helper_PGMcrys_v1._forward_represenation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L556>`__

.. code-block:: python

   def _forward_represenation_(self, r, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Map Cartesian coordinates through the selected state's representation.


``model_helper_PGMcrys_v1.forward_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L564>`__

.. code-block:: python

   def forward_(self, r, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Apply representation then trainable coupling: ``r -> x -> z``.


``model_helper_PGMcrys_v1._inverse_represenation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L572>`__

.. code-block:: python

   def _inverse_represenation_(self, X, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Invert representation variables with the selected state's map.


``model_helper_PGMcrys_v1.inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L580>`__

.. code-block:: python

   def inverse_(self, Z, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Apply inverse coupling then inverse representation: ``z -> x -> r``.


``model_helper_PGMcrys_v1.step_ML_graph_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L591>`__

.. code-block:: python

   def step_ML_graph_(self, r_and_crystal_index: list=0)

.. rubric:: Docstring

.. code-block:: text

   Perform one state-conditioned maximum-likelihood gradient update.


``model_helper_PGMcrys_v1.step_ML`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L612>`__

.. code-block:: python

   def step_ML(self, r, crystal_index: int=0, u=None, batch_size: int=1000)

.. rubric:: Docstring

.. code-block:: text

   Train on a random state-specific minibatch and return ``<u + ln q>``.


``model_helper_PGMcrys_v1.forward_graph_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L626>`__

.. code-block:: python

   def forward_graph_(self, r, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Graph-compatible state-indexed forward wrapper.


``model_helper_PGMcrys_v1.inverse_graph_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L632>`__

.. code-block:: python

   def inverse_graph_(self, z, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Graph-compatible state-indexed inverse wrapper.


``model_helper_PGMcrys_v1.forward`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L637>`__

.. code-block:: python

   def forward(self, r, crystal_index: int=0)

.. rubric:: Docstring

.. code-block:: text

   Convert input to float tensor and run the compiled forward map.


``model_helper_PGMcrys_v1.inverse`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L641>`__

.. code-block:: python

   def inverse(self, z, crystal_index: int=0)

.. rubric:: Docstring

.. code-block:: text

   Run the compiled inverse map for one state.


``model_helper_PGMcrys_v1.ln_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L645>`__

.. code-block:: python

   def ln_model(self, r, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Evaluate normalised log density ``ln q(r)`` for one state.


``model_helper_PGMcrys_v1.sample_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L652>`__

.. code-block:: python

   def sample_model(self, m: int, crystal_index: int=0)

.. rubric:: Docstring

.. code-block:: text

   Generate ``m`` state-specific Cartesian samples and log densities.


``model_helper_PGMcrys_v1.test_inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L660>`__

.. code-block:: python

   def test_inverse_(self, r, crystal_index=0, graph=True)

.. rubric:: Docstring

.. code-block:: text

   same as method with the same name in model_helper but has crystal_index as arg


``PGMcrys_v1`` (class)
^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L696>`__

.. code-block:: python

   class PGMcrys_v1(ic_maps: list, n_layers: int=4, optimiser_LR_decay=[0.001, 0.0], DIM_connection=10, n_att_heads=4, identity_init=False, initialise=True)

.. rubric:: Docstring

.. code-block:: text

   !! : molecule should have >3 atoms (also true in ic_map)


``PGMcrys_v1.load_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L699>`__

.. code-block:: python

   def load_model(path_and_name: str, VERSION='NEW')

.. rubric:: Docstring

.. code-block:: text

   Load a current single-file model or a paper-era legacy artifact.


``PGMcrys_v1.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L706>`__

.. code-block:: python

   def __init__(self, ic_maps: list, n_layers: int=4, optimiser_LR_decay=[0.001, 0.0], DIM_connection=10, n_att_heads=4, identity_init=False, initialise=True)

.. rubric:: Docstring

.. code-block:: text

   Build a position/conformation flow shared across crystal states.

   ``ic_maps`` supplies one compatible coordinate map per state. Each
   layer alternates conditioned position and conformation splines.
   ``DIM_connection`` controls exchanged feature width, ``n_att_heads``
   selects dense versus attention conformer conditioners, and
   ``identity_init`` starts spline transforms at identity when true.


``PGMcrys_v1.get_C2P_P2C_extensions_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L858>`__

.. code-block:: python

   def get_C2P_P2C_extensions_(self, m, crystal_index)

.. rubric:: Docstring

.. code-block:: text

   psi_{C->P}, psi_{P->C} are extended along last axis by 1 additional dimension
       This extra dimensions is called 'crystal encoding'.
       To be able to concateneate this crystal encoding,
       the batch size axes needs to match.
       The following steps adjust the batch axis:

   TODO: move this to layer, because this does not match the other types of layer.
       The other types of layer will use self.P2C_extension_shape for both P and C.


``PGMcrys_v1._forward_coupling_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L882>`__

.. code-block:: python

   def _forward_coupling_(self, X, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Apply alternating C-to-P and P-to-C spline couplings forward.


``PGMcrys_v1._inverse_coupling_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L904>`__

.. code-block:: python

   def _inverse_coupling_(self, Z, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Invert alternating couplings in exact reverse order.


``PGMmol`` (class)
^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L928>`__

.. code-block:: python

   class PGMmol(ic_maps: list, n_layers: int=4, optimiser_LR_decay=[0.001, 0.0], DIM_connection=None, n_att_heads=None, initialise=True)

.. rubric:: Docstring

.. code-block:: text

   molecule should have >3 atoms


``PGMmol.load_model`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L931>`__

.. code-block:: python

   def load_model(path_and_name: str)

.. rubric:: Docstring

.. code-block:: text

   Load an isolated-molecule model from the current artifact format.


``PGMmol.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L935>`__

.. code-block:: python

   def __init__(self, ic_maps: list, n_layers: int=4, optimiser_LR_decay=[0.001, 0.0], DIM_connection=None, n_att_heads=None, initialise=True)

.. rubric:: Docstring

.. code-block:: text

   Build a conformation-only flow for one isolated molecule.

   One compatible ``SingleMolecule_map`` may be supplied per metastable
   state. Translation and global rotation are absent, so only conformer
   coupling layers are created. ``DIM_connection`` and ``n_att_heads`` are
   retained for API compatibility and unused.


``PGMmol.get_extension_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L1028>`__

.. code-block:: python

   def get_extension_(self, m, crystal_index)

.. rubric:: Docstring

.. code-block:: text

   Return the broadcast scalar embedding for a molecular state.


``PGMmol._forward_coupling_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L1038>`__

.. code-block:: python

   def _forward_coupling_(self, X, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Transform isolated-molecule internal variables to the base space.


``PGMmol._inverse_coupling_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L1053>`__

.. code-block:: python

   def _inverse_coupling_(self, Z, crystal_index=0)

.. rubric:: Docstring

.. code-block:: text

   Invert isolated-molecule coupling layers in reverse order.


``PGMmol.test_inverse_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/NN/pgm.py#L1067>`__

.. code-block:: python

   def test_inverse_(self, r, crystal_index=0, graph=True)

.. rubric:: Docstring

.. code-block:: text

   same as method with the same name in model_helper but has crystal_index as arg
