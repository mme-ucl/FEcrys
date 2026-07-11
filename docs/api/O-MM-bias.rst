.. _api-O-MM-bias:

O.MM.bias
=========

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py>`__

.. warning:: Module docstring pending.


Classes and functions
---------------------

``_potential_energy_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L83>`__

.. code-block:: python

   def _potential_energy_(self, r, b=None, which='u+v')

.. rubric:: Docstring

.. code-block:: text

   Evaluate physical and/or bias energy for a coordinate batch.

   Parameters
   ----------
   self : SingleComponent
       Simulation object to which :class:`BIAS` has attached energy-group
       properties.
   r : array-like, shape (n_frames, n_atoms, 3)
       Cartesian coordinates in the units expected by ``_set_r_``.
   b : array-like, shape (n_frames, 3, 3), optional
       Periodic boxes. The current context box is retained when omitted.
   which : {"u", "v", "u+v", "v+u"}
       Select physical, bias, or total potential energy.

   Returns
   -------
   numpy.ndarray, shape (n_frames, 1)
       Reduced energies in ``kT``.

   Notes
   -----
   The active context is updated once per frame. Its original coordinates and
   box are restored before returning.


``BIAS`` (class)
^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L131>`__

.. code-block:: python

   class BIAS(sc)

.. rubric:: Docstring

.. code-block:: text

   Manage static bias forces attached to a ``SingleComponent`` system.

   The manager records each bias constructor, keeps bias forces in OpenMM
   group 15, and exposes separate physical/bias energy evaluation. Construct
   and add biases before reinitialising the OpenMM simulation context.


``BIAS.set_potential_energy_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L139>`__

.. code-block:: python

   def set_potential_energy_(self, define_current_CV_attribute=False)

.. rubric:: Docstring

.. code-block:: text

   Attach force-group energy accessors to the simulation object.

   Group 15 is reserved for bias forces; all earlier groups are treated
   as the physical system. The method adds ``_current_BIAS_``,
   ``_current_U_``, ``_current_U_add_BIAS_``, and ``potential_energy_`` to
   the simulation object's class/instance. Energies read directly from
   these properties are in kJ/mol. ``define_current_CV_attribute`` is
   retained for a planned CustomCVForce implementation and currently has
   no effect.


``BIAS.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L193>`__

.. code-block:: python

   def __init__(self, sc)

.. rubric:: Docstring

.. code-block:: text

   Initialise an empty bias registry for a configured simulation.

   ``sc.system`` must not already contain force-group-15 forces. The
   constructor does not modify the OpenMM system until :meth:`add_bias_`
   is called.


``BIAS.all_groups`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L222>`__

.. code-block:: python

   def all_groups(self)

.. rubric:: Docstring

.. code-block:: text

   Force-group number for every force in current system order.


``BIAS.add_bias_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L226>`__

.. code-block:: python

   def add_bias_(self, cls, name=None, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Construct, register, and add one bias to the OpenMM system.

   ``cls`` must accept ``sc=...`` plus ``kwargs`` and expose a
   ``bias_related_forces`` sequence. ``name`` must be unique; an integer
   sequence number is used when omitted. The method updates accounting,
   installs energy accessors, marks the simulation as biased, and returns
   ``None``. Reinitialise the simulation context before running dynamics.


``BIAS.remove_all_bias_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L271>`__

.. code-block:: python

   def remove_all_bias_(self)

.. rubric:: Docstring

.. code-block:: text

   Remove all group-15 forces and reset the bias registry.

   This mutates ``sc.system`` but does not rebuild an existing simulation
   context. The method asserts that the original physical-force count is
   restored.


``BIAS.remove_all_bias_.index_bias_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L281>`__

.. code-block:: python

   def index_bias_()

.. rubric:: Docstring

.. code-block:: text

   Return current system indices of all force-group-15 forces.


``BIAS.save_simulation_data_zero_bias_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L321>`__

.. code-block:: python

   def save_simulation_data_zero_bias_(self, path_and_name: str=None, dont_save_just_stat=False, eps=1e-10)

.. rubric:: Docstring

.. code-block:: text

   Select and optionally save frames on which the bias is effectively zero.

   Bias energies below ``eps`` (in reduced ``kT`` units) are classified as
   unbiased. The method reports retained fraction and mean physical
   energies, stores biased/unbiased indices and statistics on this object,
   and serializes a restart-style dataset unless ``dont_save_just_stat``
   is true. ``path_and_name`` is passed directly to ``save_pickle_``.


``_set_inds_torsion_for_bias_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L367>`__

.. code-block:: python

   def _set_inds_torsion_for_bias_(self, inds_torsion)

.. rubric:: Docstring

.. code-block:: text

   Map one molecule's torsion indices to every molecule in the topology.

   Returns the four original within-molecule indices and an integer array of
   shape ``(n_mol, 4)`` in simulation topology order.


``_set_means_torsions_for_bias_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L387>`__

.. code-block:: python

   def _set_means_torsions_for_bias_(self, means)

.. rubric:: Docstring

.. code-block:: text

   Broadcast a repeating sequence of torsion centres across molecules.

   A scalar is repeated for every molecule. A sequence must have a length
   dividing ``n_mol`` and is tiled in molecule order. Returns shape
   ``(n_mol,)`` in radians.


``WALLS`` (class)
^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L408>`__

.. code-block:: python

   class WALLS(sc, inds_torsion: list, means: list, width_percentage: float=68.17, height=200)

.. rubric:: Docstring

.. code-block:: text

   Confine a torsion to metastable wells with smooth periodic walls.

   This static bias discourages rare transitions out of selected torsional
   basins. It is intended for cases where most sampled frames remain unbiased;
   a retained zero-bias fraction above roughly 95% is a useful diagnostic.

   example for case of n_mol=16:

       sc.inject_methods_from_another_class_(WALLS)
       av_value = 1.88 # minimiser of unbiased marginal FES that would be sampled in the absence of the rare event
       sc.set_walls_(inds_torsion=[18, 15, 10,  8], means=[-av_value,av_value])
       Verbose output:
           inds_torsion provided: [18 15 10  8]
           inds_torsion for self.topology: [5, 4, 15, 13] # can be different if permuation of atoms used
           16 means: [-1.6  1.6 -1.6  1.6 -1.6  1.6 -1.6  1.6 -1.6  1.6 -1.6  1.6 -1.6  1.6
           -1.6  1.6]
           bias: system was updated to include bias, run initialise_simulation_ to apply this change to simulation


``WALLS.smooth_periodic_square_well_v2_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L440>`__

.. code-block:: python

   def smooth_periodic_square_well_v2_(self, phi, centre)

.. rubric:: Docstring

.. code-block:: text

   Evaluate the wall energy for angles ``phi`` around ``centre``.

   Angles are periodic in radians. The returned NumPy array is zero inside
   the permitted region and rises quadratically in the transformed
   distance to ``self.height`` at the periodic boundary.


``WALLS.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L455>`__

.. code-block:: python

   def __init__(self, sc, inds_torsion: list, means: list, width_percentage: float=68.17, height=200)

.. rubric:: Docstring

.. code-block:: text

   Build one periodic wall torsion for every molecule.

   ``inds_torsion`` gives four within-molecule atom indices. ``means`` is
   one or a repeating sequence of desired angles in radians.
   ``width_percentage`` is the unbiased fraction of the full 2-pi domain,
   and ``height`` is the maximum wall energy in kJ/mol. The constructed
   group-15 ``CustomTorsionForce`` is exposed in ``bias_related_forces``
   but is not itself added to the system.


``WALLS.check_plot_torsion_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L491>`__

.. code-block:: python

   def check_plot_torsion_(self, r, plot=True, inds=None)

.. rubric:: Docstring

.. code-block:: text

   Calculate and optionally plot the biased torsion for every molecule.

   ``r`` is reshaped into molecules and the returned angles have shape
   ``(n_frames, n_mol)`` in radians. When plotting, ``inds`` optionally
   highlights selected frames in red and the wall profile is scaled to
   each histogram for visual comparison.


``FLOOR`` (class)
^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L523>`__

.. code-block:: python

   class FLOOR(sc, inds_torsion: list, percentage_raise: float=50.0, specific_AD=True)

.. rubric:: Docstring

.. code-block:: text

   Raise selected torsional wells with an accelerated-MD-style bias.

   The bias is applied to matching ``PeriodicTorsionForce`` and
   ``RBTorsionForce`` terms. Its practical sampling effectiveness remains
   experimental and should be checked by reweighting diagnostics.


``FLOOR.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L535>`__

.. code-block:: python

   def __init__(self, sc, inds_torsion: list, percentage_raise: float=50.0, specific_AD=True)

.. rubric:: Docstring

.. code-block:: text

   Construct floor forces for a selected molecular torsion.

   ``percentage_raise`` chooses the fraction of each torsion's energy
   range below which the floor acts. With ``specific_AD=True``, only exact
   A-B-C-D terms (in either direction) match; otherwise every term rotating
   about the selected B-C bond matches. Generated forces are assigned to
   group 15 and collected in ``bias_related_forces``.


``FLOOR.__init__.ADD_`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L572>`__

.. code-block:: python

   def ADD_(a1, a2, a3, a4)

.. rubric:: Docstring

.. code-block:: text

   Return whether one force-field torsion matches the selection.


``FLOOR.check_plot_torsion_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/bias.py#L673>`__

.. code-block:: python

   def check_plot_torsion_(self, r, plot=True, weights=None)

.. rubric:: Docstring

.. code-block:: text

   Calculate and optionally plot raw and reweighted torsion histograms.

   Returns angles with shape ``(n_frames, n_mol)`` in radians. ``weights``
   must provide one value per frame and is used for the black reweighted
   histogram; the unweighted distribution is shown in red.
