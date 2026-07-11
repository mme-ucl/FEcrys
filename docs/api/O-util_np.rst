.. _api-O-util_np:

O.util_np
=========

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py>`__

.. rubric:: Docstring

.. code-block:: text

   General NumPy utilities used throughout FECrys.

   The module contains array-shape conversions, molecular-geometry calculations,
   trajectory processing, serialization helpers, and small numerical routines.
   Unless stated otherwise, coordinate arrays use a final Cartesian axis of
   length three and preserve any leading batch dimensions.


Classes and functions
---------------------

``inject_methods_from_another_class_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L33>`__

.. code-block:: python

   def inject_methods_from_another_class_(target_instance, source_class, include_properties=False)

.. rubric:: Docstring

.. code-block:: text

   Attach methods from a class to one existing object.

   Parameters
   ----------
   target_instance : object
       Object that will receive the methods. The object's class is not
       otherwise changed.
   source_class : type
       Class whose public callables are bound to ``target_instance``.
   include_properties : bool, default=False
       If true, also copy property and descriptor objects to the target's
       class. This affects every instance of the target class.

   Notes
   -----
   Names beginning with ``__`` are ignored. Existing attributes with the same
   names are overwritten. The operation mutates ``target_instance`` in place
   and returns ``None``.


``save_pickle_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L69>`__

.. code-block:: python

   def save_pickle_(x, name, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Serialize a Python object to a pickle file.

   Parameters
   ----------
   x : object
       Object to serialize.
   name : str or path-like
       Destination filename. Parent directories must already exist.
   verbose : bool, default=True
       Print the destination after a successful write.

   Notes
   -----
   The destination is overwritten. Pickle files must only be loaded from
   trusted sources because unpickling can execute arbitrary code.


``load_pickle_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L90>`__

.. code-block:: python

   def load_pickle_(name)

.. rubric:: Docstring

.. code-block:: text

   Deserialize and return an object from a trusted pickle file.

   Parameters
   ----------
   name : str or path-like
       Pickle file to read.

   Returns
   -------
   object
       The object stored in ``name``.


``reshape_to_molecules_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L107>`__

.. code-block:: python

   def reshape_to_molecules_np_(r, n_molecules, n_atoms_in_molecule)

.. rubric:: Docstring

.. code-block:: text

   View batched coordinates as separate molecules.

   Parameters
   ----------
   r : numpy.ndarray
       Coordinates with the number of frames on axis 0 and a compatible
       total number of Cartesian values in the remaining axes.
   n_molecules : int
       Number of molecules per frame.
   n_atoms_in_molecule : int
       Number of atoms in each molecule.

   Returns
   -------
   numpy.ndarray
       Reshaped coordinates with shape ``(n_frames, n_molecules,
       n_atoms_in_molecule, 3)``. No coordinate values are changed.


``reshape_to_atoms_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L129>`__

.. code-block:: python

   def reshape_to_atoms_np_(r, n_molecules, n_atoms_in_molecule)

.. rubric:: Docstring

.. code-block:: text

   View batched coordinates as one atom array per frame.

   Returns an array of shape ``(n_frames, n_molecules *
   n_atoms_in_molecule, 3)`` without changing coordinate values. The input
   must contain exactly the required number of elements.


``reshape_to_flat_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L139>`__

.. code-block:: python

   def reshape_to_flat_np_(r, n_molecules, n_atoms_in_molecule)

.. rubric:: Docstring

.. code-block:: text

   Flatten all molecular Cartesian coordinates within each frame.

   Returns an array of shape ``(n_frames, n_molecules *
   n_atoms_in_molecule * 3)``. The operation only changes the view/shape of
   the data.


``cumulative_average_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L151>`__

.. code-block:: python

   def cumulative_average_(x, axis=None)

.. rubric:: Docstring

.. code-block:: text

   Return the running arithmetic mean of an array along ``axis``.

   With ``axis=None`` the input is flattened, following ``numpy.cumsum``.
   The output has the same shape as the cumulative sum and entry *i* is the
   mean up to and including entry *i*.


``sta_array_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L161>`__

.. code-block:: python

   def sta_array_(x)

.. rubric:: Docstring

.. code-block:: text

   Min-max scale an array to the interval [0, 1].

   A constant input has zero range and therefore produces NaNs through NumPy
   division; callers should handle that case explicitly when it is possible.


``half_way_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L171>`__

.. code-block:: python

   def half_way_(a, c)

.. rubric:: Docstring

.. code-block:: text

   Return the midpoint of two scalar values, independent of their order.


``take_random_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L177>`__

.. code-block:: python

   def take_random_(x, m=20000)

.. rubric:: Docstring

.. code-block:: text

   Sample rows uniformly without replacement from the first axis.

   Parameters
   ----------
   x : numpy.ndarray
       Values to sample; axis 0 identifies observations.
   m : int, default=20000
       Maximum number of observations to return.

   Returns
   -------
   numpy.ndarray
       ``min(m, len(x))`` randomly ordered observations. The global NumPy
       random state controls reproducibility.


``find_split_indices_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L195>`__

.. code-block:: python

   def find_split_indices_(u, split_where: int, tol=1e-05, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Find a random train/validation split with balanced mean energies.

   Parameters
   ----------
   u : array-like, shape (n_samples, ...)
       Potential energies sampled during molecular dynamics. The mean is
       computed over all supplied values.
   split_where : int
       Number of samples assigned to the training prefix.
   tol : float, default=1e-5
       Maximum absolute difference allowed between each subset mean and the
       mean of the complete dataset.
   verbose : bool, default=True
       Report whether a suitable permutation was found.

   Returns
   -------
   numpy.ndarray or None
       A permutation of sample indices, or ``None`` if none of 1,000 random
       attempts meets the tolerance. Apply the same permutation to every
       aligned array; the first ``split_where`` entries form the training set.


``joint_grid_from_marginal_grids_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L233>`__

.. code-block:: python

   def joint_grid_from_marginal_grids_(*marginal_grids, flatten_output=True)

.. rubric:: Docstring

.. code-block:: text

   Construct the Cartesian product of one-dimensional marginal grids.

   Parameters
   ----------
   *marginal_grids : array-like
       One one-dimensional coordinate grid per dimension.
   flatten_output : bool, default=True
       Return a point table when true; retain the tensor grid when false.

   Returns
   -------
   numpy.ndarray
       Shape ``(prod(n_bins), n_dimensions)`` when flattened, otherwise
       ``(n_dimensions, *n_bins)``.

   Notes
   -----
   This is a convenience alternative to ``numpy.meshgrid``. The implementation
   currently supports at most the number of dimensions encoded by its einsum
   labels.


``tidy_crystal_xyz_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L283>`__

.. code-block:: python

   def tidy_crystal_xyz_(r, b, n_atoms_mol, ind_rO, batch_size=1000)

.. rubric:: Docstring

.. code-block:: text

   Remove periodic jumps from a single-component crystal trajectory.

   Parameters
   ----------
   r : numpy.ndarray, shape (n_frames, n_atoms, 3) or (n_atoms, 3)
       Cartesian coordinates. Each molecule must already be whole.
   b : numpy.ndarray, shape (n_frames, 3, 3) or (3, 3)
       Periodic box vectors stored by row. A single box is broadcast across
       frames.
   n_atoms_mol : int
       Number of atoms in each molecule; all molecules must have this size.
   ind_rO : int
       Within-molecule index of a slowly moving reference atom used to track
       each molecule through the periodic boundaries.
   batch_size : int, default=1000
       Number of frames processed together during initial wrapping.

   Returns
   -------
   numpy.ndarray, shape (n_frames, n_atoms, 3)
       Coordinates with molecular reference atoms unwrapped continuously and
       their global mean position removed. Molecular packing—and therefore
       periodic potential energy—should be unchanged.

   Notes
   -----
   The method assumes a stable crystal and may be unreliable for very small
   or unstable cells. Unwrap broken molecules before calling this function.


``tidy_crystal_xyz_.check_shape_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L313>`__

.. code-block:: python

   def check_shape_(x)

.. rubric:: Docstring

.. code-block:: text

   Convert coordinates to an array with an explicit frame axis.


``tidy_crystal_xyz_.wrap_points_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L328>`__

.. code-block:: python

   def wrap_points_(R, box)

.. rubric:: Docstring

.. code-block:: text

   Wrap Cartesian points into their corresponding periodic boxes.


``tidy_crystal_xyz_.dot_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L358>`__

.. code-block:: python

   def dot_(Ri, mat)

.. rubric:: Docstring

.. code-block:: text

   Apply one 3-by-3 matrix to the final axis of each point.


``get_torsion_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L383>`__

.. code-block:: python

   def get_torsion_np_(r, inds_4_atoms)

.. rubric:: Docstring

.. code-block:: text

   Calculate signed dihedral angles for four indexed atoms.

   Parameters
   ----------
   r : numpy.ndarray, shape (..., n_atoms, 3)
       Cartesian coordinates.
   inds_4_atoms : sequence of four int
       Atom indices ``(A, B, C, D)`` defining the A-B-C-D torsion.

   Returns
   -------
   numpy.ndarray, shape (..., 1)
       Signed angles in radians in the interval ``[-pi, pi]``.

   Notes
   -----
   Vector norms are clipped below at ``1e-8`` for numerical stability. The
   formulation is adapted from the bgflow project.


``get_angle_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L443>`__

.. code-block:: python

   def get_angle_np_(R, inds_3_atoms)

.. rubric:: Docstring

.. code-block:: text

   Calculate bond angles for three indexed atoms.

   ``inds_3_atoms`` defines A-B-C, with B as the vertex. Returns radians with
   shape ``(..., 1)``; values are clipped away from exactly 0 and pi for
   numerical stability.


``get_distance_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L475>`__

.. code-block:: python

   def get_distance_np_(R, inds_2_atoms)

.. rubric:: Docstring

.. code-block:: text

   Calculate Euclidean distances between two indexed atoms.

   Parameters
   ----------
   R : numpy.ndarray, shape (..., n_atoms, 3)
       Cartesian coordinates in any consistent length unit.
   inds_2_atoms : sequence of two int
       Indices of the atom pair.

   Returns
   -------
   numpy.ndarray, shape (..., 1)
       Pair distances in the input coordinate unit, clipped below at ``1e-8``.


``color_text_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L506>`__

.. code-block:: python

   def color_text_(text, p='_R')

.. rubric:: Docstring

.. code-block:: text

   Wrap text in ANSI terminal formatting codes.

   ``p`` selects a colour by its initial (for example ``'r'`` for red), an
   uppercase selector requests bold text, and an underscore requests
   underlining. The returned string includes a final reset code.


``TestConverged_1D`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L528>`__

.. code-block:: python

   class TestConverged_1D(x, tol=0.2, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Heuristic convergence diagnostic for a one-dimensional time series.

   The diagnostic compares the cumulative mean with its own cumulative mean,
   normalises the discrepancy by a running variance, and declares convergence
   when the final scaled error is no greater than ``tol``. It is a visual and
   exploratory heuristic, not a statistical hypothesis test.


``TestConverged_1D.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L537>`__

.. code-block:: python

   def __init__(self, x, tol=0.2, verbose=True)

.. rubric:: Docstring

.. code-block:: text

   Calculate the convergence trace for ``x``.

   Parameters
   ----------
   x : array-like
       Scalar observations in time order; the input is flattened.
   tol : float, default=0.2
       Maximum final diagnostic value considered converged.
   verbose : bool, default=True
       Print the final convergence decision.


``TestConverged_1D.__call__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L572>`__

.. code-block:: python

   def __call__(self)

.. rubric:: Docstring

.. code-block:: text

   Return whether the final diagnostic value meets the tolerance.


``TestConverged_1D.where`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L577>`__

.. code-block:: python

   def where(self)

.. rubric:: Docstring

.. code-block:: text

   Indices at which the diagnostic is no greater than ``tol``.


``TestConverged_1D.recommend_cut_from`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L582>`__

.. code-block:: python

   def recommend_cut_from(self)

.. rubric:: Docstring

.. code-block:: text

   Estimate and return an index after which the series is converged.


``TestConverged_1D.show_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L591>`__

.. code-block:: python

   def show_(self, window=1, centre=False, show_x=True, color='black')

.. rubric:: Docstring

.. code-block:: text

   Plot observations and their cumulative mean.

   Parameters control the y-axis half-width, centring about the final
   mean, visibility of raw observations, and plot colour. The plot is
   drawn on Matplotlib's current axes and the method returns ``None``.


``K_to_C_`` (function)
^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L617>`__

.. code-block:: python

   def K_to_C_(K)

.. rubric:: Docstring

.. code-block:: text

   Convert an absolute temperature from kelvin to degrees Celsius.


``C_to_K_`` (function)
^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L621>`__

.. code-block:: python

   def C_to_K_(C)

.. rubric:: Docstring

.. code-block:: text

   Convert a temperature from degrees Celsius to kelvin.


``ADAM_np_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/util_np.py#L627>`__

.. code-block:: python

   def ADAM_np_(grad_, x0, constraint_=lambda x: x, max_itter=1e+20, alpha=0.005, betas=[0.7, 0.999], tol=0.0001)

.. rubric:: Docstring

.. code-block:: text

   Minimise an objective using gradients and an Adam-like update.

   Parameters
   ----------
   grad_ : callable
       Function mapping the current parameter array to an equally shaped
       gradient array.
   x0 : numpy.ndarray
       Initial parameters. Updates preserve this shape.
   constraint_ : callable, optional
       Projection or transformation applied after every update.
   max_itter : int or float, default=1e20
       Maximum number of updates (the historical spelling is retained).
   alpha : float, default=0.005
       Step-size multiplier.
   betas : sequence of two float, default=(0.7, 0.999)
       Exponential decay factors for first and second gradient moments.
   tol : float, default=1e-4
       Stop when the largest absolute gradient component is at most this
       value.

   Returns
   -------
   x : numpy.ndarray
       Final constrained parameter values.
   n_iterations : float
       Number of updates performed.

   Notes
   -----
   Unlike canonical Adam, this implementation does not apply bias correction
   to the moment estimates. No objective value or convergence flag is
   returned.
