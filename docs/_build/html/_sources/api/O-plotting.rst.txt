.. _api-O-plotting:

O.plotting
==========

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py>`__

.. rubric:: Docstring

.. code-block:: text

   Plotting and visualisation helpers for FECrys analyses.

   Most functions draw on Matplotlib's current figure or create and display a new
   figure. Coordinate inputs follow the FECrys convention of storing Cartesian
   components on the final axis. Histogram-based information measures are
   numerical estimates whose values depend on binning and coordinate units.


Classes and functions
---------------------

``plot_mol_larger_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L16>`__

.. code-block:: python

   def plot_mol_larger_(mol)

.. rubric:: Docstring

.. code-block:: text

   Display a large 2D depiction of an RDKit molecule in a notebook.

   The first conformer is removed from ``mol`` before drawing, so this
   function mutates the supplied molecule. The generated image is 800 by 800
   pixels and is displayed through IPython.


``plot_1D_histogram_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L29>`__

.. code-block:: python

   def plot_1D_histogram_(x, bins=80, range=None, density=True, kwargs_for_histogram={}, ax=None, return_max_y=False, return_xy=False, mask_0=False, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Calculate and plot a one-dimensional histogram at bin centres.

   Parameters
   ----------
   x : array-like
       Observations passed to ``numpy.histogram``.
   bins, range, density
       Histogram configuration with NumPy semantics.
   kwargs_for_histogram : dict, optional
       Additional arguments for ``numpy.histogram``.
   ax : matplotlib.axes.Axes, optional
       Axes on which to draw; the current pyplot axes are used by default.
   return_max_y, return_xy : bool, default=False
       Select whether to return the maximum finite height, the bin centres
       and heights, or both.
   mask_0 : bool, default=False
       Replace zero-height bins with NaN so lines do not connect across them.
   **kwargs
       Styling arguments forwarded to ``Axes.plot``.

   Returns
   -------
   None, float, list, or tuple
       Return type depends on ``return_max_y`` and ``return_xy``. The plotted
       x values are bin centres, not bin edges.


``interpolate_colors_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L76>`__

.. code-block:: python

   def interpolate_colors_(m, c0=[0, 0, 1, 1], c1=[1, 0, 0, 1])

.. rubric:: Docstring

.. code-block:: text

   Linearly interpolate ``m`` RGBA colours between two endpoints.

   ``c0`` and ``c1`` are four-component red, green, blue, and alpha values,
   normally in [0, 1]. The returned array has shape ``(m, 4)`` and can be
   passed to Matplotlib's ``colors`` or ``c`` arguments.


``plot_2D_histogram_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L90>`__

.. code-block:: python

   def plot_2D_histogram_(x, y, bins=80, range=[None, None], weights=None, ax=None, grid_transform_=None, scatter=False, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Plot a density-normalised two-dimensional histogram.

   ``x`` and ``y`` are flattened into paired observations. ``bins`` may be a
   scalar or a two-element list and ``range`` follows ``numpy.histogramdd``.
   An optional ``grid_transform_`` maps the two bin-centre grids before
   plotting. By default the density is drawn as contours; with ``scatter`` it
   is encoded as point colour. If ``ax`` is supplied, the Matplotlib artist is
   returned; otherwise the function draws through pyplot and returns ``None``.


``plot_2D_histogram_matshow_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L127>`__

.. code-block:: python

   def plot_2D_histogram_matshow_(x, y, bins=80, range=[None, None], weights=None, **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Display a 2D probability-density histogram as an index-space matrix.

   Unlike :func:`plot_2D_histogram_`, this helper does not construct physical
   coordinate axes. Additional keyword arguments are forwarded to
   ``matplotlib.pyplot.matshow``.


``plot_points_3D_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L143>`__

.. code-block:: python

   def plot_points_3D_(X, show_axes=True, figsize=(10, 5), autoscale=True, show_colorbar=True, view_elev_azim=[20, -10], axes_labels=['x', 'y', 'z'], **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Create and display a three-dimensional scatter plot.

   Parameters
   ----------
   X : numpy.ndarray, shape (n_points, 3)
       Cartesian point coordinates.
   show_axes, show_colorbar : bool
       Control axes and colour-bar visibility.
   figsize : tuple
       Matplotlib figure size.
   autoscale : bool
       Retained for API compatibility; currently not used by this function.
   view_elev_azim : sequence of two numbers
       View adjustment. The current implementation uses its first value for
       both elevation and azimuth.
   axes_labels : sequence of str
       Labels for x, y, and z.
   **kwargs
       Styling and colour data forwarded to ``Axes.scatter``.

   Notes
   -----
   A dotted bounding cube centred at zero is drawn using the largest value in
   ``X``. The figure is shown immediately and is not returned.


``plot_points_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L200>`__

.. code-block:: python

   def plot_points_(X, show_axes=True, figsize=(5, 3), autoscale=True, show_colorbar=True, axes_labels=['x', 'y', 'z'], **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Create and display a two- or three-dimensional scatter plot.

   ``X`` must have shape ``(n_points, n_dimensions)`` with two or at least
   three columns. The first three columns are plotted. Figure visibility,
   labels, colour bar, and scatter styling are controlled by the remaining
   arguments. The figure is shown immediately and is not returned.


``save_coordiantes_as_pdb_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L235>`__

.. code-block:: python

   def save_coordiantes_as_pdb_(coordinates, name, la=None, verbose=False, box_line='')

.. rubric:: Docstring

.. code-block:: text

   Write flattened coordinate frames to a minimal PDB trajectory.

   Parameters
   ----------
   coordinates : array-like, shape (n_frames, n_atoms * 3)
       Cartesian coordinates, rounded to two decimal places before writing.
   name : str or path-like
       Output path without the ``.pdb`` suffix.
   la : array-like, shape (n_frames, n_atoms), optional
       Per-atom values written to the occupancy and B-factor-like fields.
       Zeros are used when omitted.
   verbose : bool, default=False
       Report every frame as it is written.
   box_line : str, optional
       Line inserted before the atoms in every frame, for example a CRYST1
       record.

   Notes
   -----
   The historical function-name spelling is retained for compatibility. The
   file is overwritten, all atoms are labelled as carbon, and each frame ends
   with ``END``. This lightweight writer is intended for visualisation.


``S_1D_`` (function)
^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L307>`__

.. code-block:: python

   def S_1D_(x, bins=80, range=None)

.. rubric:: Docstring

.. code-block:: text

   Estimate one-dimensional differential entropy from a histogram.

   Returns ``(entropy, bin_width)`` using the density estimate
   ``-sum(p * log(p)) * bin_width``. The result depends on the binning and the
   physical units of ``x``.


``S_2D_`` (function)
^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L318>`__

.. code-block:: python

   def S_2D_(x, y, bins=[40, 40], range=[None, None])

.. rubric:: Docstring

.. code-block:: text

   Estimate joint differential entropy for two variables by histogramming.

   The returned scalar uses density-normalised bins and includes the bin-area
   factor. ``x`` and ``y`` are flattened before pairing.


``MI_2D_`` (function)
^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L329>`__

.. code-block:: python

   def MI_2D_(x, y, bins=[40, 40], range=[None, None])

.. rubric:: Docstring

.. code-block:: text

   Estimate normalised mutual information between two scalar variables.

   The joint histogram is converted to probabilities. The mutual-information
   numerator is divided by the joint entropy expression used internally, so
   the result is a normalised score rather than mutual information in nats.


``plot_vector_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L355>`__

.. code-block:: python

   def plot_vector_(v, r0, ax=None, color='black', **kwargs)

.. rubric:: Docstring

.. code-block:: text

   Draw one or more two-dimensional vectors from a common origin.

   ``v`` is an iterable of 2D displacement vectors and ``r0`` is expected to
   contain the origin as its first element. Lines are drawn on ``ax`` or on
   the current pyplot axes. This legacy helper returns ``None``.


``plot_cell_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L370>`__

.. code-block:: python

   def plot_cell_(_traj, s=50, cmap='coolwarm', pad_ratio=0.3)

.. rubric:: Docstring

.. code-block:: text

   Plot three orthogonal projections of the first trajectory frame.

   This legacy water-specific visualisation assumes three atoms per molecule.
   It draws the unit-cell edges, displays the figure immediately, and returns
   ``None``. New code should prefer a general molecular viewer.


``plot_OH1H2_samples_2D_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L408>`__

.. code-block:: python

   def plot_OH1H2_samples_2D_(samples, n_molecules: int, n_atoms_in_molecule: int, pad_ratio=0.1, s=0.01, colors=['red', 'black', 'grey'], scale=10, checking_disorder=False, centre=True)

.. rubric:: Docstring

.. code-block:: text

   Plot orthogonal O/H projections for water-like coordinate samples.

   The first three atoms of each molecule are interpreted as O, H1, and H2.
   Samples may be centred on the mean oxygen position. When
   ``checking_disorder`` is true, additional plots separate H1 and H2 to help
   diagnose label disorder. This is a legacy diagnostic that displays plots
   and returns ``None``.


``simple_smoothing_matrix_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L495>`__

.. code-block:: python

   def simple_smoothing_matrix_(S, N, c=1)

.. rubric:: Docstring

.. code-block:: text

   Construct a non-periodic Gaussian resampling matrix.

   Parameters ``N`` and ``S`` are the input and output grid sizes. ``c``
   controls kernel width: larger values produce narrower, more local weights.
   Returns an ``(S, N)`` matrix whose rows sum to one.


``simple_smoother_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L510>`__

.. code-block:: python

   def simple_smoother_(X, c=1.0, S=None)

.. rubric:: Docstring

.. code-block:: text

   Gaussian-smooth and optionally resample an array along every axis.

   ``S`` may be ``None`` to preserve shape, an integer applied to all axes, or
   one output size per axis. ``c`` controls locality; values approaching zero
   approach broad averaging. The implementation supports at most thirteen
   dimensions and is intended for visualisation rather than inference.


``plot_as_ascii_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L544>`__

.. code-block:: python

   def plot_as_ascii_(size=[300, 900], c=1.2)

.. rubric:: Docstring

.. code-block:: text

   Render the current Matplotlib figure as a binary ASCII image.

   The figure is first saved as ``checking_plot.png`` in the current working
   directory, converted to greyscale, resampled to ``size`` and printed using
   ``#`` and spaces. ``c`` adjusts resampling sharpness. The temporary PNG is
   not removed and any existing file with that name is overwritten.


``plot_2D_histograms_of_box_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/plotting.py#L574>`__

.. code-block:: python

   def plot_2D_histograms_of_box_(b, r0, r1=None, cmap='coolwarm', dpi=100, bins=1500, m=None, levels=7, aligment_square=0, scatter=False, s=1, move_it_left=0.0)

.. rubric:: Docstring

.. code-block:: text

   Compare atomic-position densities on the three faces of a unit cell.

   Parameters
   ----------
   b : array-like, shape (3, 3)
       Supercell vectors stored by row.
   r0, r1 : array-like, shape (n_samples, n_atoms, 3)
       Coordinate ensembles shown in the left and right panels. If ``r1`` is
       omitted, ``r0`` is shown in both.
   bins, levels : int
       Histogram resolution and contour count.
   m : int, optional
       Maximum number of frames used from each ensemble.
   scatter : bool, default=False
       Draw density-coloured grid points instead of contours.
   aligment_square : int or bool, default=0
       Add a third reference panel when truthy. The misspelling is retained
       for API compatibility.
   cmap, dpi, s, move_it_left
       Plot styling and layout controls.

   Notes
   -----
   Coordinates are converted to wrapped fractional positions before their
   face projections are transformed back to Cartesian display coordinates.
   The function prints the sample count, shows the figure, and returns
   ``None``.
