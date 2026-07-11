"""Plotting and visualisation helpers for FECrys analyses.

Most functions draw on Matplotlib's current figure or create and display a new
figure. Coordinate inputs follow the FECrys convention of storing Cartesian
components on the final axis. Histogram-based information measures are
numerical estimates whose values depend on binning and coordinate units.
"""

from .util_np import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython.display import display

## ## ## ##

def plot_mol_larger_(mol):
    """Display a large 2D depiction of an RDKit molecule in a notebook.

    The first conformer is removed from ``mol`` before drawing, so this
    function mutates the supplied molecule. The generated image is 800 by 800
    pixels and is displayed through IPython.
    """
    mol.RemoveConformer(0)
    from rdkit.Chem import Draw
    from IPython.display import Image, display
    img = Draw.MolToImage(mol, size=(800, 800)) # Specify size here
    display(img)

def plot_1D_histogram_(x, bins=80, range=None, density=True, kwargs_for_histogram={},
                       ax=None,
                       return_max_y = False,
                       return_xy = False,
                       mask_0 = False,
                       **kwargs):
    """Calculate and plot a one-dimensional histogram at bin centres.

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
    """
    hist, x_grid = np.histogram(x, bins=bins, range=range, density=density, **kwargs_for_histogram)
    if mask_0:  hist = np.where(hist==0.0, np.nan, hist)
    else: pass
    x_grid = x_grid[1:] - 0.5*(x_grid[1]-x_grid[0])
    if ax is not None: ax.plot(x_grid, hist, **kwargs)
    else:              plt.plot(x_grid, hist, **kwargs)
    if return_max_y or return_xy: 
        if return_max_y and return_xy:
            return [x_grid, hist], hist[np.where(np.isfinite(hist))].max()
        elif return_max_y:
            return hist[np.where(np.isfinite(hist))].max()
        else:
            return [x_grid, hist]
    else: pass

def interpolate_colors_(m, c0 = [0,0,1,1], c1 = [1,0,0,1]):
    """Linearly interpolate ``m`` RGBA colours between two endpoints.

    ``c0`` and ``c1`` are four-component red, green, blue, and alpha values,
    normally in [0, 1]. The returned array has shape ``(m, 4)`` and can be
    passed to Matplotlib's ``colors`` or ``c`` arguments.
    """
    c0 = np.array(c0).astype(np.float64)
    c1 = np.array(c1).astype(np.float64)

    alpha = np.linspace(0,1,m)
    cs = np.array([c0*(1-a)+c1*a for a in alpha])
    return cs

def plot_2D_histogram_(x, y, bins=80, range=[None,None], weights=None, ax=None,
                       grid_transform_=None,
                       scatter = False,
                       **kwargs):
    """Plot a density-normalised two-dimensional histogram.

    ``x`` and ``y`` are flattened into paired observations. ``bins`` may be a
    scalar or a two-element list and ``range`` follows ``numpy.histogramdd``.
    An optional ``grid_transform_`` maps the two bin-centre grids before
    plotting. By default the density is drawn as contours; with ``scatter`` it
    is encoded as point colour. If ``ax`` is supplied, the Matplotlib artist is
    returned; otherwise the function draws through pyplot and returns ``None``.
    """
    if type(bins) is list: pass
    else: bins = [bins]*2
    xy = np.stack([x.flatten(), y.flatten()],axis=-1)
    hist, axs = np.histogramdd(xy, bins=bins, range=range, density=True, weights=weights)
    axs = [ax[1:]-0.5*(ax[1]-ax[0]) for ax in axs]
    AXs = joint_grid_from_marginal_grids_(*axs,flatten_output=False)
    if grid_transform_ is None: pass
    else: AXs = grid_transform_(AXs)

    if scatter:
        del kwargs['levels']
    else:
        if 's' in kwargs:
            kwargs['linewidths'] = kwargs['s']
            del kwargs['s']
        else: pass

    if ax is None:
        if scatter: plt.scatter(AXs[0].flatten(), AXs[1].flatten(), c=hist, **kwargs)
        else:       plt.contour(*AXs, hist, **kwargs)
    else:
        if scatter: return ax.scatter(AXs[0].flatten(), AXs[1].flatten(), c=hist, **kwargs)
        else:       return ax.contour(*AXs, hist, **kwargs)

def plot_2D_histogram_matshow_(x, y, bins=80, range=[None,None],  weights=None, **kwargs):
    """Display a 2D probability-density histogram as an index-space matrix.

    Unlike :func:`plot_2D_histogram_`, this helper does not construct physical
    coordinate axes. Additional keyword arguments are forwarded to
    ``matplotlib.pyplot.matshow``.
    """
    if type(bins) is list: pass
    else: bins = [bins]*2
    ''' 2D histogram without axes '''
    xy = np.stack([x.flatten(), y.flatten()],axis=-1)
    hist, axs = np.histogramdd(xy, bins=bins, range=range, density=True, weights=weights)
    plt.matshow(hist, **kwargs)

##

def plot_points_3D_(X,
                 show_axes=True,
                 figsize=(10,5),
                 autoscale=True,
                 show_colorbar=True,
                 view_elev_azim = [20,-10],
                 axes_labels = ['x','y','z'],
                 **kwargs):
    """Create and display a three-dimensional scatter plot.

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
    """
    #from mpl_toolkits.mplot3d import Axes3D
    from itertools import product, combinations

    fig = plt.figure( figsize=figsize )
    ax = fig.add_subplot(1,1,1,projection='3d')

    #draw cube
    Max = X.max()
    r = [-Max, Max]
    for s, e in combinations(np.array(list(product(r,r,r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s,e), color="black", alpha=1, linestyle='dotted', zorder=100)

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    if not show_axes: ax.set_axis_off()
    else: pass
    img = ax.scatter(X[:,0],X[:,1],X[:,2],**kwargs, zorder=1)
    ax.view_init(elev=28.+view_elev_azim[0], azim=45+view_elev_azim[0])
    if show_colorbar: fig.colorbar(img, shrink=0.5)
    else: pass
    plt.show()

def plot_points_(X,
                 show_axes=True,
                 figsize=(5, 3),
                 autoscale=True,
                 show_colorbar=True,
                 axes_labels = ['x','y','z'],
                 **kwargs):
    """Create and display a two- or three-dimensional scatter plot.

    ``X`` must have shape ``(n_points, n_dimensions)`` with two or at least
    three columns. The first three columns are plotted. Figure visibility,
    labels, colour bar, and scatter styling are controlled by the remaining
    arguments. The figure is shown immediately and is not returned.
    """
    dim = X.shape[1]
        
    fig = plt.figure(figsize=figsize)
    if dim >= 3: ax = fig.add_subplot(111, projection='3d') ; ax.set_zlabel(axes_labels[2])
    elif dim == 2: ax = fig.add_subplot(111)
    else: pass
    
    if autoscale: pass 
    else: ax.autoscale(enable=None, axis='both', tight=False)
        
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    if not show_axes: ax.set_axis_off()
    else: pass
    if dim >= 3: img = ax.scatter(X[:,0], X[:,1], X[:,2], **kwargs)
    elif dim == 2: img = ax.scatter(X[:,0], X[:,1], **kwargs)
    else: pass
    if show_colorbar: fig.colorbar(img)
    else: pass
    plt.show()

def save_coordiantes_as_pdb_(coordinates, name, la=None, verbose=False, box_line =''):
    """Write flattened coordinate frames to a minimal PDB trajectory.

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
    """
    coordinates = np.array(coordinates).round(decimals=2)
    n_frames = coordinates.shape[0]
    if la is None: values = np.zeros((n_frames,int(coordinates.shape[1]/3)))
    else: values = la
    pdb = open(name+'.pdb', "w")
    spaces = {0: "", 1: " ", 2: "  ", 3: "   ", 4: "    ", 5: "     ", 6: "      "}
    zeros =  {0: '', 1: '0', 2: '00', 3: '000', 4: '0000', 5: '00000'}
    frame = 0
    for i in range(0, n_frames):
        pdb.write(box_line + '\n')
        atom_index = 0
        stride = 0
        for j in range(0, int(len(coordinates[frame]) / 3)):
            atom_index = atom_index + 1
            pdb_row = []
            x_index = int(stride) ; stride = stride + 1
            y_index = int(stride) ; stride = stride + 1
            z_index = int(stride) ; stride = stride + 1
            x = str(coordinates[frame, x_index])
            y = str(coordinates[frame, y_index])
            z = str(coordinates[frame, z_index])
            n_spaces_to_add = 7 - len(str(atom_index))
            I = ("ATOM", spaces[n_spaces_to_add], str(atom_index), "  C   HET X")
            n_spaces_to_add = 4 - len(str(atom_index))
            II = (spaces[n_spaces_to_add], str(atom_index), "      ")
            n_0s_to_add = 6 - len(x)
            III = (x, zeros[n_0s_to_add], "  ")
            n_0s_to_add = 6 - len(y)
            IV = (y, zeros[n_0s_to_add], "  ")
            n_0s_to_add = 6 - len(z)
            if values[i,j] <0: sign = '-'
            else:              sign = ' '
            V = (z, zeros[n_0s_to_add], " "+sign+str(np.abs(float(values[i,j])).round(2))+" "+sign+str(np.abs(float(values[i,j])))+"           C")
            pdb_row.append(''.join(I))
            pdb_row.append(''.join(II))
            pdb_row.append(''.join(III))
            pdb_row.append(''.join(IV))
            pdb_row.append(''.join(V))
            pdb.write(''.join(pdb_row) + "\n")
        pdb.write("END" + "\n") # pdb.write("ENDMOL" + "\n")
        frame = frame + 1
        if verbose: print("new frame, "+str(i))
        else: pass
    pdb.close()
    print("saved",name+'.pdb')

##

def S_1D_(x, bins=80, range=None):
    """Estimate one-dimensional differential entropy from a histogram.

    Returns ``(entropy, bin_width)`` using the density estimate
    ``-sum(p * log(p)) * bin_width``. The result depends on the binning and the
    physical units of ``x``.
    """
    hist, ax = np.histogram(x, bins=bins, range=range, density=True)
    dx = ax[1]-ax[0]
    return - dx*(np.ma.log(hist)*hist).sum(), dx

def S_2D_(x, y, bins = [40,40], range = [None,None]):
    """Estimate joint differential entropy for two variables by histogramming.

    The returned scalar uses density-normalised bins and includes the bin-area
    factor. ``x`` and ``y`` are flattened before pairing.
    """
    xy = np.stack([x.flatten(),y.flatten()],axis=-1)
    hist, axs = np.histogramdd(xy, bins=bins, range=range, density=True)
    dxdy = np.prod([ax[1]-ax[0] for ax in axs])
    return - dxdy*(np.ma.log(hist)*hist).sum()

def MI_2D_(x, y, bins = [40,40], range = [None,None]):
    """Estimate normalised mutual information between two scalar variables.

    The joint histogram is converted to probabilities. The mutual-information
    numerator is divided by the joint entropy expression used internally, so
    the result is a normalised score rather than mutual information in nats.
    """
    xy = np.stack([x.flatten(),y.flatten()],axis=-1)
    h, axs = np.histogramdd(xy, bins=bins, range=range, density=True)
    h /= h.sum()
    mi = - (h * np.ma.log(np.ma.divide(h, np.outer(h.sum(1), h.sum(0))))).sum() / (h*np.ma.log(h)).sum() 
    return mi

##

'''
def plot_cell_(traj):
    import nglview as nv
    view = nv.show_mdtraj(traj)
    view.stage.set_parameters(cameraType= "orthographic")
    view.add_unitcell()
    return view#.display()
'''

# OLD:

def plot_vector_(v, r0, ax=None, color='black', **kwargs):
    """Draw one or more two-dimensional vectors from a common origin.

    ``v`` is an iterable of 2D displacement vectors and ``r0`` is expected to
    contain the origin as its first element. Lines are drawn on ``ax`` or on
    the current pyplot axes. This legacy helper returns ``None``.
    """
    r1 = np.array(r0)+np.array(v)
    if ax is None:
        for i in range(len(v)):
            plt.plot([r0[0][0],r1[i][0]],[r0[0][1],r1[i][1]],color=color, **kwargs)
    else:
        for i in range(len(v)):
            ax.plot([r0[0][0],r1[i][0]],[r0[0][1],r1[i][1]],color=color, **kwargs)

def plot_cell_(_traj, s=50, cmap='coolwarm', pad_ratio=0.3):
    """Plot three orthogonal projections of the first trajectory frame.

    This legacy water-specific visualisation assumes three atoms per molecule.
    It draws the unit-cell edges, displays the figure immediately, and returns
    ``None``. New code should prefer a general molecular viewer.
    """
    xyz = np.array(_traj.xyz[0])
    box = np.array(_traj.unitcell_vectors[0])
    Min, Max = xyz.min(0), xyz.max(0)
    pad = (Max-Min)*pad_ratio
    s *= (pad_ratio/0.7)
    Min -= pad
    Max += pad
    n_mol = xyz.shape[0]//3
    size = np.array([1,0.4,0.4]*n_mol)
    color = np.array([1,0.1,0.4]*n_mol)
    scale = 10
    fig,ax = plt.subplots(1,3,figsize=(scale,scale))
    axes = 'XYZ'
    for i in range(3):
        x = np.mod(i,3)
        y = np.mod(i+1,3)
        z = np.mod(i+2,3)
        ax[x].scatter(xyz[:,x],xyz[:,y],c=color,s=size*s, cmap=cmap)
        ax[x].set_xlabel(axes[x])
        ax[x].set_ylabel(axes[y])
        ax[x].axis('scaled')
        ax[x].set_xlim(Min[x],Max[x])
        ax[x].set_ylim(Min[y],Max[y])
        plot_vector_([box[x,[x,y]]], [np.array([0,0])], ax[x], color='black')
        plot_vector_([box[y,[x,y]]], [np.array([0,0])], ax[x], color='black')
        plot_vector_([box[x,[x,y]]], [box[y,[x,y]]], ax[x], color='black')
        plot_vector_([box[y,[x,y]]], [box[x,[x,y]]], ax[x], color='black')
        ax[x].set_box_aspect(1)
    fig.tight_layout()
    plt.show()

def plot_OH1H2_samples_2D_(samples, n_molecules:int, n_atoms_in_molecule:int,
                           pad_ratio=0.1,
                           s=0.01, colors = ['red', 'black', 'grey'],
                           scale = 10,
                           checking_disorder = False,
                           centre = True,
                          ):
    """Plot orthogonal O/H projections for water-like coordinate samples.

    The first three atoms of each molecule are interpreted as O, H1, and H2.
    Samples may be centred on the mean oxygen position. When
    ``checking_disorder`` is true, additional plots separate H1 and H2 to help
    diagnose label disorder. This is a legacy diagnostic that displays plots
    and returns ``None``.
    """
    samples = np.array(samples)
    m = samples.shape[0]
    _samples = reshape_to_molecules_np_(samples,n_atoms_in_molecule=n_atoms_in_molecule,n_molecules=n_molecules)[...,:3,:]
    if centre:
        mass = np.array([1.,0.,0.])[np.newaxis,np.newaxis,:,np.newaxis] # take COM of oxygens only.
        _samples -= (_samples*mass).sum(axis=(1,2), keepdims=True) / n_molecules
    else: pass
    O = _samples[:,:,0,:].reshape([m*n_molecules,3])
    H1 = _samples[:,:,1,:].reshape([m*n_molecules,3])
    H2 = _samples[:,:,2,:].reshape([m*n_molecules,3])

    Min, Max = _samples.min((0,1,2)), _samples.max((0,1,2))
    pad = (Max-Min)*pad_ratio
    s *= (pad_ratio/0.7)
    Min -= pad
    Max += pad
    size = np.array([1,0.4,0.4])*s
    if checking_disorder:
        fig,ax = plt.subplots(1,3,figsize=(scale,scale))
        axes = 'XYZ'
        for i in range(3):
            x = np.mod(i,3)
            y = np.mod(i+1,3)
            z = np.mod(i+2,3)
            ax[x].scatter(O[:,x],O[:,y],s=size[0], color=colors[0])
            ax[x].scatter(H1[:,x],H1[:,y],s=size[1], color=colors[1])
            #ax[x].scatter(H2[:,x],H2[:,y],s=size[2], color=colors[2])
            ax[x].set_xlabel(axes[x])
            ax[x].set_ylabel(axes[y])
            ax[x].axis('scaled')
            ax[x].set_xlim(Min[x],Max[x])
            ax[x].set_ylim(Min[y],Max[y])
            ax[x].set_box_aspect(1)
        fig.tight_layout()
        plt.show()
    
        fig,ax = plt.subplots(1,3,figsize=(scale,scale))
        axes = 'XYZ'
        for i in range(3):
            x = np.mod(i,3)
            y = np.mod(i+1,3)
            z = np.mod(i+2,3)
            ax[x].scatter(O[:,x],O[:,y],s=size[0], color=colors[0])
            #ax[x].scatter(H1[:,x],H1[:,y],s=size[1], color=colors[1])
            ax[x].scatter(H2[:,x],H2[:,y],s=size[2], color=colors[2])
            ax[x].set_xlabel(axes[x])
            ax[x].set_ylabel(axes[y])
            ax[x].axis('scaled')
            ax[x].set_xlim(Min[x],Max[x])
            ax[x].set_ylim(Min[y],Max[y])
            ax[x].set_box_aspect(1)
        fig.tight_layout()
        plt.show()
    else: pass
    fig,ax = plt.subplots(1,3,figsize=(scale,scale))
    axes = 'XYZ'
    for i in range(3):
        x = np.mod(i,3)
        y = np.mod(i+1,3)
        z = np.mod(i+2,3)
        ax[x].scatter(O[:,x],O[:,y],s=size[0], color=colors[0])
        ax[x].scatter(H1[:,x],H1[:,y],s=size[1], color=colors[1])
        ax[x].scatter(H2[:,x],H2[:,y],s=size[2], color=colors[2])
        ax[x].set_xlabel(axes[x])
        ax[x].set_ylabel(axes[y])
        ax[x].axis('scaled')
        ax[x].set_xlim(Min[x],Max[x])
        ax[x].set_ylim(Min[y],Max[y])
        ax[x].set_box_aspect(1)
    fig.tight_layout()
    plt.show()

def simple_smoothing_matrix_(S,N,c=1):
    """Construct a non-periodic Gaussian resampling matrix.

    Parameters ``N`` and ``S`` are the input and output grid sizes. ``c``
    controls kernel width: larger values produce narrower, more local weights.
    Returns an ``(S, N)`` matrix whose rows sum to one.
    """
    xs = np.linspace(0,1,S)
    z = np.linspace(0,1,N)
    c = -0.5/( (z[1]-z[0])/c )**2 ; W = []
    for x in xs:
        W.append(  np.exp(c*(z-x)**2)  )
    W = np.array(W)
    return W/W.sum(1)[:,np.newaxis]

def simple_smoother_(X,c=1.,S=None):
    """Gaussian-smooth and optionally resample an array along every axis.

    ``S`` may be ``None`` to preserve shape, an integer applied to all axes, or
    one output size per axis. ``c`` controls locality; values approaching zero
    approach broad averaging. The implementation supports at most thirteen
    dimensions and is intended for visualisation rather than inference.
    """
    # for smoothing arrays(X) shaped as (N1,) or (N1,N2) or any (N1,N2,...,Nd) up to d=13.
    ' c: lim(c -> 0) -> ~ line of best fit '
    ' S: can be None (output same shape), or an int, or a list of ints (one for each axis of the input array). '

    Ns = X.shape ; dim = len(Ns)
    
    if S is None: Ss = Ns
    elif type(S) is int: Ss = tuple([S]*dim)
    else: Ss = S
        
    summation_indices = 'ijklmnopqrstu'
    output_indices = 'abcdefghvwxyz'
    
    einsum_args = []
    string = ''

    for ax in range(dim):
        einsum_args.append(simple_smoothing_matrix_(Ss[ax],Ns[ax],c=c))
        string += (output_indices[ax] + summation_indices[ax] + ',')
    
    string += (summation_indices[:dim] + '->' + output_indices[:dim]) 
    einsum_args.insert(0,string)
    einsum_args.append(X)
    
    return np.einsum(*einsum_args)

def plot_as_ascii_(size = [300,900], c=1.2):
    """Render the current Matplotlib figure as a binary ASCII image.

    The figure is first saved as ``checking_plot.png`` in the current working
    directory, converted to greyscale, resampled to ``size`` and printed using
    ``#`` and spaces. ``c`` adjusts resampling sharpness. The temporary PNG is
    not removed and any existing file with that name is overwritten.
    """
    plt.savefig('checking_plot.png')
    
    from PIL import Image

    image = Image.open('checking_plot.png')
    X = np.array(image.convert('RGB').getdata()) # image.convert('RGB')
    X = X.reshape([image.size[1],image.size[0]]+ [X.shape[-1]]).mean(-1) # greyscale
    
    A,B = X.shape
    a,b = size
    Wa = simple_smoothing_matrix_(A,a,c=c)
    Wb = simple_smoothing_matrix_(B,b,c=c)
    Y = Wa.T.dot(X).dot(Wb)

    Y -= Y.min()
    Y /= Y.max()
    Y = 1.0 - Y
    Ymean = Y.mean()
    
    for row in Y:
        print(''.join(np.where(row>Ymean,'#',' ').tolist()))
        
def plot_2D_histograms_of_box_(b,
                               r0,
                               r1=None,
                               cmap='coolwarm', dpi=100,
                               bins = 1500,
                               m = None,
                               levels = 7,
                               aligment_square = 0,
                               scatter = False,
                               s = 1,
                               move_it_left = 0.0,
                              ):
    """Compare atomic-position densities on the three faces of a unit cell.

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
    """
    axes = [[0,1],[1,2],[0,2]]
    
    r0 = np.array(r0)[:m]
    if m is None:
        m = len(r0)
    else:
        pass 
    
    if r1 is None:
        r1 = np.array(r0)
    else:
        r1 = np.array(r1)[:m]

    assert len(r1) == len(r0)
    
    b = np.array(b)
    
    print('# samples histogramed:',m)
    
    x0 = np.mod(np.einsum('...i,ij->...j',r0,np.linalg.inv(b)),1.)
    x1 = np.mod(np.einsum('...i,ij->...j',r1,np.linalg.inv(b)),1.)

    this = np.einsum('...i,ij->...j',x0,b)

    _range = [[-this.max()*1.4,
                this.max()*1.4]]*2

    translation = [
    - np.array([ 0.0, 0.0 ]).reshape([2,1,1]),
    - np.array([ 0.2 + move_it_left , np.abs(b[2,2]) + 0.2 ]).reshape([2,1,1]),
    - np.array([ b[2,0], np.abs( b[2,2]) +0.2]).reshape([2,1,1]),
    ]
    flip = [
    np.array([1, 1]).reshape([2,1,1]),
    np.array([-1, 1]).reshape([2,1,1]),
    np.array([1, 1]).reshape([2,1,1]),
    ]

    fig, ax = plt.subplots(1, 2 + aligment_square, dpi=dpi, figsize=(10*2,4*2))
    for i in range(3):
        ax0, ax1 = axes[i]

        def grid_transformA_(x):
            """Map a fractional face grid to its displayed Cartesian plane."""
            A = ax0
            B = ax1
            b2D = b[[A, B]][:,[A, B]]
            # b2D /= np.linalg.norm(b2D, axis=-1,keepdims=True)
            # b2D *= np.linalg.norm(b, axis=-1)[[A,B]][:,np.newaxis]
            return np.einsum('i...,ij->j...',x,b2D)*flip[i]

        def grid_transform_(x):
            """Apply the face transform and its panel-layout translation."""
            return grid_transformA_(x) + translation[i]

        plot_2D_histogram_(x0[...,ax0],
                           x0[...,ax1],
                            bins = bins,
                           range = [[0.,1.]]*2,
                           grid_transform_ = grid_transform_,
                           levels = levels, cmap=cmap, ax = ax[0], scatter=scatter, s=s)

        plot_2D_histogram_(x1[...,ax0],
                           x1[...,ax1],
                            bins = bins,
                           range = [[0.,1.]]*2,
                           grid_transform_ = grid_transform_,
                           levels = levels, cmap=cmap, ax = ax[1], scatter=scatter, s=s)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_xlim(*_range[0])
    ax[0].set_ylim(*_range[0])
    ax[1].set_xlim(*_range[0])
    ax[1].set_ylim(*_range[0])
    
    if aligment_square:
        ax[2].plot([0,1],[0,1], linewidth=1, color='black')
        ax[2].plot([0,1],[1,0], linewidth=1, color='black')
        ax[2].set_xlim(*_range[0])
        ax[2].set_ylim(*_range[0])
        ax[2].set_xlim(*_range[0])
        ax[2].set_ylim(*_range[0])
    else: pass
        
    plt.tight_layout()
    plt.show()
