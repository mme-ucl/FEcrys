"""General NumPy utilities used throughout FECrys.

The module contains array-shape conversions, molecular-geometry calculations,
trajectory processing, serialization helpers, and small numerical routines.
Unless stated otherwise, coordinate arrays use a final Cartesian axis of
length three and preserve any leading batch dimensions.
"""

from . import DIR_main
import sys

import os
import re
import copy
import time

from pathlib import Path
import subprocess
#import glob

import numpy as np
import scipy as sp

from rdkit import Chem
import mdtraj

import pickle

import textwrap

## ## 

def inject_methods_from_another_class_(target_instance, source_class, include_properties=False):
    """Attach methods from a class to one existing object.

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
    """
    import types
    for name, item in source_class.__dict__.items():
        
        if name.startswith("__"): continue
        else: pass
            
        if callable(item):
            setattr(target_instance, name, types.MethodType(item, target_instance))
        else: pass
            
        if include_properties and isinstance(item, (property, types.GetSetDescriptorType, types.MemberDescriptorType)):
            setattr(target_instance.__class__, name, item)
        else: pass

## ## 

def save_pickle_(x, name, verbose=True):
    """Serialize a Python object to a pickle file.

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
    """
    with open(name, "wb") as f: pickle.dump(x, f)
    if verbose: print('saved',name)
    else: pass
    
def load_pickle_(name):
    """Deserialize and return an object from a trusted pickle file.

    Parameters
    ----------
    name : str or path-like
        Pickle file to read.

    Returns
    -------
    object
        The object stored in ``name``.
    """
    with open(name, "rb") as f: x = pickle.load(f) ; return x

## ## 

def reshape_to_molecules_np_(r, n_molecules, n_atoms_in_molecule):
    """View batched coordinates as separate molecules.

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
    """
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules, n_atoms_in_molecule, 3])

def reshape_to_atoms_np_(r, n_molecules, n_atoms_in_molecule):
    """View batched coordinates as one atom array per frame.

    Returns an array of shape ``(n_frames, n_molecules *
    n_atoms_in_molecule, 3)`` without changing coordinate values. The input
    must contain exactly the required number of elements.
    """
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules*n_atoms_in_molecule, 3])
    
def reshape_to_flat_np_(r, n_molecules, n_atoms_in_molecule):
    """Flatten all molecular Cartesian coordinates within each frame.

    Returns an array of shape ``(n_frames, n_molecules *
    n_atoms_in_molecule * 3)``. The operation only changes the view/shape of
    the data.
    """
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules*n_atoms_in_molecule*3])

## ## 

def cumulative_average_(x, axis=None):
    """Return the running arithmetic mean of an array along ``axis``.

    With ``axis=None`` the input is flattened, following ``numpy.cumsum``.
    The output has the same shape as the cumulative sum and entry *i* is the
    mean up to and including entry *i*.
    """
    return np.cumsum(x, axis=axis) / np.cumsum(np.ones_like(x), axis=axis)


def sta_array_(x):
    """Min-max scale an array to the interval [0, 1].

    A constant input has zero range and therefore produces NaNs through NumPy
    division; callers should handle that case explicitly when it is possible.
    """
    return (x - x.min()) / (x.max() - x.min())

cdist_ = sp.spatial.distance.cdist

def half_way_(a,c):
    """Return the midpoint of two scalar values, independent of their order."""
    ac = sorted([a,c])
    b = min(ac) + 0.5*(max(ac) - min(ac))
    return b

def take_random_(x, m=20000):
    """Sample rows uniformly without replacement from the first axis.

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
    """
    return x[np.random.choice(x.shape[0],min([m,x.shape[0]]),replace=False)]

def find_split_indices_(u, split_where:int, tol=0.00001, verbose=True):
    """Find a random train/validation split with balanced mean energies.

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
    """
    u = np.array(u)
    n = u.shape[0]
    target = u.mean()
    for i in range(1000):
        inds_rand = np.random.choice(n,n,replace=False)
        randomised = np.array(u[inds_rand])
        if np.abs(randomised[:split_where].mean() - target) < tol and np.abs(randomised[split_where:].mean() - target) < tol:
            if verbose: print('found !')
            else: pass
            return inds_rand
        else: pass
    if verbose: print('! not found')
    else: pass
    return None

def joint_grid_from_marginal_grids_(*marginal_grids, flatten_output=True):
    """Construct the Cartesian product of one-dimensional marginal grids.

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
    """

    list_marginal_grids = list(marginal_grids)
    letters = 'jklmnopqrst'
    dim = len(list_marginal_grids)
    bins = [len(x) for x in list_marginal_grids]

    Xs = []
    string_input = 'io,'
    string_output = 'oi'
    for i in range(dim):
        X = np.ones([bins[i],dim])
        X[:,i] = np.array(list_marginal_grids[i])
        Xs.append(X)
        if i > 0:
            string_input += letters[i]+'o,'
            string_output += letters[i]
        else: pass

    string = string_input[:-1]+'->'+string_output #; print(string)
    
    joint_grid = np.einsum(string,*Xs)

    if flatten_output:
        joint_grid = joint_grid.T.reshape(-1, dim)
    else: pass

    return joint_grid

def tidy_crystal_xyz_(r, b, n_atoms_mol, ind_rO, batch_size=1000):
    """Remove periodic jumps from a single-component crystal trajectory.

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
    """
    def check_shape_(x):
        """Convert coordinates to an array with an explicit frame axis."""
        x = np.array(x)
        shape = len(x.shape)
        assert shape in [2,3]
        if len(x.shape) == 3: pass
        else: x = x[np.newaxis,...]
        return x

    r = check_shape_(r)
    n_frames = r.shape[0]
    batch_size = min([batch_size, n_frames])
    
    if len(b.shape) == 2: b = np.array([b]*n_frames)
    else: assert b.shape[0] == n_frames
    def wrap_points_(R, box):
        """Wrap Cartesian points into their corresponding periodic boxes."""
        # R   : (... 3), shaped as molecules
        # box : (...,3, 3) # rows
        st = 'oabi,oij->oabj'
        return np.einsum(st, np.mod(np.einsum(st, R, np.linalg.inv(box)), 1.0), box)
    
    N = r.shape[1]
    n_mol = N // n_atoms_mol
    assert n_mol == N / n_atoms_mol
    '''
    # step 1 : put atoms with index rO into box (and bring whole molecule with it)
    '''
    r = reshape_to_molecules_np_(r, n_atoms_in_molecule=n_atoms_mol, n_molecules=n_mol)
    for i in range(n_frames//batch_size):
        _from = i*batch_size
        _to = (i+1)*batch_size
        rO = r[_from:_to,:,ind_rO:ind_rO+1]
        r[_from:_to] = r[_from:_to] - rO + wrap_points_(rO,b[_from:_to])

    if n_frames - _to > 0:
        _from = _to
        rO = r[_from:,:,ind_rO:ind_rO+1]
        r[_from:] = r[_from:] - rO + wrap_points_(rO,b[_from:])
    else: pass
    '''
    # step 2: bring any atoms with index rO that are still jumping to pre-jump position (and bring whole molecule with it)
    using method copied from: https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/transformations/nojump.py
    this should give lattice looking like the first frame throughout a crystaline trajectory
    '''
    def dot_(Ri, mat):
        """Apply one 3-by-3 matrix to the final axis of each point."""
        st = 'abi,ij->abj'
        return np.einsum(st, Ri, mat)

    rO = np.array(r[:,:,ind_rO:ind_rO+1])
    b_inv = np.linalg.inv(b)
    
    rO_revised = np.zeros_like(rO)
    rO_revised[0] = rO[0]
    rO_0 = dot_(rO[0], b_inv[0])
    for i in range(1,n_frames):
        rO_1 = dot_(rO[i], b_inv[i])
        rO_1 -= np.round( rO_1 - rO_0 )
        rO_revised[i] = dot_(rO_1, b[i])

    r = r - rO + rO_revised
    # if remove_COM:
    r -= r[:,:,ind_rO:ind_rO+1].mean(1, keepdims=True)
    # else: pass
    r = reshape_to_atoms_np_(r, n_atoms_in_molecule=n_atoms_mol, n_molecules=n_mol)
    return r

## ## 

def get_torsion_np_(r, inds_4_atoms):
    """Calculate signed dihedral angles for four indexed atoms.

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
    """
    # r            : (..., # atoms, 3)
    # inds_4_atoms : (4,)
    
    A,B,C,D = inds_4_atoms
    rA = r[...,A,:] # (...,3)
    rB = r[...,B,:] # (...,3)
    rC = r[...,C,:] # (...,3)
    rD = r[...,D,:] # (...,3)
    
    vBA = rA - rB   # (...,3)
    vBC = rC - rB   # (...,3)
    vCD = rD - rC   # (...,3)

    _clip_low_at_ = 1e-8
    _clip_high_at_ = 1e+18
    clip_positive_ = lambda x : np.clip(x, _clip_low_at_, _clip_high_at_) 
    norm_clipped_ = lambda x : clip_positive_(np.linalg.norm(x,axis=-1,keepdims=True))
    unit_clipped_ = lambda x : x / norm_clipped_(x)
    
    uBC = unit_clipped_(vBC) # (...,3)

    w = vCD - np.sum(vCD*uBC, axis=-1, keepdims=True)*uBC # (...,3)
    v = vBA - np.sum(vBA*uBC, axis=-1, keepdims=True)*uBC # (...,3)
    
    uBC1 = uBC[...,0] # (...,)
    uBC2 = uBC[...,1] # (...,)
    uBC3 = uBC[...,2] # (...,)
    
    zero = np.zeros_like(uBC1) # (...,)
    S = np.stack([np.stack([ zero, uBC3,-uBC2],axis=-1),
                np.stack([-uBC3, zero, uBC1],axis=-1),
                np.stack([ uBC2,-uBC1, zero],axis=-1)],axis=-1) # (...,3,3)
    
    y = np.expand_dims(np.einsum('...j,...jk,...k->...',w,S,v), axis=-1) # (...,1)
    x = np.expand_dims(np.einsum('...j,...j->...',w,v), axis=-1)         # (...,1)
    
    phi = np.arctan2(y,x) # (...,1)

    return phi # (...,1)

def get_angle_np_(R, inds_3_atoms):
    """Calculate bond angles for three indexed atoms.

    ``inds_3_atoms`` defines A-B-C, with B as the vertex. Returns radians with
    shape ``(..., 1)``; values are clipped away from exactly 0 and pi for
    numerical stability.
    """
    # R            : (..., # atoms, 3)
    # inds_3_atoms : (3,)

    A,B,C = inds_3_atoms
    rA = R[...,A,:] # (...,3)
    rB = R[...,B,:] # (...,3)
    rC = R[...,C,:] # (...,3)

    _clip_low_at_ = 1e-8
    _clip_high_at_ = 1e+18
    clip_positive_ = lambda x : np.clip(x, _clip_low_at_, _clip_high_at_) 
    norm_clipped_ = lambda x : clip_positive_(np.linalg.norm(x,axis=-1,keepdims=True))
    unit_clipped_ = lambda x : x / norm_clipped_(x)

    uBA = unit_clipped_(rA - rB) # (...,3)
    uBC = unit_clipped_(rC - rB) # (...,3)

    dot = np.sum(uBA*uBC, axis=-1, keepdims=True)             # (...,1)
    dot = np.clip(dot, -1.0, 1.0)                           # (...,1)
    
    theta = np.arccos(dot) # (...,1)
    theta = np.clip(theta, _clip_low_at_, np.pi-_clip_low_at_) # (...,1)
 
    return theta # (...,1)

def get_distance_np_(R, inds_2_atoms):
    """Calculate Euclidean distances between two indexed atoms.

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
    """
    # R            : (..., # atoms, 3)
    # inds_2_atoms : (2,)
    A,B = inds_2_atoms
    rA = R[...,A,:]  # (...,3)
    rB = R[...,B,:]  # (...,3)
    vBA = rA - rB    # (...,3)

    _clip_low_at_ = 1e-8
    _clip_high_at_ = 1e+18
    clip_positive_ = lambda x : np.clip(x, _clip_low_at_, _clip_high_at_) 
    norm_clipped_ = lambda x : clip_positive_(np.linalg.norm(x,axis=-1,keepdims=True))

    return norm_clipped_(vBA) # (...,1)

## ## 

def color_text_(text, p='_R'):
    """Wrap text in ANSI terminal formatting codes.

    ``p`` selects a colour by its initial (for example ``'r'`` for red), an
    uppercase selector requests bold text, and an underscore requests
    underlining. The returned string includes a final reset code.
    """
    # REF: https://stackoverflow.com/questions/8924173/how-can-i-print-bold-text-in-python
    selection = ''
    if '_' in p: selection += '\033[4m'
    else: pass
    if p.isupper(): selection += '\033[1m'
    else: pass
    color = {   'p'   : '\033[95m', 'c'   : '\033[96m', 'dc'  : '\033[36m',
                'b'   : '\033[94m', 'g'   : '\033[92m', 'y'   : '\033[93m',
                'r'   : '\033[91m', 'o'   : '\033[38;5;208m', 'i' : '',
    }[p.replace('_', '').lower()]
    selection += color
    return selection + str(text) + '\033[0m'

## ## 

class TestConverged_1D:
    """Heuristic convergence diagnostic for a one-dimensional time series.

    The diagnostic compares the cumulative mean with its own cumulative mean,
    normalises the discrepancy by a running variance, and declares convergence
    when the final scaled error is no greater than ``tol``. It is a visual and
    exploratory heuristic, not a statistical hypothesis test.
    """

    def __init__(self,
                 x,
                 tol = 0.2,
                 verbose = True,
                ):
        """Calculate the convergence trace for ``x``.

        Parameters
        ----------
        x : array-like
            Scalar observations in time order; the input is flattened.
        tol : float, default=0.2
            Maximum final diagnostic value considered converged.
        verbose : bool, default=True
            Print the final convergence decision.
        """
        self.tol = tol
        
        x = np.array(x).flatten()
        MU = cumulative_average_(x)
        VAR = cumulative_average_((x-MU)**2)
        err = np.abs(MU - cumulative_average_(MU))**2
        err = np.ma.divide(err,VAR)**0.5
        err *= 10.0
        self.err = np.array(err)
        
        if verbose:
            gR_ = lambda _bool : ['R','g'][np.array(_bool).astype(np.int32)]
            b = self.__call__()
            print(f'with tol = {self.tol}, is converged: {color_text_(b, gR_(b))}')
        else: pass
        
        self.MU = MU
        self.x = np.array(x)

    def __call__(self):
        """Return whether the final diagnostic value meets the tolerance."""
        return self.err[-1] <= self.tol

    @property
    def where(self):
        """Indices at which the diagnostic is no greater than ``tol``."""
        return np.where(self.err <= self.tol)[0]
    
    @property
    def recommend_cut_from(self,):
        """Estimate and return an index after which the series is converged."""
        # index of frame after which the quantity may be converged
        idx = len(self.x) - len(TestConverged_1D(np.flip(self.x), tol=self.tol, verbose=False).where)
        if TestConverged_1D(self.x[idx:], tol=self.tol, verbose=False)():
            print('the quantity might be converged after frame with index:', idx)
        else: print('!!')
        return idx
    
    def show_(self, window=1, centre=False, show_x = True, color='black'):
        """Plot observations and their cumulative mean.

        Parameters control the y-axis half-width, centring about the final
        mean, visibility of raw observations, and plot colour. The plot is
        drawn on Matplotlib's current axes and the method returns ``None``.
        """
        # scatter is faster than plot
        import matplotlib.pyplot as plt
        mean = self.MU[-1]
        m = len(self.x)
        t = np.arange(m)
        
        if centre:
            if show_x: plt.scatter(t, self.x-mean, alpha=0.5,s=0.01, color=color)
            plt.scatter(t, self.MU - mean, alpha=1, s=1, color=color)
            plt.plot([0,m], [0]*2, color=color, linestyle='--')
            plt.ylim(-window, window)
        else:
            if show_x: plt.scatter(t, self.x, alpha=0.5,s=0.01, color=color)
            plt.scatter(t, self.MU, alpha=1, s=1, color=color)
            plt.plot([0,m], [mean]*2, color=color, linestyle='--')
            plt.ylim(mean-window, mean+window)
        
## ## 

def K_to_C_(K):
    """Convert an absolute temperature from kelvin to degrees Celsius."""
    return K - 273.15

def C_to_K_(C):
    """Convert a temperature from degrees Celsius to kelvin."""
    return C + 273.15

## ## 

def ADAM_np_(grad_,
            x0, 
            constraint_ = lambda x : x,
            max_itter = 1e20,
            alpha=0.005, 
            betas=[0.7,0.999], 
            tol=1e-4,
            ):
    """Minimise an objective using gradients and an Adam-like update.

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
    """

    beta1, beta2 = betas 
    one_minus_beta1 = 1.0-beta1
    one_minus_beta2 = 1.0-beta2
    eps = 1e-8

    v = np.zeros_like(x0)
    s = np.zeros_like(x0)

    a = 1.0
    grad =  np.ones_like(x0)*1e10

    while np.abs(grad).max() > tol and max_itter > a - 1.0:
        grad = grad_(x0)

        v = beta1*v + one_minus_beta1*grad
        s = beta2*s + one_minus_beta2*grad*grad

        _v = v #/ (1.0 - beta1**a)
        _s = s #/ (1.0 - beta2**a)

        sqrt_s_add_eps_inv = 1.0 / (_s**0.5 + eps)
        x0 = x0 - alpha*_v*sqrt_s_add_eps_inv
        x0 = constraint_(x0)
        a += 1.0
        
    return x0, a-1.0

## ## 











