from .mm_helper import *

"""Static torsional biases for FECrys OpenMM simulations.

Bias forces are assigned to OpenMM force group 15. Physical potential energy
``u``, bias energy ``v``, and their sum can therefore be evaluated separately.
Public energy arrays returned by this module are reduced by ``beta`` and are
expressed in units of ``kT`` unless stated otherwise.
"""

## ## ## ## ## ## ## ## ## ## ## ##


''' static bias example: WALLS (prevent transitions)

# prevent rare events using static bias that allows enough unbiased data

dataset = load_pickle_(path_dataset)
sc = SingleComponent(**dataset['args_initialise_object'])
sc.initialise_system_(**dataset['args_initialise_system'])
bias = BIAS(sc)
bias.add_bias_(WALLS, name='walls1', inds_torsion=[26, 20, 18, 15], means=[np.pi], width_percentage=30)
sc.initialise_simulation_(**dataset['args_initialise_simulation'])
n_frames = 100000 # NB: rate events may try to happen once or twice in a 600k steps (60ns) trajectory
sc.run_simulation_(n_frames, 50)
bias.save_simulation_data_zero_bias_(dont_save_just_stat=True) # shows % of unbiased data; should generally be > 95% when WALLS are useful.
bias.biasing_instances['walls1'].check_plot_torsion_(sc.xyz, inds=bias.inds_biased) # biased : red scatter, unbiased: black
bias.save_simulation_data_zero_bias_('check')

load_pickle_('check')['MD dataset']['xyz'].shape # = (% unbiased) * n_frames ; becaused saving only unbiased data.
'''

## ## ## ## ## ## ## ## ## ## ## ##

''' static bias example: FLOOR (may or may not encourage transitions)

# raise the floor of specific torsional energy well for a given torsion such that transitions are sampled ergodically

# same function as http://docs.openmm.org/7.3.0/api-python/generated/simtk.openmm.amd.AMDIntegrator.html
# but applied only to the specific torsional angle energy, allowing to keep the same thermostat as default.
# specific torsion(s) better to reweighting efficiency, and hopefully less frames to sample ergodically (if inds_torsion chosen well).


dataset = load_pickle_(path_dataset)
sc = SingleComponent(**dataset['args_initialise_object'])
sc.initialise_system_(**dataset['args_initialise_system'])

bias = BIAS(sc)
bias.add_bias_(FLOOR, name='floor0', inds_torsion=[26, 20, 18, 15], percentage_raise=50, specific_AD =False)
bias.add_bias_(FLOOR, name='floor1', inds_torsion=[18, 20, 25, 22], percentage_raise=60, specific_AD =True)
bias.add_bias_(FLOOR, name='floor2', inds_torsion=[20, 25, 22, 19], percentage_raise=60, specific_AD =True)
bias.add_bias_(FLOOR, name='floor3', inds_torsion=[25, 22, 19, 18], percentage_raise=60, specific_AD =True)
bias.add_bias_(FLOOR, name='floor4', inds_torsion=[22, 19, 18, 20], percentage_raise=60, specific_AD =True)
bias.add_bias_(FLOOR, name='floor5', inds_torsion=[19, 18, 20, 25], percentage_raise=60, specific_AD =True)

sc.initialise_simulation_(**dataset['args_initialise_simulation'])
n_frames = 5000 # should cross, else adjust setting or use another method.
sc.run_simulation_(n_frames, 50)
v = sc.potential_energy_(sc.xyz, which='v')
bias.biasing_instances['floor0'].check_plot_torsion_(sc.xyz, weights=np.exp(v))

# TODO ADD: bias.save_biased_simulation_data_(dont_save_just_stat=True) to show if reweighting works well (effective sample size) 
# TODO ADD: allow leading from save to get back the same instance as for trajectory, because potential_energy_ is needed (unlike in WALLS)

'''
## ## ## ## ## ## ## ## ## ## ## ##

''' static bias example: arbitrary function (e.g., stationary bias from WTmetaD)

# splines : less efficent compared to analytical extression, but allows 2D functions

'''

## ## ## ## ## ## ## ## ## ## ## ##

''' time-dependant bias example: WTmetaD TODO 

# splines

'''
## ## ## ## ## ## ## ## ## ## ## ##

def _potential_energy_(self, r, b=None, which='u+v'):
    """Evaluate physical and/or bias energy for a coordinate batch.

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
    """

    if which == 'u':             energy_ = lambda : self._current_U_
    elif which == 'v':           energy_ = lambda : self._current_BIAS_
    elif which in ['u+v','v+u']: energy_ = lambda : self._current_U_add_BIAS_ # faster (= u + v above)
    else: assert which in ['u', 'v','u+v', 'v+u']

    n_frames = r.shape[0]
    _r = np.array(self._current_r_)
    _b = np.array(self._current_b_)
    energy = np.zeros([n_frames,1]) # kJ/mol
    if b is None:
        for i in range(n_frames):
            self._set_r_(r[i])
            energy[i,0] = energy_()
    else:
        for i in range(n_frames):
            self._set_r_(r[i])
            self._set_b_(b[i])
            energy[i,0] = energy_()
    self._set_r_(_r)
    self._set_b_(_b)
    return energy * self.beta # kT

class BIAS:
    """Manage static bias forces attached to a ``SingleComponent`` system.

    The manager records each bias constructor, keeps bias forces in OpenMM
    group 15, and exposes separate physical/bias energy evaluation. Construct
    and add biases before reinitialising the OpenMM simulation context.
    """

    def set_potential_energy_(self, define_current_CV_attribute=False):
        """Attach force-group energy accessors to the simulation object.

        Group 15 is reserved for bias forces; all earlier groups are treated
        as the physical system. The method adds ``_current_BIAS_``,
        ``_current_U_``, ``_current_U_add_BIAS_``, and ``potential_energy_`` to
        the simulation object's class/instance. Energies read directly from
        these properties are in kJ/mol. ``define_current_CV_attribute`` is
        retained for a planned CustomCVForce implementation and currently has
        no effect.
        """

        all_groups = self.all_groups

        CUT = np.where(np.array(all_groups)==15)[0][0]

        groups_U = all_groups[:CUT] # physical system
        groups_V = all_groups[CUT:] # bias only in group 15. Added at the end -> should be at the end of this list.
        assert all([x!=15 for x in groups_U]) # checking that the list has no group 15 member earlier on in the list
        assert all([x==15 for x in groups_V]) # checking that all group 15 members are together at the end
        assert len(groups_U) == self.N_DEFAULT_FORCES

        # V [kJ/mol] is the group 15 energy:
        self.sc.__class__._current_BIAS_ = property(
        lambda self : self.simulation.context.getState(getEnergy=True, groups={15}).getPotentialEnergy()._value
        ) # this definition is new and used only in the self.potential_energy_ function

        # U [kJ/mol] is the potential energy of the physical system:
        self.sc.__class__._current_U_ = property(
        lambda self : self.simulation.context.getState(getEnergy=True, groups=set(groups_U)).getPotentialEnergy()._value
        ) # this replacement makes self._current_u_ [kT], already defined in self.sc, to be the reduced energy of the physical system

        # U + V [kJ/mol] is potential energy from all groups:
        self.sc.__class__._current_U_add_BIAS_ = property(
        lambda self : self.simulation.context.getState(getEnergy=True, groups=set(all_groups)).getPotentialEnergy()._value
        ) # this definition is new and used only in the self.potential_energy_ function

        ''' not using mm.CustomCVForce yet, but keeping for later (TODO: add metadynamics) 
        self._inds_CV_forces_system_ = np.where(np.array(all_groups)==15)[0]
        if define_current_CV_attribute:
            self.sc.__class__._current_CV_ = property(
            lambda self : np.array([
                self.system.getForces()[i].getCollectiveVariableValues(self.simulation.context)[0] # (scalar) -> scalar 
                for i in self._inds_CV_forces_system_
                if self.system.getForces()[i].getName() != 'CustomTorsionForce'
                ]) # (n_mol,) ; one 1D CV in each molecule
            )
        else: print('self._current_CV_ not available, use check_plot_torsion_ instead')
        '''

        print('bias: system was updated to include bias, run initialise_simulation_ to apply this change to simulation')

        self.sc.potential_energy_ = lambda r, b=None, which='u+v' : _potential_energy_(self=self.sc, r=r, b=b, which=which)

    def __init__(self, sc):
        """Initialise an empty bias registry for a configured simulation.

        ``sc.system`` must not already contain force-group-15 forces. The
        constructor does not modify the OpenMM system until :meth:`add_bias_`
        is called.
        """
        self.sc = sc

        all_groups = self.all_groups
        no_bias_at_start = all([x!=15 for x in all_groups])

        if no_bias_at_start:
            self.N_DEFAULT_FORCES = len(all_groups)
        else:
            print('sc.system.getForces() already has forces in group 15')
            print('this might indicate that an instance BIAS(sc) was already initialised earlier')
            print('in that case please add and remove bias using an existing instance of BIAS')
            assert all([x!=15 for x in all_groups])

        self.biasing_instances = {}
        self.inds_biasing_forces = {}
        self.list_args_add_bias = [] # log to be able to replicate the same exact list (sc.simulation.system.getForces())
        self.n_forces_total = int(self.N_DEFAULT_FORCES)
        self.n_bias = 0
        self.sc.biased = False
        self.evalauted_bias = None

    @property
    def all_groups(self,):
        """Force-group number for every force in current system order."""
        return [force.getForceGroup() for force in self.sc.system.getForces()]

    def add_bias_(self, cls, name=None, **kwargs):
        """Construct, register, and add one bias to the OpenMM system.

        ``cls`` must accept ``sc=...`` plus ``kwargs`` and expose a
        ``bias_related_forces`` sequence. ``name`` must be unique; an integer
        sequence number is used when omitted. The method updates accounting,
        installs energy accessors, marks the simulation as biased, and returns
        ``None``. Reinitialise the simulation context before running dynamics.
        """

        args_add_bias = {'cls':cls, 'name':name} | kwargs

        if name is None: 
            name = int(self.n_bias)
        else: 
            same_name = name in self.biasing_instances.keys()
            if same_name: 
                print('name of added bias (if provided) should be unique')
                assert not same_name

        instance = cls(sc = self.sc, **kwargs)
        '''
        instance : instance of WALLS or an instance of FLOOR
        is WALL **kwargs:
                inds_torsion : list,
                means : list,
                width_percentage : float = 68.17,
        if FLOOR **kwargs:

        '''
        n_forces_added = 0
        for force in instance.bias_related_forces:
            self.sc.system.addForce(force)
            n_forces_added += 1
        
        self.biasing_instances[name] = instance
        self.inds_biasing_forces[name] = (self.n_forces_total + np.arange(n_forces_added)).tolist()
        self.list_args_add_bias.append(args_add_bias)
        self.n_forces_total += n_forces_added
        self.n_bias += 1 
        self.sc.biased = True

        self.set_potential_energy_(define_current_CV_attribute=False)
        self.evalauted_bias = None

    def remove_all_bias_(self,):
        """Remove all group-15 forces and reset the bias registry.

        This mutates ``sc.system`` but does not rebuild an existing simulation
        context. The method asserts that the original physical-force count is
        restored.
        """
        
        print('! removing all bias. It can be instead easier to restart self.sc from the beginning')

        def index_bias_():
            """Return current system indices of all force-group-15 forces."""
            inds_bias = []
            for i, force in enumerate(self.sc.system.getForces()):
                if force.getForceGroup() == 15:
                    inds_bias.append(i)
            return inds_bias
        while len(index_bias_()) > 0:
            self.sc.system.removeForce(index_bias_()[0])
        assert len(self.sc.system.getForces()) == self.N_DEFAULT_FORCES

        all_groups = self.all_groups
        assert all([x!=15 for x in all_groups]) and len(all_groups) == self.N_DEFAULT_FORCES 

        self.biasing_instances = {}
        self.inds_biasing_forces = {}
        self.list_args_add_bias = []
        self.n_forces_total = int(self.N_DEFAULT_FORCES)
        self.n_bias = 0
        self.sc.biased = False
        self.evalauted_bias = None

        #self.set_potential_energy_(define_current_CV_attribute=False)

    '''
    def remove_bias_by_name_(self, name=None):
        print('not implemented, can use remove_all_bias_ or re-add again by starting from the beginning')
        
        if name is None:
            assert self.n_bias == 1
            name = 0
        else:
            print('since more than one bias was added, removing a bias requires a name (if no name given; an index)')
            print('these are the current biases involved:', self.biasing_instances)
            assert type(name) in [str, int]
        
        self.biasing_instances[name].remove_bias_() -> del self.biasing_instances[name]
        self.inds_biasing_forces -> remove correct parts in self.sc.system.getForces()
    '''

    def save_simulation_data_zero_bias_(self, path_and_name:str=None, dont_save_just_stat=False, eps=1e-10):
        """Select and optionally save frames on which the bias is effectively zero.

        Bias energies below ``eps`` (in reduced ``kT`` units) are classified as
        unbiased. The method reports retained fraction and mean physical
        energies, stores biased/unbiased indices and statistics on this object,
        and serializes a restart-style dataset unless ``dont_save_just_stat``
        is true. ``path_and_name`` is passed directly to ``save_pickle_``.
        """

        if self.evalauted_bias is None or len(self.evalauted_bias) != len(self.sc.u):
            # if self.P is None:
            if not self.sc.NPT: self.evalauted_bias = self.sc.potential_energy_(self.sc.xyz, which='v')
            else:               self.evalauted_bias = self.sc.potential_energy_(self.sc.xyz, b=self.sc.boxes, which='v')
        else: pass

        self.inds_unbiased = np.where(self.evalauted_bias < eps)[0]
        self.inds_biased = np.where(self.evalauted_bias >= eps)[0]

        self.av_u_all = np.mean(self.sc.u)
        self.av_u_unbiased = np.mean(self.sc.u[self.inds_unbiased])
        self.av_u_biased = np.mean(self.sc.u[self.inds_biased])
        percentage_kept = len(self.inds_unbiased)*100/len(self.evalauted_bias)
        print('average enregy of all data:     ',np.round(self.av_u_all,6),'/kT',"(on 100% of the data)")
        print('average enregy of unbiased data:',np.round(self.av_u_unbiased,6),'/kT',f"(on {np.round(percentage_kept, 5)}% of the data)")
        print('average enregy of biased data:  ',np.round(self.av_u_biased,6),'/kT',f"(on {np.round(100-percentage_kept, 5)}% of the data)")

        dataset = {'xyz':self.sc.xyz[self.inds_unbiased],
                    'COMs':self.sc.COMs[self.inds_unbiased],
                    'b':self.sc.boxes[self.inds_unbiased],
                    'u':self.sc.u[self.inds_unbiased],
                    'T':self.sc.temperature[self.inds_unbiased],
                    'rbv':[self.sc._current_r_, self.sc._current_b_, self.sc._current_v_],
                    'stride_save_frame':self.sc.stride_save_frame,
                    'percentage_kept_and_avus:':[percentage_kept, [self.av_u_all, self.av_u_unbiased, self.av_u_biased]],
                    }
        assert hasattr(self.sc, 'simulation')
        simulation_data = {'MD dataset':dataset,
                            'args_initialise_object': self.sc.args_initialise_object,
                            'args_initialise_system': self.sc.args_initialise_system,
                            'args_initialise_simulation': self.sc.args_initialise_simulation,
                            'list_args_add_bias': self.list_args_add_bias,
                            }
        if dont_save_just_stat: pass
        else: save_pickle_(simulation_data, path_and_name)

def _set_inds_torsion_for_bias_(self, inds_torsion):
    """Map one molecule's torsion indices to every molecule in the topology.

    Returns the four original within-molecule indices and an integer array of
    shape ``(n_mol, 4)`` in simulation topology order.
    """

    inds_torsion = np.array(inds_torsion).flatten()
    assert len(inds_torsion) == 4
    assert max(inds_torsion) < self.n_atoms_mol # in first molecule
    inds_torsion_input = np.array(inds_torsion)
    # inds_torsion_molecules : (n_mol, 4)
    print('inds_torsion provided:',inds_torsion)
    inds_torsion = [self.forward_atom_index_(i) for i in inds_torsion]
    print('inds_torsion in self.topology:', inds_torsion)
    inds_torsion = np.array(inds_torsion)
    inds_torsion_molecules = np.array([inds_torsion + i*self.n_atoms_mol for i in range(self.n_mol)])
    # inds_torsion_molecules : (n_mol, 4)
    return inds_torsion_input, inds_torsion_molecules

def _set_means_torsions_for_bias_(self, means):
    """Broadcast a repeating sequence of torsion centres across molecules.

    A scalar is repeated for every molecule. A sequence must have a length
    dividing ``n_mol`` and is tiled in molecule order. Returns shape
    ``(n_mol,)`` in radians.
    """

    if type(means) in [int, float]:
        means = float(means)
        means = np.array([means]*self.n_mol)
    else:
        means = np.array(means).flatten()
        assert self.n_mol % len(means) == 0
        means = np.concatenate([means]*self.n_mol, axis=0)[:self.n_mol]
    print(self.n_mol,'means:', means)

    return means # (n_mol,)

## ## ## ## 

class WALLS:
    '''Confine a torsion to metastable wells with smooth periodic walls.

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

    '''

    """
    def smooth_periodic_square_well_(self, phi, centre, width_percentage=70., height = 200.):
        ''' ok '''
        # https://www.desmos.com/calculator/olpfhe5azr
        phi = np.array(phi)
        w = np.pi*(100-width_percentage)/100
        assert w <= 1.0 # ~= assert width_percentage >= 68.17
        z = (np.mod((phi - centre - np.pi)/w + np.pi, 2*np.pi/w) - np.pi)**2
        return np.where(z<1.0, (height/0.36788)*np.exp(1/(z-1)), 0.0)
    """

    def smooth_periodic_square_well_v2_(self, phi, centre):
        """Evaluate the wall energy for angles ``phi`` around ``centre``.

        Angles are periodic in radians. The returned NumPy array is zero inside
        the permitted region and rises quadratically in the transformed
        distance to ``self.height`` at the periodic boundary.
        """
        # https://www.desmos.com/calculator/bmswl7qqo6
        phi = np.array(phi)
        d0 = np.abs(phi - centre + np.pi)
        d1 = np.minimum(d0, 2*np.pi-d0)
        rr = (np.pi*(100-self.width_percentage)/100)**2
        z = np.maximum(0, rr - d1**2) / rr
        return self.height * (z**2)

    def __init__(self,
                 sc,
                 inds_torsion : list,
                 means : list,
                 width_percentage : float = 68.17,
                 height = 200,
                 ):
        """Build one periodic wall torsion for every molecule.

        ``inds_torsion`` gives four within-molecule atom indices. ``means`` is
        one or a repeating sequence of desired angles in radians.
        ``width_percentage`` is the unbiased fraction of the full 2-pi domain,
        and ``height`` is the maximum wall energy in kJ/mol. The constructed
        group-15 ``CustomTorsionForce`` is exposed in ``bias_related_forces``
        but is not itself added to the system.
        """
        self.sc = sc
        self.inds_torsion, self.inds_torsion_molecules = _set_inds_torsion_for_bias_(self.sc, inds_torsion)
        self.means = _set_means_torsions_for_bias_(self.sc, means)
        self.width_percentage = width_percentage
        self.height = height
        
        # create the self.force to put into BIAS, and BIAS will then put it into self.sc.system
        
        force = mm.CustomTorsionForce('height*(z^2); z=max(0,rr-d1^2)/rr; d1=min(d0,2*pi-d0); d0=abs(theta-theta0+pi)')
        force.addGlobalParameter('pi', np.pi)
        force.addGlobalParameter('rr', (np.pi*(100-self.width_percentage)/100)**2)
        force.addGlobalParameter('height', self.height * unit.kilojoules_per_mole)
        force.addPerTorsionParameter('theta0')
        for i in range(self.sc.n_mol):
            force.addTorsion(*self.inds_torsion_molecules[i], [self.means[i]])
        force.setForceGroup(15)
        self.force = force

        self.bias_related_forces = [self.force]

    def check_plot_torsion_(self, r, plot=True, inds=None):
        """Calculate and optionally plot the biased torsion for every molecule.

        ``r`` is reshaped into molecules and the returned angles have shape
        ``(n_frames, n_mol)`` in radians. When plotting, ``inds`` optionally
        highlights selected frames in red and the wall profile is scaled to
        each histogram for visual comparison.
        """
        
        r = reshape_to_molecules_np_(r, n_atoms_in_molecule=self.sc.n_atoms_mol, n_molecules=self.sc.n_mol)
        phi = get_torsion_np_(r, self.inds_torsion)[...,0] # (m, n_mol)

        if plot:
            phi_grid = np.linspace(-np.pi, np.pi, 500)
            for i in range(self.sc.n_mol):
                fig = plt.figure(figsize=(2,1))
                phi_mol = phi[:,i]
                max_h = plot_1D_histogram_(phi_mol, range=[-np.pi,np.pi], color='black', return_max_y = True)
                plt.scatter(phi_mol,[-1]*len(phi_mol), s=1, color='black')
                if inds is not None: plt.scatter(phi_mol[inds],[-1]*len(phi_mol[inds]), s=1, color='red')
                else: pass
                # the self.force plotted: 
                bias_grid  = self.smooth_periodic_square_well_v2_(phi_grid, self.means[i])*max_h/self.height
                plt.plot(phi_grid, bias_grid, color='C0', linestyle='--')

                plt.show()
        else: pass

        return phi

## ## ## ## 

class FLOOR:
    """Raise selected torsional wells with an accelerated-MD-style bias.

    The bias is applied to matching ``PeriodicTorsionForce`` and
    ``RBTorsionForce`` terms. Its practical sampling effectiveness remains
    experimental and should be checked by reweighting diagnostics.
    """
    
    '''
    https://www.desmos.com/calculator/vkfvabggss
    '''

    def __init__(self,
                 sc,
                 inds_torsion : list,
                 percentage_raise : float = 50.0,
                 specific_AD = True,
                 ):
        """Construct floor forces for a selected molecular torsion.

        ``percentage_raise`` chooses the fraction of each torsion's energy
        range below which the floor acts. With ``specific_AD=True``, only exact
        A-B-C-D terms (in either direction) match; otherwise every term rotating
        about the selected B-C bond matches. Generated forces are assigned to
        group 15 and collected in ``bias_related_forces``.
        """
        self.sc = sc
        self.inds_torsion, self.inds_torsion_molecules = _set_inds_torsion_for_bias_(self.sc, inds_torsion)
        self.percentage_raise = percentage_raise
        self.specific_AD = specific_AD

        #############################################################

        ''' 
        letting A-B-C-D represent four neighbouring bonded atoms

        specific_AD = False -> all torsional energy terms in system related to rotation around B-C get biased
        specific_AD = True  -> torsional energy terms in system related specifically to A-B-C-D or D-C-B-A get biased
        '''
        # specific_AD
        ABCD_molecule = [x for x in self.inds_torsion_molecules]
        ABCD_molecule += [np.flip(x) for x in ABCD_molecule]
        ABCD_molecule = np.stack(ABCD_molecule, axis=0)
        # just to check these exact four indices from the FF are also in the list ABCD_molecule (direction does not matter)
        a1_a2_a3_a4_in_ABCD_molecule_ = lambda a1, a2, a3, a4 : np.abs(np.array([a1, a2, a3, a4])[np.newaxis,:] - ABCD_molecule).sum(1).min() == 0 
        
        # not specific_AD 
        BC_molecule = [set(x[1:3]) for x in self.inds_torsion_molecules]

        def ADD_(a1, a2, a3, a4) -> bool:
            """Return whether one force-field torsion matches the selection."""
            if set([a2,a3]) in BC_molecule:
                if self.specific_AD:
                    if a1_a2_a3_a4_in_ABCD_molecule_(a1, a2, a3, a4):
                        return True
                    else: 
                        return False
                else: 
                    return True
            else:
                return False

        #############################################################

        alpha = 1.0
        bias_expression = 'step(d) * d * d / (alpha + d) ; d = E - U '

        n_added_1 = 0
        n_added_2 = 0
        self.bias_related_forces = []
        for force in self.sc.system.getForces():

            if isinstance(force, mm.PeriodicTorsionForce):

                bias_1 = mm.CustomTorsionForce(bias_expression + '; U = k * (1 + cos(n * theta - theta0))')
                bias_1.addGlobalParameter("alpha", alpha)

                bias_1.addPerTorsionParameter("k")
                bias_1.addPerTorsionParameter("n")
                bias_1.addPerTorsionParameter("theta0")

                bias_1.addPerTorsionParameter("E")

                for i in range(force.getNumTorsions()):
                    a1, a2, a3, a4, n, theta0, k = force.getTorsionParameters(i)

                    if ADD_(a1, a2, a3, a4):
                        E = 2.0*(self.percentage_raise/100.0) * k # kJ/mol
                        bias_1.addTorsion(a1, a2, a3, a4, [k, n, theta0, E])
                        n_added_1 += 1
                    else: pass

                bias_1.setForceGroup(15)
                self.bias_1 = bias_1

                if n_added_1 > 0:
                    self.bias_related_forces.append(self.bias_1)
                    print('FLOOR : number of 1D biases added (per molecule) in relation to PeriodicTorsionForce:', n_added_1/self.sc.n_mol )
                else:
                    print('FLOOR : PeriodicTorsionForce:', 'no bias was added' )
                            
            else: pass

            if isinstance(force, mm.RBTorsionForce):

                def find_min_max_RBTorsionForce_(c0,c1,c2,c3,c4,c5):
                    """Estimate an RB torsion's extrema on a dense angle grid."""
                    ct = - np.cos(np.linspace(-np.pi, np.pi, 2000))
                    U = c0 + c1*ct + c2*(ct**2) + c3*(ct**3) + c4*(ct**4) + c5*(ct**5)
                    return np.min(U), np.max(U)
                
                bias_2 = mm.CustomTorsionForce(bias_expression + '; U = c0 + c1*ct + c2*(ct^2) + c3*(ct^3) + c4*(ct^4) + c5*(ct^5) ; ct = - cos(theta)')
                bias_2.addGlobalParameter("alpha", alpha)

                bias_2.addPerTorsionParameter("c0")
                bias_2.addPerTorsionParameter("c1")
                bias_2.addPerTorsionParameter("c2")
                bias_2.addPerTorsionParameter("c3")
                bias_2.addPerTorsionParameter("c4")
                bias_2.addPerTorsionParameter("c5")

                bias_2.addPerTorsionParameter("E")

                for i in range(force.getNumTorsions()):
                    a1, a2, a3, a4, c0, c1, c2, c3, c4, c5 = force.getTorsionParameters(i)

                    if ADD_(a1, a2, a3, a4):
                        Min, Max = find_min_max_RBTorsionForce_(c0, c1, c2, c3, c4, c5)
                        E = (self.percentage_raise/100.0)*(Max - Min) + Min
                        bias_2.addTorsion(a1, a2, a3, a4, [c0, c1, c2, c3, c4, c5, E])
                        n_added_2 += 1
                    else:
                        pass

                bias_2.setForceGroup(15)
                self.bias_2 = bias_2

                if n_added_2 > 0:
                    self.bias_related_forces.append(self.bias_2)
                    print('FLOOR : number of 1D biases added (per molecule) in relation to RBTorsionForce:', n_added_2/self.sc.n_mol )
                else:
                    print('FLOOR : RBTorsionForce:', 'no bias was added' )

            else: pass

        if n_added_2 == 0 and n_added_1 == 0:
            print('!! FLOOR : NO BIAS added. Double check why this happened.')
        else: pass
        print('')

    def check_plot_torsion_(self, r, plot=True, weights=None):
        """Calculate and optionally plot raw and reweighted torsion histograms.

        Returns angles with shape ``(n_frames, n_mol)`` in radians. ``weights``
        must provide one value per frame and is used for the black reweighted
        histogram; the unweighted distribution is shown in red.
        """
        
        r = reshape_to_molecules_np_(r, n_atoms_in_molecule=self.sc.n_atoms_mol, n_molecules=self.sc.n_mol)
        phi = get_torsion_np_(r, self.inds_torsion)[...,0] # (m, n_mol)

        ws = np.array(weights).flatten()
        if plot:
            phi_grid = np.linspace(-np.pi, np.pi, 500)
            for i in range(self.sc.n_mol):
                fig = plt.figure(figsize=(2,1))
                phi_mol = phi[:,i]
                plot_1D_histogram_(phi_mol, range=[-np.pi,np.pi], color='red')
                plt.scatter(phi_mol,[-1]*len(phi_mol), s=1, color='black')
                plot_1D_histogram_(phi_mol, range=[-np.pi,np.pi], color='black', kwargs_for_histogram={'weights':ws})

                # the self.force plotted: 
                plt.show()
        else: pass

        return phi
    

