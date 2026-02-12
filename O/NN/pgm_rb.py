from ..interface import *

''' 

REF: NPT-flow; 10.1088/2632-2153/acefa8

## ## ## ## 

PGMcrys_v1_rb

    *{This version of the model is still experimental.}*

    This is the same as PGMcrys_v1 but has 6 additional box DOFs concatenated to the end of x^{P} array.
    
    * {1st} reason why this is experimental: the Gibbs FE estimates here were not yet compared to ground truth

NB:
    numerical stability of inversion (jacobean) not as good as PGMcrys_v1, but this does not seem to be helped by setting use_tfp = True, although it is likely coming from the edges of the domain 
    *This point is the {2nd} reason why this is experimental
    numerical stability in inversion (positions) looks bad in min/max, the outliers are because of PBCs wrapping/unwrapping for rO positions and torsions
        rO : molecules in the samples from the model are jumping PBCs; this is expected.

summary of differences compared to pgm.py + interface.py (the NVT methods):

    rb = concant([r,b], axis=-2) is the (N+3,3) array, a 3(N+1) dimensional microstate (in terms of DOFs), 
        where r (N,3) is positions and b (3,3) is the box array (with 6 DOFs; the lower-triangular elements)
        rb is carrier around as one array as much as possible for convenience

    PBC from rO atoms is not removed; these atoms can jump between the periodic images in the data, in contrast to PGMcrys_v1 where data must not contain any PBC effects (removed, or absent, a-priori)
        COM is therefore removed by fixing one rO atom (in the arbitrary first molecule of the supercell)
        once the atom is removed the 1D marginals of other rO atoms (DOFs) are treated like stiff torsional angles (unwrapped around 0 and treated as non-periodic variables \in [-1,1])
        once non-periodic, they can be whitened, either with the box (setting 2), or excluding the box (setting 1), or no whitening at all (setting 0)
            whitening is optional because dimensionality is not changed in any of these three settings
            whitening included as option only to slightly help the model learn (the remaining non-linear correlations)

    in the interface obj below, notations for all u and r arrays are not the same as in the rest of FEcrys:
        u := enthalpy (isotropic pressure applied on an anisotropic box, with the correct log Jacobean of the box from the literature [DOI: 10.3389/fchem.2021.718920, 10.1088/2632-2153/acefa8])
            the Jacobean term takes into account two maps (1 DOF; scalar volume) -> (9 DOF; unconstrained box) -> (6 DOF; rotationally fixed (lower-triangular) box)
        r := rb # use rb_to_r_b_ to separate them if visualising samples from the model

NB:
    The 6 DOFs of the box are effectively on R^6, with a large energy barrier keeping the diagonal positive in MD data, however, 
        in some of the crystals, it was observed that openMM flexible barostat during an NPT trajectory allows one of the lower off-diagonal elements of the box 
        to 'jump'/'mirror' instantaneously between negatives and positives values (of similar absolute length) without a clear path in between. 
        These jumps have no visible effects on the atoms; molecules are definitely not mirrored 
            likely because openMM flexible barostat uses COM from each molecule by default for the trial moves
        This is likely not a discontinuity in this box space, but the outcome of the metropolis sampling, in combination with certain crystallographic features of the packing
            There seems to be no trivial 2D mirror plane(s) to map all the box elements from such NPT trajectory to one sign or the other, without affecting the energy.
            Sampled boxes of this kind, when left unprocessed, leaves a 'gap'/discontinuity with no MD box-samples between the two states.
                We can assume that both metastable states are sampled ergodically, in a trajectory that contains sufficient training data, and try using this data anyway for training.
                However, PGMcrys_v1_rb was not yet tested on such a dataset.
                Caution: PGMcrys_v1_rb may struggle arbitrarily with such data, because the space in between would remain in the scope of the model, and will receive high-energy samples.
                * {3rd} reason why this is experimental.

'''
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def rb_to_r_b_(rb):
    # reshape as atoms.
    r = rb[:,:-3]
    b = rb[:,-3:]
    return r, b

def r_b_to_rb_tf_(r,b):
    return tf.concat([r,b], axis=1)

def r_b_to_rb_np_(r,b):
    return np.concatenate([r,b], axis=1)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def box_forward_tf_(b):
    ' placeholder '
    ladJ = 0.0 # (m,1)
    h = tf.stack([b[...,0,0],b[...,1,1],b[...,2,2],b[...,1,0],b[...,2,0],b[...,2,1]],axis=-1)
    return h, ladJ

def box_inverse_tf_(h):
    ' placeholder '
    ladJ = 0.0 # (m,1)
    zero = tf.zeros_like(h[...,0])
    b = tf.stack([tf.stack([ h[...,0], zero    , zero    ],axis=-1),
                  tf.stack([ h[...,3], h[...,1], zero    ],axis=-1),
                  tf.stack([ h[...,4], h[...,5], h[...,2]],axis=-1),], axis=-2)
    return b, ladJ

def box_forward_np_(b):
    ' placeholder '
    ladJ = 0.0 # (m,1)
    h = np.stack([b[...,0,0],b[...,1,1],b[...,2,2],b[...,1,0],b[...,2,0],b[...,2,1]],axis=-1)
    return h, ladJ

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class SingleComponent_map_rb(SingleComponent_map):
    ''' !! : molecule must have >3 atoms to use this M_{IC} layer '''
    def __init__(self,
                 PDB_single_mol: str,   
                ):
        super().__init__(PDB_single_mol)
        self.VERSION = 'NEW'
        
    ## ## ## ##

    def remove_COM_from_data_(self, rb):
        '''
        important step : fixing the first rO atoms (i.e., rO atom in the first molecule to zero)
        other rO atoms shifted into the [0,1) box, taking the molecules with them.
        Only the rO atoms are under PBCs, because molecules are whole and not transformed here in any way.
        '''
        # rb : (m,N+3,3)
        # r  : (m, N, 3)
        # b  : (m, 3, 3)

        r, b = rb_to_r_b_(rb)
        r = np.array(r).astype(np.float64)
        b = np.array(b).astype(np.float64)

        self._first_times_crystal_(r.shape)

        r = reshape_to_molecules_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        # rO : (m. n_mol, n_atoms_mol, 3)

        ind_rO = self.inds_atoms_CB[0]
        rO = r[...,ind_rO:ind_rO+1,:]
        # rO : (m. n_mol, 1, 3)

        print('removing COM from data by fixing one atom:')

        rO_shifted = rO - rO[:,:1]
        s = np.einsum('omni,oij->omnj', rO_shifted, np.linalg.inv(b))
        s = np.mod(s, 1.0)
        rO_shifted = np.einsum('omni,oij->omnj', s, b)

        r = np.array(r-rO+rO_shifted).astype(np.float32)

        r = reshape_to_atoms_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        # r : (m, N, 3)

        b = np.array(b).astype(np.float32)

        rb = r_b_to_rb_np_(r,b)

        return rb

    def initalise_(self,
                   rb_dataset,
                   batch_size=10000,
                   n_mol_unitcell = 1,
                   COM_remover = 'blank', # fixed atom method because whole molecules in openmm NPT jump between images
                   focused = 'blank',     # no point comparing. Use symmetry reduction for molecules like water, benzene, etc.
                   whiten_setting = 0,    # 0 : no whitening, 1 : whitening only positions, 2 : whitening positions and box
                   # 0 or 2 better than 1, 2 slightly better overall. [0,2 treat all 1D DOFs of the x^{P} array in the same way]
                   ):
        self.focused = True
        assert len(rb_dataset.shape) == 3

        r_dataset, b_dataset = rb_to_r_b_(rb_dataset)

        # r_dataset : (m,N,3)
        # b_dataset : (m,3,3)

        assert b_dataset.shape == (r_dataset.shape[0], 3,3)
        self.single_box_in_dataset = False # !

        X_IC, X_CB = self._forward_init_(r_dataset, batch_size=batch_size)[:2]
        rO, q, a, d0, d1 = X_CB

        assert not self.n_mol_supercell % n_mol_unitcell, 'ic_map.initialise_ : please check n_mol_unitcell is a factor of n_mol (number of molecules in r_dataset)'
        self.n_mol_unitcell = n_mol_unitcell
        self.n_unitcells = self.n_mol_supercell // self.n_mol_unitcell

        sO = np.einsum('omi,oij->omj', rO, np.linalg.inv(b_dataset))
        # sO : (m, n_mol, 3)

        sO_flat = reshape_to_flat_np_(sO,  n_molecules=self.n_mol, n_atoms_in_molecule=1)
        self.FA = NotWhitenFlow(sO_flat, whiten_anyway=False)
        sO_flat = self.FA._forward_np(sO_flat)

        # adding safety step just in case data translated (COM not removed during initialisation)
        sO_flat = np.mod(sO_flat, 1.0) # fixed atom at 0 and missing, can do this.

        self.ranges_sO = np2tf_(np.ones(3*(self.n_mol-1)))      # 1
        self.centres_sO = np2tf_(np.ones(3*(self.n_mol-1))*0.5) # 0.5

        sO_flat = scale_shift_x_general_(np2tf_(sO_flat),
                                         physical_ranges_x = self.ranges_sO, 
                                         physical_centres_x = self.centres_sO,
                                         model_range = 2.0*PI, model_centre = 0.0,
                                         forward = True)[0].numpy()
        self.Focused_Positions = FocusedTorsions(sO_flat, axis=[0], focused=self.focused)
        assert self.Focused_Positions.mask_periodic.max() < 0.5, '! at least 1 molecule may have tanslated > 80% along the box; data may be non-ergodic; a solution (in sym.py) not yet implemented.'

        h = box_forward_np_(b_dataset)[0]
        self.ranges_h, self.centres_h = get_ranges_centres_(h, axis=[0]) # (6,)

        ## ## ## ##
        self.whiten_setting = whiten_setting
        print(f'whiten_setting {whiten_setting} \n')
        if self.whiten_setting == 0: 
            self.white_ = self.white_setting_0_
        else:
            xO = self.Focused_Positions(np2tf_(sO_flat), forward=True)[0].numpy()
            if self.whiten_setting == 1:
                self.WF_keepdims = WhitenFlow(xO, removedims=0)
                xO = self.WF_keepdims._forward_np(xO) # (m, 3*n_mol-3) -> (m, 3*n_mol-3)
                self.ranges_xO, self.centres_xO = get_ranges_centres_(xO, axis=[0])
                self.white_ = self.white_setting_1_
            else:
                h = scale_shift_x_(np2tf_(h), physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = True)[0].numpy()
                xOh = np.concatenate([xO, h], axis=-1) 
                self.WF_keepdims = WhitenFlow(xOh, removedims=0)
                xOh = self.WF_keepdims._forward_np(xOh) # (m, 3*n_mol+3) -> (m, 3*n_mol+3)
                self.ranges_xOh, self.centres_xOh = get_ranges_centres_(xOh, axis=[0])
                self.white_ = self.white_setting_2_
        ## ## ## ## 

        ## ## ## ## 

        # X_IC : (m, n_mol, n_atoms_IC, 3)
        # q    : (m, n_mol, 4)
        # a    : (m, n_mol, 1)
        # d0   : (m, n_mol, 1)
        # d1   : (m, n_mol, 1)

        bonds = X_IC[...,0]  ; bonds = np.concatenate([d0,d1,bonds],axis=-1)
        angles = X_IC[...,1] ; angles = np.concatenate([a,angles],axis=-1)
        torsions = X_IC[...,2]

        # bonds    : (m, n_mol, n_atoms_IC + 2)
        # angles   : (m, n_mol, n_atoms_IC + 1)
        # torsions : (m, n_mol, n_atoms_IC    )
        # q        : (m, n_mol, 4)
        
        bonds    = self.reshape_to_unitcells_(bonds   , combined_unitcells_with_batch_axis=True, forward=True, numpy = True) 
        angles   = self.reshape_to_unitcells_(angles  , combined_unitcells_with_batch_axis=True, forward=True, numpy = True) 
        torsions = self.reshape_to_unitcells_(torsions, combined_unitcells_with_batch_axis=True, forward=True, numpy = True) 
        q        = self.reshape_to_unitcells_(q       , combined_unitcells_with_batch_axis=True, forward=True, numpy = True)

        # bonds    : (m*n_unitcells, n_mol_unitcell, n_atoms_IC + 2)
        # angles   : (m*n_unitcells, n_mol_unitcell, n_atoms_IC + 1)
        # torsions : (m*n_unitcells, n_mol_unitcell, n_atoms_IC    )
        # q        : (m*n_unitcells, n_mol_unitcell, 4)

        # setting axis = [0,1] recovers old behavior
        self.Focused_Bonds = FocusedBonds(bonds,          axis=[0], focused=self.focused)
        self.Focused_Angles = FocusedAngles(angles,       axis=[0], focused=self.focused)
        self.Focused_Torsions = FocusedTorsions(torsions, axis=[0], focused=self.focused)
        self.Focused_Hemisphere = FocusedHemisphere(q,              focused=True        )

        self.n_DOF_mol = self.n_atoms_IC*3 + 3 + 3
        self.ln_base_C = - np2tf_( self.n_mol_supercell*self.n_DOF_mol*np.log(2.0) )
        self.ln_base_P = - np2tf_( 3*(self.n_mol_supercell-1)*np.log(2.0) + 6*np.log(2.0) )
        
        self.set_periodic_mask_()

    def sample_base_P_(self, m):
        # p_{0} (base distribution) positions, and box:
        return tf.clip_by_value(tf.random.uniform(shape=[m, 3*(self.n_mol_supercell-1) + 6], minval=-1.0,  maxval=1.0), -1.0, 1.0)
    
    def white_setting_0_(self, inputs, forward=True):
        ladj = 0.0
        if forward:
            xO, h = inputs
            ##
            h, ladJ_scale_h = scale_shift_x_(h, physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = True)
            ladj += ladJ_scale_h
            ##
            return tf.concat([xO, h], axis=-1), ladj
        else:
            xOh = inputs[0]
            xO = xOh[...,:-6]
            h  = xOh[...,-6:]
            ##
            h, ladJ_scale_h = scale_shift_x_(h, physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = False)
            ladj += ladJ_scale_h
            ##
            return xO, h, ladj

    def white_setting_1_(self, inputs, forward=True):
        ladj = 0.0
        if forward:
            xO, h = inputs
            ##
            h, ladJ_scale_h = scale_shift_x_(h, physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = True)
            ladj += ladJ_scale_h
            ##
            xO, ladJ_whiten = self.WF_keepdims.forward(xO)
            ladj += ladJ_whiten
            xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = True)
            ladj += ladJ_scale_xO
            return tf.concat([xO, h], axis=-1), ladj
        else:
            xOh = inputs[0]
            xO = xOh[...,:-6]
            h  = xOh[...,-6:]
            ##
            xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = False)
            ladj += ladJ_scale_xO
            xO, ladJ_whiten = self.WF_keepdims.inverse(xO)
            ladj += ladJ_whiten
            ##
            h, ladJ_scale_h = scale_shift_x_(h, physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = False)
            ladj += ladJ_scale_h
            ##
            return xO, h, ladj

    def white_setting_2_(self, inputs, forward=True):
        ladj = 0.0
        if forward:
            xO, h = inputs
            ##
            h, ladJ_scale_h = scale_shift_x_(h, physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = True)
            ladj += ladJ_scale_h
            ##
            xOh = tf.concat([xO, h], axis=-1) 
            xOh, ladJ_whiten = self.WF_keepdims.forward(xOh)
            ladj += ladJ_whiten
            xOh, ladJ_scale_xOh = scale_shift_x_(xOh, physical_ranges_x = self.ranges_xOh, physical_centres_x = self.centres_xOh, forward = True)
            ladj += ladJ_scale_xOh
            return xOh, ladj
        else:
            xOh = inputs[0]
            xOh, ladJ_scale_xOh = scale_shift_x_(xOh, physical_ranges_x = self.ranges_xOh, physical_centres_x = self.centres_xOh, forward = False)
            ladj += ladJ_scale_xOh
            xOh, ladJ_whiten = self.WF_keepdims.inverse(xOh)
            ladj += ladJ_whiten
            xO = xOh[...,:-6]
            h  = xOh[...,-6:]
            ##
            h, ladJ_scale_h = scale_shift_x_(h, physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = False)
            ladj += ladJ_scale_h
            ##
            return xO, h, ladj

    ##
    def forward_rb_(self, rb):

        # rb : (m,N+3,3)

        r, b = rb_to_r_b_(rb)

        # r  : (m, N, 3)
        # b  : (m, 3, 3)
       
        outputs, ladJ = self.forward_(r)
        rO, X = outputs

        #### EXTERNAL : ####

        sO = tf.einsum('omi,oij->omj', rO, tf.linalg.inv(b))
        V = b[...,0,0] * b[...,1,1] * b[...,2,2]
        ladJ_unit_box_forward = - tf.math.log(V)*(self.n_mol-1.0)
        ladJ += ladJ_unit_box_forward[...,tf.newaxis]

        sO_flat = self.FA.forward(reshape_to_flat_tf_(sO, n_molecules=self.n_mol, n_atoms_in_molecule=1))[0]

        # adding safety step just in case data translated (COM not removed during initialisation)
        sO_flat = tf.math.floormod(sO_flat, 1.0) # fixed atom at 0 and missing, can do this.

        sO_flat, ladJ_scale_sO = scale_shift_x_general_(sO_flat,
                                                        physical_ranges_x = self.ranges_sO, 
                                                        physical_centres_x = self.centres_sO,
                                                        model_range = 2.0*PI, model_centre = 0.0,
                                                        forward = True)
        ladJ += ladJ_scale_sO
        
        xO, ladJ_scale_positions = self.Focused_Positions(sO_flat, forward=True)
        ladJ += ladJ_scale_positions

        h = box_forward_tf_(b)[0]

        xOh, ladJ_white = self.white_([xO,h], forward=True)
        ladJ += ladJ_white
        '''
        h, ladJ_scale_h = scale_shift_x_(h, physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = True)
        ladJ += ladJ_scale_h

        # xO : (m, 3*n_mol-3)

        ## ## ## ## ## ## ## ## ## 
        if self.whiten_anyway:
            xO, ladJ_whiten = self.WF_keepdims.forward(xO)
            ladJ += ladJ_whiten 
            xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = True)
            ladJ += ladJ_scale_xO
        else: pass
        ## ## ## ## ## ## ## ## ## 

        xOh = tf.concat([xO, h], axis=-1) 
        '''

        #### 

        variables = [xOh, X]
        return variables, ladJ
    
    def inverse_rb_(self, variables_in):

        ladJ = 0.0

        xOh, X = variables_in

        '''
        xO = xOh[...,:-6]
        h  = xOh[...,-6:]

        ## ## ## ## ## ## ## ## ## 
        if self.whiten_anyway:
            xO, ladJ_scale_xO = scale_shift_x_(xO, physical_ranges_x = self.ranges_xO, physical_centres_x = self.centres_xO, forward = False)
            ladJ += ladJ_scale_xO
            xO, ladJ_whiten = self.WF_keepdims.inverse(xO)
            ladJ += ladJ_whiten 
        else: pass
        ## ## ## ## ## ## ## ## ## 

        h, ladJ_scale_h = scale_shift_x_(h, physical_ranges_x = self.ranges_h, physical_centres_x = self.centres_h, forward = False)
        ladJ += ladJ_scale_h
        '''
        xO, h, ladJ_white = self.white_([xOh], forward=False)
        ladJ += ladJ_white

        b = box_inverse_tf_(h)[0]

        sO_flat, ladJ_scale_positions = self.Focused_Positions(xO, forward=False)
        ladJ += ladJ_scale_positions

        sO_flat, ladJ_scale_sO = scale_shift_x_general_(sO_flat,
                                                        physical_ranges_x = self.ranges_sO, 
                                                        physical_centres_x = self.centres_sO,
                                                        model_range = 2.0*PI, model_centre = 0.0,
                                                        forward = False)
        ladJ += ladJ_scale_sO

        # adding safety step just in case data translated (COM not removed during initialisation)
        sO_flat = tf.math.floormod(sO_flat, 1.0) # fixed atom at 0 and missing, can do this.

        sO = reshape_to_atoms_tf_(self.FA.inverse(sO_flat)[0], n_molecules=self.n_mol, n_atoms_in_molecule=1)

        rO = tf.einsum('omi,oij->omj', sO, b)
        V = b[...,0,0] * b[...,1,1] * b[...,2,2]
        ladJ_unit_box_inverse = tf.math.log(V)*(self.n_mol-1.0)
        ladJ += ladJ_unit_box_inverse[...,tf.newaxis]

        r, ladj = self.inverse_([rO, X])
        ladJ += ladj

        # r  : (m, N, 3)
        # b  : (m, 3, 3)
        # rb : (m,N+3,3)
        return r_b_to_rb_tf_(r, b), ladJ

    def xO_reshape_(self, x, forward=True):
        print('!! not here')
        return None

    @property
    def flow_mask_xO(self,):
        print('!! not here')
        return None

    @property
    def flow_mask_X(self,):
        # all intramolecular DOFs are relevant (conformations and rotations)
        # (1, n_mol, n_DOF_mol) of ones
        mask = np.ones([1, self.n_mol, self.n_DOF_mol]).astype(np.int32)
        return mask # (1, n_mol, n_DOF_mol) # can self.reshape_to_unitcells_ if needed
    
####################################################################################################

class PGMcrys_v1_rb(tf.keras.models.Model, model_helper_PGMcrys_v1, model_helper):
    ''' !! : molecule should have >3 atoms (also true in ic_map) '''
    @staticmethod
    def load_model(path_and_name : str, VERSION='blank'):
        return PGMcrys_v1_rb._load_model_(path_and_name, PGMcrys_v1_rb)

    def __init__(self,
                 ic_maps : list,
                 n_layers : int = 4,
                 optimiser_LR_decay = [0.001,0.0],
                 DIM_connection = 10,
                 n_att_heads = 4,
                 initialise = True, # for debugging in eager mode
                 ):
        super().__init__()
        self.init_args = {  'ic_maps' : ic_maps,
                            'n_layers' : n_layers,
                            'optimiser_LR_decay' : optimiser_LR_decay,
                            'DIM_connection' : DIM_connection,
                            'n_att_heads' : n_att_heads}
        
        ####
        if str(type(ic_maps)) not in ["<class 'list'>","<class 'tensorflow.python.training.tracking.data_structures.ListWrapper'>"]: 
            ic_maps = [ic_maps]
        else: pass
        for ic_map in ic_maps:
            assert isinstance(ic_map, SingleComponent_map_rb)
            if hasattr(ic_map, 'single_box_in_dataset'): assert ic_map.single_box_in_dataset == False
            else: ic_map.single_box_in_dataset = False
        
        self.ic_maps = ic_maps
        self.n_mol = int(self.ic_maps[0].n_mol)
        assert all([ic_map.n_mol == self.n_mol for ic_map  in self.ic_maps])
        assert all([ic_map.n_atoms_mol == ic_maps[0].n_atoms_mol for ic_map in self.ic_maps])
        self.n_maps = len(self.ic_maps)

        if self.n_maps > 1:
            print('matching ic_maps for the single model:')
            [ic_map.match_topology_(ic_maps) for ic_map in self.ic_maps]
            print('checking that ic_maps match the model:')
            assert all(np.abs(ic_map.periodic_mask - ic_maps[0].periodic_mask).sum()==0 for ic_map in self.ic_maps)
        else: pass

        self.periodic_mask = np.array(self.ic_maps[0].periodic_mask)
        assert all([np.abs(self.periodic_mask - ic_map.periodic_mask).sum() == 0 for ic_map in self.ic_maps])
        # preparing how psi_{C->P} and psi_{P->C} are extended along last axis with crystal encoding:

        number_of_parallel_molecules = self.n_mol
                     
        if self.n_maps > 1:
            self.dim_crystal_encoding = 1
            self.crystal_encodings = np2tf_(np.linspace(-1.,1.,self.n_maps))
            self.C2P_extension_shape = np2tf_(np.zeros([1,1]))
            self.P2C_extension_shape = np2tf_(np.zeros([self.n_mol, 1]))
        else:
            self.dim_crystal_encoding = 0
            self.crystal_encodings = np2tf_(np.array([0]))
            self.C2P_extension_shape = np2tf_(np.zeros([1,0]))
            self.P2C_extension_shape = np2tf_(np.zeros([self.n_mol, 0]))

        ####

        self.n_layers = n_layers
        self.optimiser_LR_decay = optimiser_LR_decay
        self.DIM_connection = DIM_connection
        self.n_att_heads = n_att_heads

        ##
        self.DIM_P2C_connection = self.DIM_connection  
        self.DIM_C2P_connection = self.n_mol*self.DIM_connection
        self.DIM_C2C_connection = None

        n_hidden_main = 2
        n_hidden_connection = 1
        hidden_activation = tf.nn.leaky_relu #tanh
        n_bins = 5
        print(f'n_att_heads = {self.n_att_heads}, n_layers = {self.n_layers}')

        self.layers_P = [ POSITIONS_FLOW_LAYER(
                            n_mol = self.n_mol + 2, # box DOFs at the end of xO array (xOh is the new name)
                            layer_index = i,
                            DIM_P2C_connection = self.DIM_P2C_connection,
                            DIM_C2P_connection = self.DIM_C2P_connection + self.dim_crystal_encoding,
                            name = 'POSITIONS_FLOW_LAYER',
                            n_hidden_main = n_hidden_main,
                            n_hidden_connection = n_hidden_connection,
                            hidden_activation = hidden_activation,
                            use_tfp = False,
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            n_P2C = self.n_mol,
                        ) for i in range(self.n_layers)]

        if self.n_att_heads in [None, 0]:
            half_layer_class = SPLINE_COUPLING_HALF_LAYER
            kwargs_for_given_half_layer_class = {
                                                'n_hidden' : 2,
                                                'dims_hidden' : None,
                                                'hidden_activation' : hidden_activation,
                                                }
        else:
            half_layer_class = SPLINE_COUPLING_HALF_LAYER_AT
            kwargs_for_given_half_layer_class = {
                                                'flow_mask' : None,
                                                'n_mol' : self.n_mol,
                                                'n_heads' : self.n_att_heads, # 4
                                                'embedding_dim' : self.DIM_connection,
                                                'n_hidden_kqv' : [2,2,2],
                                                'hidden_activation' : hidden_activation,
                                                'one_hot_kqv' : [True]*3,
                                                'n_hidden_decode' : 1,
                                                #'new' : False,
                                                }

        self.layers_C = [ CONFORMER_FLOW_LAYER(
                            periodic_mask = self.periodic_mask,
                            layer_index = i,
                            DIM_P2C_connection = self.DIM_P2C_connection +  self.dim_crystal_encoding,
                            n_mol = self.n_mol,
                            DIM_C2P_connection = self.DIM_C2P_connection,
                            n_hidden_connection = n_hidden_connection,
                            half_layer_class = half_layer_class,
                            kwargs_for_given_half_layer_class = kwargs_for_given_half_layer_class,
                            use_tfp = False,
                            n_bins = n_bins,
                            min_bin_width = 0.001,
                            knot_slope_range = [0.001, 50.0],
                            name = 'CONFORMER_FLOW_LAYER',
                        ) for i in range(self.n_layers)]

        ## p_{0}:
        self.ln_base_ = self.ic_maps[0].ln_base_
        self.sample_base_ = self.ic_maps[0].sample_base_
        
        ## trainability:
        self.all_parameters_trainable = True
        if initialise: self.initialise()
        else: pass

    def get_C2P_P2C_extensions_(self, m, crystal_index):
        # same as v1.

        number = self.crystal_encodings[crystal_index]

        C2P_extension = self.C2P_extension_shape + number   # (1, 1)
        C2P_extension = tf.stack([C2P_extension]*m, axis=0) # (m, 1, 1)

        P2C_extension = self.P2C_extension_shape + number   # (n_mol, 1)
        P2C_extension = tf.stack([P2C_extension]*m, axis=0) # (m, n_mol, 1)

        return [C2P_extension, P2C_extension] # crystal embeddings, zero dimensional if training on just 1 state

    ##

    def _forward_coupling_(self, X, crystal_index=0):
        ladJ = 0.0
        x_P, X_C = X
        X_P = self.layers_P[0].convert_to_flow_(x_P)
        C2P_extension, P2C_extension = self.get_C2P_P2C_extensions_(m=x_P.shape[0], crystal_index=crystal_index)

        for i in range(self.n_layers):

            aux_C2P   = self.layers_C[i].convert_to_aux_(X_C)
            aux_C2P   = tf.concat([aux_C2P, C2P_extension], axis=-1)
            X_P, ladj = self.layers_P[i].forward_(X_P, aux=aux_C2P) ; ladJ += ladj

            aux_P2C   = self.layers_P[i].convert_to_aux_(X_P)
            aux_P2C   = tf.concat([aux_P2C, P2C_extension], axis=-1)
            X_C, ladj = self.layers_C[i].forward_(X_C, aux=aux_P2C) ; ladJ += ladj

        x_P = self.layers_P[-1].convert_from_flow_(X_P)
        Z = [x_P, X_C]
        return Z, ladJ
    
    def _inverse_coupling_(self, Z, crystal_index=0):
        ladJ = 0.0
        x_P, X_C = Z
        X_P = self.layers_P[-1].convert_to_flow_(x_P)
        C2P_extension, P2C_extension = self.get_C2P_P2C_extensions_(m=x_P.shape[0], crystal_index=crystal_index)

        for i in reversed(range(self.n_layers)):

            aux_P2C   = self.layers_P[i].convert_to_aux_(X_P)
            aux_P2C   = tf.concat([aux_P2C, P2C_extension], axis=-1)
            X_C, ladj = self.layers_C[i].inverse_(X_C, aux=aux_P2C) ; ladJ += ladj

            aux_C2P   = self.layers_C[i].convert_to_aux_(X_C)
            aux_C2P   = tf.concat([aux_C2P, C2P_extension], axis=-1)
            X_P, ladj = self.layers_P[i].inverse_(X_P, aux=aux_C2P) ; ladJ += ladj

        x_P = self.layers_P[-1].convert_from_flow_(X_P)
        X = [x_P, X_C]
        return X, ladJ

    ##

    def _forward_represenation_(self, rb, crystal_index=0):
        ladJ = 0.0
        X, ladj_rep = self.ic_maps[crystal_index].forward_rb_(rb)
        ladJ += ladj_rep
        return X, ladJ
    
    def _inverse_represenation_(self, X, crystal_index=0):
        ladJ = 0.0
        rb, ladj_rep = self.ic_maps[crystal_index].inverse_rb_(X)
        ladJ += ladj_rep
        return rb, ladJ

    ##

####################################################################################################

class NN_interface_sc_rb(NN_interface_helper):
    def __init__(self,
                 name : str,
                 path_dataset : str,
                 fraction_training : float = 0.8, # 0 << x < 1
                 training : bool = True,
                 ):
        super().__init__()
        self.name = name + '_SC_' # ...
        self.path_dataset = path_dataset 
        self.fraction_training = fraction_training
        self.training = training
        self.ic_map_class = SingleComponent_map_rb

        if not self.training:
            if type(self.path_dataset) is str:
                simulation_data = load_pickle_(self.path_dataset)
                self.b = simulation_data['MD dataset']['b']
                self.P = simulation_data['args_initialise_simulation']['P']
                assert self.P is not None
                self.T = simulation_data['args_initialise_simulation']['T']
                self.u = self.reduced_enthalpy_from_reduced_energies_and_boxes_(simulation_data['MD dataset']['u'], boxes=self.b)
                self.u_mean = self.u.mean()
                print(f'This MD dataset contains {len(self.u)} datapoints')
            else:
                self.u_mean = np.array(self.path_dataset)
        else:
            self.import_MD_dataset_()
            print(f'This MD dataset contains {len(self.u)} datapoints')

    def reduced_enthalpy_from_reduced_energies_and_boxes_(self, reduced_energies, boxes):
        # enthalpy: see O/MM/Tx.py

        CONST_kB = 1e-3*8.31446261815324         # kilojoule/(kelvin*mole)
        CONST_PV_to_kJ_per_mol = 0.0610193412507 # 1 (atm) * (nm**3) = 0.0610193412507 kilojoule/mole

        kT = CONST_kB * self.T ; beta = 1.0 / kT

        assert len(boxes.shape) == 3
        h11, h22, h33 = boxes[...,0,0], boxes[...,1,1], boxes[...,2,2] # nm

        V = h11 * h22 * h33                                                    # (m,) # (nm)**3
        PV_reduced = beta * CONST_PV_to_kJ_per_mol * self.P * V  # (m,)        # P = float / atm
        ladJ_Vto6 = np.log(h22) + 2.0*np.log(h33)                              # (m,)

        enthalpies = reduced_energies.flatten() + PV_reduced + ladJ_Vto6

        return enthalpies.reshape([-1,1])  

    def u_(self, rb):
        r, h = rb_to_r_b_(rb)

        assert len(h.shape) == 3

        ## ## box to reduced form: just in case
        a,b,c = [h[:,i] for i in range(3)]
        c = c - b*np.round(c[:,1:2]/b[:,1:2])
        c = c - a*np.round(c[:,0:1]/a[:,0:1])
        b = b - a*np.round(b[:,0:1]/a[:,0:1])
        _h = np.stack([a, b, c], axis=1)
        ## ##
        
        u_reduced = self.sc.u_(r, b=_h)

        return self.reduced_enthalpy_from_reduced_energies_and_boxes_(u_reduced, boxes=h) # should be the same for _h

    def import_MD_dataset_(self,):
            self.Ts = 'NO'

            simulation_data = load_pickle_(self.path_dataset)
            self.sc = SingleComponent(**simulation_data['args_initialise_object'])
            self.sc.initialise_system_(**simulation_data['args_initialise_system'])
            self.sc.initialise_simulation_(**simulation_data['args_initialise_simulation'])
            assert self.sc.n_DOF in [3*(self.sc.N - 1), 3*self.sc.N]

            self.T = self.sc.T # Kelvin
            self.P = self.sc.P # atm
        
            _r = simulation_data['MD dataset']['xyz'].astype(np.float32)
            b = simulation_data['MD dataset']['b'].astype(np.float32)
            u = simulation_data['MD dataset']['u']

            self.u = self.reduced_enthalpy_from_reduced_energies_and_boxes_(u, boxes=b)
            self.u_mean = self.u.mean()

            self.r = r_b_to_rb_np_(_r,b)

            assert len(self.r) == len(self.u)
            self.n_training = int(self.u.shape[0]*self.fraction_training)

    def truncate_data_(self, m=None):
        m_initial = len(self.u)

        self.set_training_validation_split_(n_training=m)
        self.r = self.r_training
        self.u = self.u_training
        assert len(self.r) == len(self.u) == m
        self.n_training = int(m*self.fraction_training)
        print(f'{m} out of {m_initial} datapoints will be used from this dataset')

        del self.r_training
        del self.r_validation
        del self.u_training
        del self.u_validation
        self.inds_rand = None

    def set_ic_map_step1(self, ind_root_atom=11, option=None):
        '''
        self.sc.mol is available for this reason; to check the indices of atoms in the molecule
        once (ind_root_atom, option) pair is chosen, keep a fixed note of this for this molecule
        '''
        assert self.training
        self.ind_root_atom = ind_root_atom
        self.option = option

        PDBmol = './'+str(self.sc._single_mol_pdb_file_)
        self.ic_map = self.ic_map_class(PDB_single_mol = PDBmol)
        self.ic_map.set_ABCD_(ind_root_atom = self.ind_root_atom, option = self.option)

    def set_ic_map_step2(self,
                         inds_rand=None,
                         check_PES = True,
                         ):
        self.r = self.ic_map.remove_COM_from_data_(self.r)
        self.set_training_validation_split_(n_training=self.n_training, inds_rand=inds_rand)
        if check_PES: self.check_PES_matching_dataset_()
        else: pass

    def set_ic_map_step3(self,
                         n_mol_unitcell : int = 1, # !! important in this new version
                         whiten_setting = 2,
                        ):
        self.n_mol_unitcell = n_mol_unitcell

        self.ic_map.initalise_(rb_dataset = self.r,
                               n_mol_unitcell = self.n_mol_unitcell,
                               whiten_setting = whiten_setting,
                               )
        m = 1000
        r = np.array(self.r_validation[:m])
        x, ladJf = self.ic_map.forward_rb_(r)
        _r, ladJi = self.ic_map.inverse_rb_(x)
        print('ic_map inversion errors on a small random batch:')
        print('positons:', np.abs(_r-r).max())
        print('volume:', np.abs(ladJf + ladJi).max())

        try:
            value = np.abs(np.array(x)).max()
            if value > 1.000001: print('!!', value)
            else: print(value)
        except:
            values = []
            for _x in x:
                try: values.append(np.abs(np.array(_x)).max())
                except: pass
            value = max(values) # 1.0000004 , 1.0000006
            if value > 1.000001: print('!!', value)
            else: print(value)

    # set_model not here and the rest not here.

def adjust_ranges_(nn):
    # can use after step 3 in NN_interface_sc_multimap
    n_states = nn.n_crystals
    for atr in ['Focused_Bonds', 'Focused_Angles']: 
        ranges  = [getattr(nn.nns[k].ic_map, atr).ranges for k in range(n_states)]
        centres = [getattr(nn.nns[k].ic_map, atr).centres for k in range(n_states)]
        Min = tf.reduce_min(tf.stack([centres[k] - 0.5*ranges[k] for k in range(n_states)], axis=0), axis=0)
        Max = tf.reduce_max(tf.stack([centres[k] + 0.5*ranges[k] for k in range(n_states)], axis=0), axis=0)
        Ranges = Max - Min
        Centres = Min + 0.5*Ranges
        for k in range(n_states):
            getattr(nn.nns[k].ic_map, atr).ranges  = Ranges
            getattr(nn.nns[k].ic_map, atr).centres = Centres

class NN_interface_sc_multimap_rb(NN_interface_helper):
    def __init__(self,
                 name : str,
                 paths_datasets : list,
                 fraction_training : float = 0.8, # 0 << x < 1
                 running_in_notebook : bool = False,
                 training : bool = True,
                 ):
        super().__init__()
        self.name = name + '_SC_' # ...
        assert type(paths_datasets) == list
        self.paths_datasets = paths_datasets
        self.fraction_training = fraction_training
        self.running_in_notebook = running_in_notebook 
        self.training = training
        self.model_class = PGMcrys_v1_rb
        self.ic_map_class = SingleComponent_map_rb
        ##
        self.n_crystals = len(self.paths_datasets)

        self.nns = [NN_interface_sc_rb(
                    name = name,
                    path_dataset = self.paths_datasets[i],
                    fraction_training = self.fraction_training,
                    training = self.training,
                    ) for i in range(self.n_crystals)]

    def save_inds_rand_(self,):
        for k in range(self.n_crystals):
            self.nns[k].save_inds_rand_(key='_crystal_index='+str(k))
        #[nn.save_inds_rand_() for nn in self.nns]

    def load_inds_rand_(self,):
        for k in range(self.n_crystals):
            self.nns[k].load_inds_rand_(key='_crystal_index='+str(k))
        #[nn.load_inds_rand_() for nn in self.nns]

    def set_ic_map_step1(self, ind_root_atom=11, option=None):
        self.ind_root_atom = ind_root_atom
        self.option = option
        [nn.set_ic_map_step1(ind_root_atom=self.ind_root_atom, option=self.option) for nn in self.nns];

    def set_ic_map_step2(self, check_PES=True):
        [nn.set_ic_map_step2(inds_rand=None, check_PES=check_PES) for nn in self.nns];

    def set_ic_map_step3(self,
                         n_mol_unitcells : list = None, # !! important in this new version
                         whiten_setting = 2,
                        ):
        if n_mol_unitcells is None:
            print(
            f'''
            !! for each of the {self.n_crystals} supercell datasets provided,
            it is important to provide here a list (n_mol_unitcells) describing 
            the number of molecules in each underlying unitcell, respectively.
            '''
            )
            n_mol_unitcells = [None]*self.n_crystals
        else:
            if type(n_mol_unitcells) is int and self.n_crystals == 1: n_mol_unitcells = [n_mol_unitcells]
            else: 
                assert type(n_mol_unitcells) is list
                assert len(n_mol_unitcells) == self.n_crystals
        
        self.n_mol_unitcells = n_mol_unitcells 

        for i in range(self.n_crystals):
            self.nns[i].set_ic_map_step3(n_mol_unitcell = self.n_mol_unitcells[i],
                                         whiten_setting = whiten_setting,
                                        )

    def set_model(self,
                  learning_rate = 0.001,
                  evaluation_batch_size = 5000,
                  ##
                  n_layers = 4,
                  DIM_connection = 10,
                  n_att_heads = 4,
                  initialise = True, # for debugging in eager mode
                  test_inverse = True,
                  ):
        self.model = self.model_class(
                                    ic_maps = [nn.ic_map for nn in self.nns],
                                    n_layers = n_layers,
                                    optimiser_LR_decay = [learning_rate, 0.0],
                                    DIM_connection = DIM_connection,
                                    n_att_heads = n_att_heads,
                                    initialise = initialise,
                                    )
        self.model.print_model_size()
        self.evaluation_batch_size = evaluation_batch_size
        if test_inverse:
            for crystal_index in range(self.n_crystals):
                r = np2tf_(self.nns[crystal_index].r_validation[:self.evaluation_batch_size])
                inv_test_res0 = self.model.test_inverse_(r, crystal_index=crystal_index, graph =True)
                print('inv_test_res0',inv_test_res0) # TODO make interpretable
        else: pass

    def set_trainer(self, n_batches_between_evaluations = 50):
        self.trainer = TRAINER(model = self.model,
                               max_training_batches = 50000,
                               n_batches_between_evaluations = n_batches_between_evaluations,
                               running_in_notebook = self.running_in_notebook,
                               )
        
    def u_(self, r, k):
        return self.nns[k].u_(r)

    def train(self,
              n_batches = 2000,
              save_BAR = True,
              save_mBAR = False,

              save_misc = True,
              verbose = True,
              verbose_divided_by_n_mol = True,
              f_halfwindow_visualisation = 1.0, # int or list
              test_inverse = False,
              evaluate_on_training_data = False,
              #
              evaluate_main = True,
              training_batch_size = 1000, # 1000 was always used
              ):
        
        if save_BAR: name_save_BAR_inputs = str(self.name_save_BAR_inputs)
        else: name_save_BAR_inputs = None

        if save_mBAR: name_save_mBAR_inputs = str(self.name_save_mBAR_inputs)
        else: name_save_mBAR_inputs = None

        self.trainer.train(
                        n_batches = n_batches,

                        # needed, and always available: xyz coordinates of training and validation sets
                        list_r_training   = [nn.r_training for nn in self.nns],
                        list_r_validation = [nn.r_validation for nn in self.nns],

                        # needed, and always available: potential energies of training and validation sets
                        list_u_training   = [nn.u_training for nn in self.nns],
                        list_u_validation = [nn.u_validation for nn in self.nns],

                        # not needed if not relevant: [weights associated with training and validation sets]
                        list_w_training = None,
                        list_w_validation = None,

                        # if the FF is cheap can use, BAR_V estimates during training will be saved.
                        list_potential_energy_functions = [self.nns[k].u_ for k in range(self.n_crystals)],

                        # evalaution cost vs statistical significance of gradient of the loss (1000 is best)
                        training_batch_size = training_batch_size,
                        # evalaution cost vs variance of FE estimates (affects standard error from pymbar if save_BAR True)
                        evaluation_batch_size = self.evaluation_batch_size,

                        # always True, unless just quickly checking if the training works at all
                        evaluate_main = evaluate_main,
                        # pymbar will run later, but need to save those inputs during training
                        name_save_BAR_inputs = name_save_BAR_inputs,
                        name_save_mBAR_inputs = name_save_mBAR_inputs,

                        # statistical significance vs model quality. Dont need to suffle if the learned map is ideal, but it is not ideal.
                        shuffle = True,
                        # verbose: when running in jupyter notebook y_axis width of running plot that are shown during training
                        f_halfwindow_visualisation = f_halfwindow_visualisation, # f_latt / kT if verbose_divided_by_n_mol, else f_crys / kT
                        verbose = verbose, # whether to plot matplotlib plots during training
                        verbose_divided_by_n_mol = verbose_divided_by_n_mol, # plot and report lattice FEs (True) or crystal FEs (False) [lattice_FE = crystal_FE/n_mol]
                        
                        # evalaution cost:
                        evaluate_on_training_data = evaluate_on_training_data, # not needed.
                        test_inverse = test_inverse, # not needed if model ok, but useful to check sometimes. 
                            # Inversion errors must not be too biased (i.e., must include zero error). Some error is unavoidable due to model depth, and overfitting.
                            # BAR rewighting is quite robust even in the presence of large inversion errors on some sample, but only if the error distribution includes zero.
                            # If setting test_inverse True, this information is collected during training, but adds a lot of overhead cost during training.
                            # TODO: statistics collected about the quality of the inverses (inv_test) may be adjusted from what they are currently (mean, max).
                        # Disk space needed for this:
                        save_generated_configurations_anyway = False, 
                        # not tested yet but here in case the FF is expensive (self.u_ function missing, not defined, on purpose)
                        # when FF expensive, yet model is always cheap to sample, so sample model on each batch and save this along with other BAR inputs that can be used later
                        # use validation error to choose best model, and on those xyz samples from the model compute BAR (after evalauting the FF on the selected files)
                        # dont know when the model minimises validation error, so saving each evaluation batch and choose the most promising pymbar inputs afterwards.
                    )
        print('training time so far:', np.round(self.trainer.training_time, 2), 'minutes')

        if save_misc:
            self.save_misc_()
            print('misc training outputs were saved')
        else: pass
        if test_inverse:
            self.save_inv_test_results_()
            print('inv_test results were saved')
        else: pass

    def load_misc_(self,):
        self._FEs, self._SDs, self.estimates, self.evaluation_grid, self.AVMD_f_T_all, self.training_time = load_pickle_(self.name_save_misc)
        self.n_estimates = self.evaluation_grid.shape[0]
        for k in range(self.n_crystals):
            self.nns[k]._FEs = self._FEs[k]
            self.nns[k]._SDs = self._SDs[k]
            self.nns[k].estimates = self.estimates[k:k+1]
            self.nns[k].evaluation_grid = self.evaluation_grid
            self.nns[k].AVMD_f_T_all = self.AVMD_f_T_all[:,k:k+1]
            self.nns[k].n_estimates = self.n_estimates
        for nn in self.nns:
            nn.FEs = FE_of_model_curve_(nn.estimates[0,:,1], nn.estimates[0,:,7])
            nn.SDs = FE_of_model_curve_(nn.estimates[0,:,1], (nn.estimates[0,:,7]-nn.FEs)**2)**0.5

    def load_energies_during_training_(self):
        [self.nns[k].load_energies_during_training_(index_of_state=k) for k in range(self.n_crystals)];

    def plot_energies_during_training_(self, crystal_index=0, **kwargs):
        self.nns[crystal_index].plot_energies_during_training_(**kwargs)

    def solve_BAR_using_pymbar_(self, rerun=False):
        for k in range(self.n_crystals):
            self.nns[k].solve_BAR_using_pymbar_(rerun=rerun, index_of_state=k, key='_crystal_index='+str(k))

    def plot_result_(self, crystal_index=0, **kwargs):
        self.nns[crystal_index].plot_result_(**kwargs)
        

    def solve_mBAR_using_pymbar_(self, rerun=False,
                                 n_bootstraps = 0, # default in MBAR class https://pymbar.readthedocs.io/en/latest/mbar.html
                                 uncertainty_method = None, # can set 'bootstrap' if n_bootstraps not None
                                 save_output = True):
        # from energies that were saved during training
        
        n_states = int(self.n_crystals)
        try:
            self.estimates_mBAR = load_pickle_(self.name_save_mBAR_inputs+'_mBAR_output')
            print('found saved mBAR result')
            if rerun:
                print('rerun=True, so rerunning BAR calculation again.')
                assert 'go to'=='except'
            else: pass

        except:
            self.estimates_mBAR = np.zeros([4, self.n_estimates, n_states, n_states])
            HIGH = np.eye(n_states) + 1e20

            for i in range(self.n_estimates):
                clear_output(wait=True)
                print(i)

                name = self.name_save_mBAR_inputs+'_mBAR_input_'+str(i)+'_T'
                try:
                    x = load_pickle_(name) ; m = x.shape[-1]
                    mbar_res = MBAR(x.reshape([n_states, n_states*m]),
                                    np.array([m]*n_states),
                                    n_bootstraps=n_bootstraps,
                                    ).compute_free_energy_differences(uncertainty_method=uncertainty_method)
                    FE = mbar_res['Delta_f']
                    SE = mbar_res['dDelta_f']
                except:
                    FE = HIGH ; SE = HIGH
                    print('BAR: T estimate'+str(i)+'skipped')
                self.estimates_mBAR[0,i] = FE
                self.estimates_mBAR[1,i] = SE
                
                name = self.name_save_mBAR_inputs+'_mBAR_input_'+str(i)+'_V'
                try:
                    x = load_pickle_(name) ; m = x.shape[-1]
                    mbar_res = MBAR(x.reshape([n_states, n_states*m]),
                                    np.array([m]*n_states),
                                    n_bootstraps=n_bootstraps,
                                    ).compute_free_energy_differences(uncertainty_method=uncertainty_method)
                    FE = mbar_res['Delta_f']
                    SE = mbar_res['dDelta_f']
                except:
                    FE = HIGH ; SE = HIGH
                    print('BAR: V estimate'+str(i)+'skipped')
                self.estimates_mBAR[2,i] = FE
                self.estimates_mBAR[3,i] = SE

            self.estimates_mBAR  = np.where(np.isnan(self.estimates_mBAR),1e20,self.estimates_mBAR)

            if save_output:
                save_pickle_(self.estimates_mBAR, self.name_save_mBAR_inputs+'_mBAR_output')
                print('saved mBAR result')
            else: 
                print('mBAR results were recomputed but not saved')
            #self.set_final_result_()

    def sample_model_(self, m, crystal_index=0):
        n_draws = m//self.evaluation_batch_size
        return np.concatenate([self.model.sample_model(self.evaluation_batch_size, crystal_index=crystal_index)[0] for i in range(n_draws)],axis=0)

    def save_samples_(self, m:int=20000):
        for crystal_index in range(self.n_crystals):
            save_pickle_(self.sample_model_(m, crystal_index=crystal_index), self.name_save_samples+'_crystal_index='+str(crystal_index))

    def load_samples_(self, crystal_index=None):
        if crystal_index is None:
            self.samples_from_model = []
            for crystal_index in range(self.n_crystals):
                self.samples_from_model.append(load_pickle_(self.name_save_samples+'_crystal_index='+str(crystal_index)))
        else:
            self.samples_from_model = load_pickle_(self.name_save_samples+'_crystal_index='+str(crystal_index))




