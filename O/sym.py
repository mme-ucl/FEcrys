from .NN.representation_layers import *
from .plotting import *
from .MM.sc_system import *

## ## ## ## 

def cluster_symmetric_torsion_(x, symmetry_order:int=1, offset=np.pi):
    x = np.array(x).flatten()
    assert symmetry_order >= 1
    n = symmetry_order
    decision_boundaries = np.linspace(-np.pi, np.pi, n+1)[:n] + offset/n
    cluster_assignments = np.digitize(x, decision_boundaries)
    cluster_assignments = np.where(cluster_assignments==n, 0, cluster_assignments)
    #check = np.unique(cluster_assignments)
    #if len(check) <n: print(f'!! cluster_symmetric_torsion : {len(check)} of {n} states were sampled')
    #else: pass
    #assert max(check) == n-1
    return cluster_assignments

def test_cluster_symmetric_torsion_(symmetry_order=3, m=200, flattness=0.5):
    '''
    test_cluster_symmetric_torsion_(symmetry_order=10, m=1000, flattness=0.2)
    '''
    n = symmetry_order
    x = np.mod(np.random.randn(m)*flattness,np.pi*2/n)
    x = np.concatenate([x+b for b in np.linspace(-np.pi, np.pi,n+1)[:n]], axis=0)
    x = np.mod(x + PI, 2.0*PI) - PI

    cluster_assignments = cluster_symmetric_torsion_(x, symmetry_order=n)

    plot_1D_histogram_(x, color='black')
    [plot_1D_histogram_(x[cluster_assignments==k]) for k in range(n)]
    [plt.scatter(x[cluster_assignments==k],[0]*len(x[cluster_assignments==k])) for k in range(n)]

class DatasetSymmetryReduction:
    '''

    Chemically identical atoms can be swapped. 
    The functional form of the forcefield over the atoms that can be swapped is the same.
    (i.e., atoms that can be swapped have the same charge, LJ params, and bonded environment)

    When to use:
        can't use  : if rare even not symmetry-related (cannot use in the absence of symmetry)
        not needed : symmetry-related rate events do not affect (ergodicity in) distributions that are invariant to the symmetry
        use when   : when evaluating data on potentials / distributions that are not symmetry-aware
            this includes conventional PGM (current), and ECM (lambda < 1) states (unless lambda 0 distribution ideal gas)

    current methods: methyl, trimethyl
    TODO: add more methods if needed

    lookup_indices:

        (option 1)
        depends on initial permutation
        depending on the crystal unitcell, there are currently four choices 0,1,2,3 per molecule
        can add more in self.LOOKUPS if needed, check plots to adjust.
        Current default is [0,3] which works for one of the cases here with 4 molecules in unitcell

        (option 1) : entropic correction when lookup_indices chosen to reduce all symmetry-related states to one state:
            FE_ergodic = FE_sym_randomised = [ FE_sym_reduced - (# methyl + # trimethyl) * ln(3) ] ; lattice FE (kT)
            [from the point of view of current PGM: 3**(# methyl + # trimethyl) number of FE basins reduced to 1,
             this significantly reduces the demand on a symmetry-unaware model]
             
        (option 2) 
        An unsupervised mode is simply setting lookup_indices = [-1] which will randomise 
        the rotations such that all states are sampled uniformly. This gives E_sym_randomised.
        (option 2) : no entropic correction needed. Fixes the ergodicity problem but does not make PGM training easier.

    '''
    
    def save_sym_reduced_dataset_(self, path_dataset=None, key='_sym_reduced'):
        '''
        Before saving the processed dataset, can check_energy_() to confirm that energy did not change. 
        Configurations before (self.r_init) and after (self.r) the reduction should have exactly the same energy.
        [Allowing a slight noise if comparing to energies saved during trajectory; because of numerical precision]
        This is because only the atomic indices are swapped in this processing step (coordinates are not changed).
        '''
        if hasattr(self, 'path_dataset'):
            path_dataset = str(self.path_dataset)
        else: pass
        path_dataset_sym = path_dataset + key
        dataset = load_pickle_(path_dataset)
        dataset['MD dataset']['xyz'] = self.r
        save_pickle_(dataset, path_dataset_sym)

    def check_energy_(self, m=None):
        self.u_sym = self.sc.u_(self.r[:m], b=self.sc.boxes[:m])
        print('np.abs(self.u_sym - self.sc.u).max():', np.abs(self.u_sym - self.sc.u[:m]).max())

    def __init__(self,
                 path_dataset : str,
                 r = None,
                 n_mol=None,
                 n_atoms_mol=None,
                 PDB_single_mol=None,
                 ):
        
        if path_dataset is None: pass
        else:
            self.path_dataset = path_dataset

            self.sc = SingleComponent.initialise_from_save_(self.path_dataset)
            r = self.sc.xyz
            b = self.sc.boxes
            n_mol = self.sc.n_mol
            n_atoms_mol = self.sc.n_atoms_mol
            PDB_single_mol = str(self.sc._single_mol_pdb_file_.absolute())
            self.sc._xyz = []

        # r : (m, N, 3)
        self.r_init = np.array(r)
        self.b      = np.array(b)
        self.restart_()
        self.n_mol = n_mol
        self.n_atoms_mol = n_atoms_mol
        self.N = self.n_mol * self.n_atoms_mol
        self.PDB_single_mol = PDB_single_mol
        # 
        self.ic_map = SC_helper(self.PDB_single_mol)

        ## n=3:
        perms00 = [[0,1,2],
                   [0,2,1],
                   [1,0,2],
                   [1,2,0],
                   [2,0,1],
                   [2,1,0],
                  ]
        perms01 = [[perm[i] for i in [0,2,1]] for perm in perms00]
        
        unperm_ = lambda perm : np.array([np.where(np.array(perm).flatten() == i)[0][0] for i in range(3)]).tolist()
        
        perms10 = [unperm_(perm) for perm in perms00]
        perms11 = [[perm[i] for i in [0,2,1]] for perm in perms10]

        # nothing
        self.LOOKUP_00 = dict(zip([tuple(x) for x in perms00], [tuple(x) for x in perms00]))
        # swap last two
        self.LOOKUP_01 = dict(zip([tuple(x) for x in perms00], [tuple(x) for x in perms01]))
        # unpermute
        self.LOOKUP_10 = dict(zip([tuple(x) for x in perms00], [tuple(x) for x in perms10]))
        # untpermuate and swap last two
        self.LOOKUP_11 = dict(zip([tuple(x) for x in perms00], [tuple(x) for x in perms11]))
        self.LOOKUPS = [self.LOOKUP_00, self.LOOKUP_01, self.LOOKUP_10, self.LOOKUP_11]

        # [1,2] < [0,3] for that one, methyls (and trimethyls)

        self.LOOKUP_random_rotation = [ [0,1,2],
                                        [2,0,1],
                                        [1,2,0],
                                      ]

    def restart_(self,):
        self.r = np.array(self.r_init)
        self.n_frames = len(self.r)

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    def set_ABCD_(self, ind_root_atom, option:int=None):
        self.ind_rO = ind_root_atom
        self.ic_map.set_ABCD_(ind_root_atom=self.ind_rO, option=option)

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    def cluster_symmetric_torsion_(self, phi, symmetry_order:int, offset=np.pi):
        C = [cluster_symmetric_torsion_(phi, symmetry_order=symmetry_order, offset=offset)]
        for _ in range(symmetry_order-1): 
            C.append(np.mod(C[-1]+1, symmetry_order))
        return np.stack(C, axis=-1).reshape([self.n_frames, symmetry_order])

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
    def _prepare_sort_methyl_(self,):
        methyl_group_smarts = "[CH3]"
        methyl_pattern = Chem.MolFromSmarts(methyl_group_smarts)
        matches = self.ic_map.mol.GetSubstructMatches(methyl_pattern)
        inds_C = [x[0] for x in matches]

        self.inds_torsions_ch3 = []
        for i in inds_C:
            inds_H = np.where(self.ic_map.ABCD_IC[:,1]==i)[0]
            assert len(inds_H) == 3
            inds_torsions = [self.ic_map.ABCD_IC[j] for j in inds_H]
            self.inds_torsions_ch3.append(inds_torsions)
        self.n_methyl_groups = len(self.inds_torsions_ch3)
        print('# methyl groups:', self.n_methyl_groups)

    def _sort_methyl_(self, ind_methyl, ind_mol, lookup_index=0, offset=np.pi):
        inds_torsions_0 = np.array(self.inds_torsions_ch3[ind_methyl][0])
        inds_torsions_1 = np.array(self.inds_torsions_ch3[ind_methyl][1])
        inds_torsions_2 = np.array(self.inds_torsions_ch3[ind_methyl][2])

        a = ind_mol*self.n_atoms_mol ; b = a + self.n_atoms_mol

        phi_h0 = get_torsion_np_(reshape_to_molecules_np_(self.r[:,a:b], n_atoms_in_molecule=self.n_atoms_mol, n_molecules=1)[:,0], inds_torsions_0) 
        C = self.cluster_symmetric_torsion_(phi_h0, 3, offset=offset)
        
        inds_H = np.array([inds_torsions_0, inds_torsions_1, inds_torsions_2])
        inds_H_j =  inds_H + ind_mol * self.n_atoms_mol

        if lookup_index >= 0:
            for frame in range(self.n_frames):
                permutation = list(self.LOOKUPS[lookup_index][tuple(C[frame].tolist())])
                self.r[frame,inds_H_j,:] = np.take(self.r[frame,inds_H_j,:], permutation, axis=0)
        else:
            for frame in range(self.n_frames):
                #permutation = np.random.choice(3, 3, replace=False)
                permutation = self.LOOKUP_random_rotation[np.random.choice(3, 1, replace=False)[0]]
                self.r[frame,inds_H_j,:] = np.take(self.r[frame,inds_H_j,:], permutation, axis=0)

    ''' some cases need different offsets for different methyl groups, adding.
    def sort_methyl_(self, lookup_indices=[0,3]):
        if hasattr(self, 'n_trimethyl_groups'): pass
        else: self._prepare_sort_methyl_()

        if type(lookup_indices[0]) is int: lookup_indices = [lookup_indices]*self.n_methyl_groups
        else:                              assert len(lookup_indices) == self.n_methyl_groups
        
        for i in range(self.n_methyl_groups):
            lookup_inds = (list(lookup_indices[i])*self.n_mol)[:self.n_mol]
            print('dealing with methyl group',i)
            [self._sort_methyl_(i, j, lookup_index = lookup_inds[j]) for j in range(self.n_mol)];
    '''
    def sort_methyl_(self, lookup_indices=[0,3], offsets=[np.pi]):
        if hasattr(self, 'n_trimethyl_groups'): pass
        else: self._prepare_sort_methyl_()

        print(f'args provided: lookup_indices={lookup_indices}, offsets={offsets}')

        if type(lookup_indices[0]) is int:
            lookup_indices = [lookup_indices]*self.n_methyl_groups
            if len(offsets) > 1: print('!! check why this is printed')
            offsets = [offsets[0]]*self.n_methyl_groups
        else:
            assert len(lookup_indices) == self.n_methyl_groups
            assert len(offsets) == self.n_methyl_groups

        for i in range(self.n_methyl_groups):
            lookup_inds = (list(lookup_indices[i])*self.n_mol)[:self.n_mol]
            print('dealing with methyl group',i)
            [self._sort_methyl_(i, j, lookup_index = lookup_inds[j], offset=offsets[i]) for j in range(self.n_mol)];
    
    def plot_methyl_(self, axes_off=True, figsize=(6,6)):
        if hasattr(self, 'n_methyl_groups'): pass
        else: self._prepare_sort_methyl_()

        _range = [-np.pi, np.pi]
        fig, ax = plt.subplots(max(self.n_mol, 2), max(self.n_methyl_groups, 2), figsize=figsize)
        _r = reshape_to_molecules_np_(self.r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        for i in range(self.n_methyl_groups):
            phi_h0 = get_torsion_np_(_r,  self.inds_torsions_ch3[i][0])
            phi_h1 = get_torsion_np_(_r,  self.inds_torsions_ch3[i][1])
            phi_h2 = get_torsion_np_(_r,  self.inds_torsions_ch3[i][2])
            m = len(phi_h0)
            # phi_h0 : (m, n_mol, 1)
            a = i#*3 
            #b = i*3 + 1
            #c = i*3 + 2
            for j in range(self.n_mol):
                plot_1D_histogram_(phi_h0[:,j,0], range=_range, color='C0', ax=ax[j,a], mask_0=True)
                plot_1D_histogram_(phi_h1[:,j,0], range=_range, color='C1', ax=ax[j,a], mask_0=True)
                plot_1D_histogram_(phi_h2[:,j,0], range=_range, color='C2', ax=ax[j,a], mask_0=True)
                ax[j,a].scatter(phi_h0[:,j,0], [-1]*m, color='C0', s=0.1)
                ax[j,a].scatter(phi_h1[:,j,0], [-2]*m, color='C1', s=0.1)
                ax[j,a].scatter(phi_h2[:,j,0], [-3]*m, color='C2', s=0.1)
                for k in [a]:#,b,c]:
                    ax[j,k].set_xlim(_range)
                    if axes_off: ax[j,k].set_axis_off()
                    else: pass
        plt.tight_layout()

    ## ##

    def _prepare_sort_trimethyl_(self,):
        trimethyl_group_smarts = "CC(C)(C)C"
        trimethyl_pattern = Chem.MolFromSmarts(trimethyl_group_smarts)
        matches = self.ic_map.mol.GetSubstructMatches(trimethyl_pattern)
        inds_group_occurrences = [list(x) for x in matches]
        
        self.inds_torsions_trimethyl_cA = []
        self.inds_trimethyl_cA012h3 = []
        self.n_trimethyl_groups = 0
        for inds_group in inds_group_occurrences:
            '''
            cD? - cC - cB - cA0 - cA0h3      (1) - (1) - (1) - [(1) - (3)] ; [move whole group according to the (1) part]
                          - cA1 - cA1h3                      - [(1) - (3)] ; [move ..]
                          - cA2 - cA2h3                      - [(1) - (3)] ; [move ..]
            '''
            ind_cC, ind_cB, ind_cA0, ind_cA1, ind_cA2 = inds_group # (5)
            ind_cD = np.where(np.abs(self.ic_map.ABCD_IC[:,:3]-np.array([[ind_cA0, ind_cB, ind_cC]])).sum(1) == 0)[0][0]
            #assert len(ind_cD) == 1
            ' actually cC and cD both dont matter'

            inds_cA0h3 = [self.ic_map.ABCD_IC[j][0] for j in np.where(self.ic_map.ABCD_IC[:,1]==ind_cA0)[0]] # (3)
            inds_cA1h3 = [self.ic_map.ABCD_IC[j][0] for j in np.where(self.ic_map.ABCD_IC[:,1]==ind_cA1)[0]] # (3)
            inds_cA2h3 = [self.ic_map.ABCD_IC[j][0] for j in np.where(self.ic_map.ABCD_IC[:,1]==ind_cA2)[0]] # (3)
            self.inds_trimethyl_cA012h3.append([inds_cA0h3, inds_cA1h3, inds_cA2h3])

            #self.inds_torsions_trimethyl_cA.append([self.ic_map.ABCD_IC[j] for j in [ind_cA0, ind_cA1, ind_cA2]])
            inds_torsions_trimethyl_cA= []
            for j in [ind_cA0, ind_cA1, ind_cA2]:
                inds_torsions_trimethyl_cA.append(self.ic_map.ABCD_IC[np.where(self.ic_map.ABCD_IC[:,0] == j)[0][0]])

            self.inds_torsions_trimethyl_cA.append(inds_torsions_trimethyl_cA)
            
            self.n_trimethyl_groups += 1

        assert len(self.inds_torsions_trimethyl_cA) == len(self.inds_trimethyl_cA012h3)
        print('# trimethyl groups:', self.n_trimethyl_groups)

    def _sort_trimethyl_(self, ind_trimethyl, ind_mol, lookup_index=0, offset=0):

        occurrence = ind_trimethyl
        i = ind_mol
        inds_torsions = self.inds_torsions_trimethyl_cA[occurrence]
        phi_cA0 = get_torsion_np_(reshape_to_molecules_np_(self.r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol),  inds_torsions[0])
        inds_cA0_cA1_cA2 = np.array([inds_torsions[0], inds_torsions[1], inds_torsions[2]])
        inds_A0h3_A1h3_A2h3 = np.array(self.inds_trimethyl_cA012h3[occurrence])

        C = self.cluster_symmetric_torsion_(phi_cA0[:,i], 3, offset=offset)

        ''' moving the carbons A0, A1, A2 (the main part) '''
        inds_cA0_cA1_cA2_i = inds_cA0_cA1_cA2 + i * self.n_atoms_mol
        r_cA0_cA1_cA2_in = self.r[:,inds_cA0_cA1_cA2_i,:] # (m,3,3)
        # (3,3) take (3) from axis 0
        #r_cA0_cA1_cA2_out =  np.array([np.take(r_cA0_cA1_cA2_in[frame], C[frame], axis=0) for frame in range(self.n_frames)])
        #self.r[:,inds_cA0_cA1_cA2_i,:] = r_cA0_cA1_cA2_out

        ''' moving the triplets of hydrogens that were on these carbons to match '''
        
        inds_A0h3_A1h3_A2h3_i = inds_A0h3_A1h3_A2h3 + i * self.n_atoms_mol
        '''
        [A0h0,A0h1,A0h2]
        [A1h0,A1h1,A1h2]
        [A2h0,A2h1,A2h2]
        moving the rows
        '''
        inds_h0 = inds_A0h3_A1h3_A2h3_i[:,0] # (3)
        r_h0_in = self.r[:,inds_h0,:] 
        #r_h0_out =  np.array([np.take(r_h0_in[frame], C[frame], axis=0) for frame in range(self.n_frames)])
        #self.r[:,inds_h0,:] = r_h0_out

        inds_h1 = inds_A0h3_A1h3_A2h3_i[:,1] # (3)
        r_h1_in = self.r[:,inds_h1,:] 
        #r_h1_out =  np.array([np.take(r_h1_in[frame], C[frame], axis=0) for frame in range(self.n_frames)])
        #self.r[:,inds_h1,:] = r_h1_out

        inds_h2 = inds_A0h3_A1h3_A2h3_i[:,2] # (3)
        r_h2_in = self.r[:,inds_h2,:] 
        #r_h2_out =  np.array([np.take(r_h2_in[frame], C[frame], axis=0) for frame in range(self.n_frames)])
        #self.r[:,inds_h2,:] = r_h2_out

        if lookup_index >= 0:
            for frame in range(self.n_frames):
                permutation = list(self.LOOKUPS[lookup_index][tuple(C[frame].tolist())])
                self.r[frame, inds_cA0_cA1_cA2_i, :] = np.take(r_cA0_cA1_cA2_in[frame],  permutation, axis=0)
                self.r[frame, inds_h0, :] = np.take(r_h0_in[frame], permutation, axis=0)
                self.r[frame, inds_h1, :] = np.take(r_h1_in[frame], permutation, axis=0)
                self.r[frame, inds_h2, :] = np.take(r_h2_in[frame], permutation, axis=0)
        else:
            for frame in range(self.n_frames):
                # permutation = np.random.choice(3, 3, replace=False)
                permutation = self.LOOKUP_random_rotation[np.random.choice(3, 1, replace=False)[0]]
                self.r[frame, inds_cA0_cA1_cA2_i, :] = np.take(r_cA0_cA1_cA2_in[frame],  permutation, axis=0)
                self.r[frame, inds_h0, :] = np.take(r_h0_in[frame], permutation, axis=0)
                self.r[frame, inds_h1, :] = np.take(r_h1_in[frame], permutation, axis=0)
                self.r[frame, inds_h2, :] = np.take(r_h2_in[frame], permutation, axis=0)
        
    def sort_trimethyl_(self, lookup_indices=[0,3], offset=0):
        if hasattr(self, 'n_trimethyl_groups'): pass
        else: self._prepare_sort_trimethyl_()

        print(f'args provided: lookup_indices={lookup_indices}, offset={offset}')

        lookup_indices = (list(lookup_indices)*self.n_mol)[:self.n_mol]
        for i in range(self.n_trimethyl_groups):
            print('dealing with trimethyl group',i)
            [self._sort_trimethyl_(i, j, lookup_index = lookup_indices[j], offset=offset) for j in range(self.n_mol)];

    def plot_trimethyl_(self, mask_0=True, axes_off=True, figsize=(2,10)):
        if hasattr(self, 'n_trimethyl_groups'): pass
        else: self._prepare_sort_trimethyl_()

        _range = [-np.pi, np.pi]
        fig, ax = plt.subplots(self.n_mol, self.n_trimethyl_groups, figsize=figsize)
        _r = reshape_to_molecules_np_(self.r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        for i in range(self.n_trimethyl_groups):
            phi_h0 = get_torsion_np_(_r,  self.inds_torsions_trimethyl_cA[i][0])
            phi_h1 = get_torsion_np_(_r,  self.inds_torsions_trimethyl_cA[i][1])
            phi_h2 = get_torsion_np_(_r,  self.inds_torsions_trimethyl_cA[i][2])
            m = len(phi_h0)
            # phi_h0 : (m, n_mol, 1)
            a = i#*3 
            #b = i*3 + 1
            #c = i*3 + 2
            for j in range(self.n_mol):
                plot_1D_histogram_(phi_h0[:,j,0], range=_range, color='C0', ax=ax[j], mask_0=mask_0)
                plot_1D_histogram_(phi_h1[:,j,0], range=_range, color='C1', ax=ax[j], mask_0=mask_0)
                plot_1D_histogram_(phi_h2[:,j,0], range=_range, color='C2', ax=ax[j], mask_0=mask_0)
                ax[j].scatter(phi_h0[:,j,0], [-1]*m, color='C0', s=1)
                ax[j].scatter(phi_h1[:,j,0], [-2]*m, color='C1', s=1)
                ax[j].scatter(phi_h2[:,j,0], [-3]*m, color='C2', s=1)
                ax[j].set_xlim(_range)
                if axes_off: ax[j].set_axis_off()
                else: ax[j].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
        plt.tight_layout()

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

    def plot_torsion_(self, index_of_atom:int, axes_off=False):
        '''
        # plot_mol_larger_(self.sc.mol) ; to see atoms
        # self.ic_map.ABCD_IC           ; to see all indices
        '''

        can_plot = index_of_atom in self.ic_map.inds_atoms_IC
        if can_plot: pass
        else:  print('! this atom does not have an associated torsional angle') ; assert can_plot

        inds_phi = self.ic_map.ABCD_IC[np.where(self.ic_map.ABCD_IC[:,0]==index_of_atom)[0][0]]
        print(f'Plotting torsion histogram of atom {index_of_atom} (indices: {inds_phi})')

        _r = reshape_to_molecules_np_(self.r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        phi = get_torsion_np_(_r, inds_phi)

        _range = [-np.pi, np.pi]
        m = len(phi)
        fig, ax = plt.subplots(self.n_mol, figsize=(6,6))
        for j in range(self.n_mol):
            plot_1D_histogram_(phi[:,j,0], range=_range, color='C0', ax=ax[j], mask_0=True)
            ax[j].scatter(phi[:,j,0], [-1]*m, color='C0', s=0.1)
            ax[j].set_xlim(_range)
            if axes_off: ax[j].set_axis_off()
            else: pass
        plt.tight_layout()

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## reshuffle indices of two selected atoms:

    def sort_n2_(self, inds_AB:list, lookup_indices:list=[-1], offset=0):

        print(f'args provided: inds_AB={inds_AB}, lookup_indices={lookup_indices}, offset={offset}')

        inds_A, inds_B = self.ic_map.ABCD_IC[[np.where(self.ic_map.ABCD_IC[:,0]==ind)[0][0] for ind in inds_AB]]
        assert [inds_A[i]==inds_B[i] for i in [1,2,3]] and inds_A[0]!=inds_B[0]
        
        lookup_indices = (list(lookup_indices)*self.n_mol)[:self.n_mol]
        print('dealing with custom n2 group')
        [self._sort_n2_(inds_A, inds_B, j, lookup_index=lookup_indices[j], offset=offset) for j in range(self.n_mol)];

    def _sort_n2_(self, inds_A, inds_B, ind_mol, lookup_index=0, offset=0):
        inds_torsions_0 = inds_A
        inds_torsions_1 = inds_B

        a = ind_mol*self.n_atoms_mol ; b = a + self.n_atoms_mol

        phi_h0 = get_torsion_np_(reshape_to_molecules_np_(self.r[:,a:b], n_atoms_in_molecule=self.n_atoms_mol, n_molecules=1)[:,0], inds_torsions_0) 
        C = self.cluster_symmetric_torsion_(phi_h0, 2, offset=offset)

        inds_X_j = np.array([inds_torsions_0[0], inds_torsions_1[0]]) + ind_mol * self.n_atoms_mol

        assert lookup_index in [-1, 0, 1]

        if lookup_index >= 0:
            LOOKUPS = [{(0,1):(0,1), (1,0):(1,0)}, {(0,1):(1,0), (1,0):(0,1)}]
            for frame in range(self.n_frames):
                permutation = list(LOOKUPS[lookup_index][tuple(C[frame].tolist())])
                self.r[frame,inds_X_j,:] = np.take(self.r[frame,inds_X_j,:], permutation, axis=0)
        else:
            LOOKUPS = [[0,1], [1,0]]
            for frame in range(self.n_frames):
                permutation = LOOKUPS[np.random.choice(2, 1, replace=False)[0]]
                self.r[frame,inds_X_j,:] = np.take(self.r[frame,inds_X_j,:], permutation, axis=0)

    def plot_n2_(self, inds_AB:list, axes_off=True, figsize=(6,6)):

        inds_A, inds_B = self.ic_map.ABCD_IC[[np.where(self.ic_map.ABCD_IC[:,0]==ind)[0][0] for ind in inds_AB]]
        assert [inds_A[i]==inds_B[i] for i in [1,2,3]] and inds_A[0]!=inds_B[0]

        _range = [-np.pi, np.pi]
        fig, ax = plt.subplots(self.n_mol, figsize=figsize)
        _r = reshape_to_molecules_np_(self.r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)

        phi_h0 = get_torsion_np_(_r, inds_A)
        phi_h1 = get_torsion_np_(_r, inds_B)
        m = len(phi_h0)

        for j in range(self.n_mol):
            plot_1D_histogram_(phi_h0[:,j,0], range=_range, color='C0', ax=ax[j], mask_0=True)
            plot_1D_histogram_(phi_h1[:,j,0], range=_range, color='C1', ax=ax[j], mask_0=True)
            ax[j].scatter(phi_h0[:,j,0], [-1]*m, color='C0', s=0.1)
            ax[j].scatter(phi_h1[:,j,0], [-2]*m, color='C1', s=0.1)
            ax[j].set_xlim(_range)
            if axes_off: ax[j].set_axis_off()
            else: pass
        plt.tight_layout()


    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## reshuffle unit cells by global translation under PBCs and re-indexing:

    def sort_unitcells_(self,
                         n_mol_unitcell,
                         n_images_search = 1,
                         ):
        ''' 
        Experimental method to improve ergodicity of finite MD data, from a point of view of a simple, 
        non-permutationally invariant (symmetry unaware) model. This method only applicable in supercell 
        data where there are more than one unit cell building blocks. 
        Symmetry augmentation : randomisation, such that each region of space inside a supercell 
        becomes effectively sampled by more than one crystallographically equivalent molecule.  
        Each molecule inside a unit cell building block is considered crystallographically unique. 
        Therefore, whole unit cells are reshuffled, rather than individual molecules.
        The output dataset (self.r) will always have the same energy as the input (self.r_init), 
        which is true for any method in this .py file, but the output from this method should still 
        be treated carefully as explained blow. This aspect matters form the point of view of a native 
        model (for accurate entropy differences between states; purpose of this .py file).        
        '''
        self.sort_unitcells_obj = PermuteUnitcell_SingleComponent(
                        n_atoms_mol = self.n_atoms_mol,
                        n_mol = self.n_mol,
                        n_mol_unitcell = n_mol_unitcell,

                        ind_rO = self.ind_rO,
                        n_images_search = n_images_search,
        )#(r=self.r, b=self.b)
        self.r = self.sort_unitcells_obj(r=self.r, b=self.b)
        print('About sort_unitcells_: if this worked *correcly, this data is only comptible with:')
        print('SingleComponent_map_r or SingleComponent_map_rb.')
        print('This is because whole molecules are jumping across PBCs (expected output).')
        print('*correctly = the whole molecules are not jumping in any other way.')
        print('Can now run check_sorted_unitcells_() to *check.')

    def check_sorted_unitcells_(self, batch_size = 10000):
        ''' check that sort_unitcells_ worked correctly '''
        r_ori = self.sort_unitcells_obj.put_in_box_m_(np.array(self.r_init), self.b)

        m = self.r.shape[0]
        rO_ori = []
        rO_aug = []
        n_batches = m // batch_size + (1 if m % batch_size else 0)
        for i in range(n_batches):
            r_batch_ori = reshape_to_molecules_np_(r_ori[i*batch_size:(i+1)*batch_size],
                                                   self.n_mol, self.n_atoms_mol)
            # (frames, molecule, atoms, 3)
            rO_ori.append(r_batch_ori[...,self.ind_rO,:])
            ##
            r_batch_aug = reshape_to_molecules_np_(self.r[i*batch_size:(i+1)*batch_size],
                                               self.n_mol, self.n_atoms_mol)
            # (frames, molecule, atoms, 3)
            rO_aug.append(r_batch_aug[...,self.ind_rO,:])
        rO_ori = np.concatenate(rO_ori, axis=0) # (m, n_mol, 3)
        rO_aug = np.concatenate(rO_aug, axis=0) # (m, n_mol, 3)
        b_inv = np.linalg.inv(self.b)
        sO_ori = np.einsum('omi,oij->omj', rO_ori, b_inv)*PI*2.0 - PI
        sO_aug = np.einsum('omi,oij->omj', rO_aug, b_inv)*PI*2.0 - PI

        av_sO_ori = average_torsion_np_(sO_ori, axis=0,  keepdims=True, pooling_method_=np.median)
        av_sO_aug = average_torsion_np_(sO_aug, axis=0,  keepdims=True, pooling_method_=np.median)

        sO_ori_centred = tf.math.floormod(sO_ori - av_sO_ori + PI, 2.0*PI) / (PI*2.0) # -> [0,1)
        sO_aug_centred = tf.math.floormod(sO_aug - av_sO_aug + PI, 2.0*PI) / (PI*2.0) # -> [0,1)

        # can now compare if minim and maxima are consistent
        self.ranges_ori = np.max(sO_ori_centred, axis=0) - np.min(sO_ori_centred, axis=0) # (n_mol, 3)
        self.ranges_aug = np.max(sO_aug_centred, axis=0) - np.min(sO_aug_centred, axis=0) # (n_mol, 3)

        print('All rO atoms in reduced supercell box [0,1) were aligned to check ranges:')
        print(f'original  : max range rO = {self.ranges_ori.max().round(3)} (should be < 0.5)')
        print(f'augmented : max range rO = {self.ranges_aug.max().round(3)} (should be < 0.5)')
        print('the above two numbers should be similar if the method worked correctly.')
        print('Further details can be seen in self.ranges_ori vs. self.ranges_aug')

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class PermuteUnitcell_SingleComponent:
    def put_in_box_m_(self, r, b):
        # r : (m,N,3)
        # b : (m,3,3)
        ''' adapted from SingleComponent_map_rb.remove_COM_from_data_ in O/NN/pgm_rb.py '''
        r = reshape_to_molecules_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        rO = r[...,self.ind_rO:self.ind_rO+1,:]
        rO_shifted = rO - rO[:,:1]
        s = np.einsum('omni,oij->omnj', rO_shifted, np.linalg.inv(b))
        s = np.mod(s, 1.0)
        rO_shifted = np.einsum('omni,oij->omnj', s, b)
        r = np.array(r-rO+rO_shifted)
        r = reshape_to_atoms_np_(r, n_atoms_in_molecule=self.n_atoms_mol, n_molecules=self.n_mol)
        return r
    
    '''
    def sq_distances_from_ref_ortho_1_( self,
                                        r,        # (N,3)
                                        r_ref,    # (N,3)
                                        box,      # (3,3)
                                        ):
        box_inv = np.linalg.inv(box)
        s     = np.einsum('...i,ij->...j', r    , box_inv)
        s_ref = np.einsum('...i,ij->...j', r_ref, box_inv)
        sv = s[:, None] - s_ref[None, :]
        v = np.einsum('...i,ij->...j', sv - np.round(sv), box)
        d_sq = (v*v).sum(-1)
        return d_sq
    '''

    def sq_distances_from_ref_general_1_(   self,
                                            r,      # (N,3)
                                            r_ref,  # (N,3)
                                            b,      # (N,3)
                                            ):
        ''' adapted from vectors_between_atoms_ in O/MM/mm_helper.py'''
        r = np.array(r) # (N,3)
        b = np.array(b)[np.newaxis,...] # (1,3,3)
    
        r_to_v_ = lambda rA, rB : np.expand_dims(rB,axis=-3) - np.expand_dims(rA,axis=-2) # (1,N,3) - (N,1,3)
        grid = np.arange(-self.n_images_search, self.n_images_search+1).astype(np.float32)
        
        rs = [r]
        for sign_a in grid:
            for sign_b in grid:
                for sign_c in grid:
                    if any((sign_a,sign_b,sign_c)):
                        rs.append(r + sign_a*b[:,0] + sign_b*b[:,1] + sign_c*b[:,2])
                    else:
                        pass
    
        len_rs = len(rs) # 27 (if n_images_search=1)

        vs = np.stack([r_to_v_(r_ref, rs[i]) for i in range(len_rs)],axis=0) # (27,N,N,3)
        ds_sq = (vs*vs).sum(-1)
        #ds = np.linalg.norm(vs, axis=-1)                                    # (27,N,N)
        d_sq = np.min(ds_sq, axis=0)                                         # (N,N)
        return d_sq

    def __init__(self, 
                 n_atoms_mol,
                 n_mol,
                 n_mol_unitcell,

                 ind_rO,
                 n_images_search = 1,
                 ):
        self.n_atoms_mol = n_atoms_mol
        self.n_mol = n_mol
        self.n_mol_unitcell = n_mol_unitcell

        ## ## 
        self.N = n_mol * n_atoms_mol
        self.n_atoms_unitcell = n_mol_unitcell * n_atoms_mol
        self.n_unitcells = self.N // self.n_atoms_unitcell
        ## ## 
        if self.n_unitcells == 1:
            print('\n !! This method will do nothing when the supercell contains only one unit cell. \n')
        else:
            print(f'\n  There are {self.n_unitcells} unit cells in the supercell. \n ')

        self.ind_rO = ind_rO

        self.n_images_search = n_images_search
        self.sq_distances_from_ref_1_ = self.sq_distances_from_ref_general_1_
   
        ## ## ## ## 

    def __call__(self, 
                    r, # (m,N,3)
                    b, # (m,3,3)
                    ):
        if self.n_unitcells > 1:
            output = []
            m = len(r)
            assert len(b) == m
            for i in range(m):
                # print in case disordered supercells (linear_sum_assignment could be slow)
                print(f'randomising unit cell permutations in frame {i+1} / {m}', end='\r')
                output.append(self.permute_unitcells_single_frame_(r[i], b[i]))
            # test that the output worked in sym.py
            return np.array(output)
        else: 
            return r

    def permute_unitcells_single_frame_(self,
                                        r,   # (N,3)
                                        box, # (3,3)
                                    ):
        def wrap_(r, b):
            s = np.einsum('...i,ij->...j', r, np.linalg.inv(b))
            _r = np.einsum('...i,ij->...j', np.mod(s, 1.0), b)
            return _r

        r = np.array(r)
        
        r = self.put_in_box_m_(r[None], box[None])[0]
        rc = r.reshape([self.n_unitcells, self.n_atoms_unitcell, 3])
        
        ## ## ## ## 

        rO = rc[:,self.ind_rO:self.ind_rO+1]              # (n_unitcells, 1, 3)

        idx_random_unitcell = np.random.randint(0, high=self.n_unitcells, size=1, dtype=int)[0]
        v = rO[idx_random_unitcell:idx_random_unitcell+1] # (1,1,3)
        _rO = wrap_(rO - v, box)                          # (n_unitcells,1,3)
        
        rc = rc - rO + _rO

        # the main step (permutation or unit cell blocks of atoms):
        d = self.sq_distances_from_ref_1_(_rO[:,0], rO[:,0], box) # (n_unitcells, n_unitcells)
        p = sp.optimize.linear_sum_assignment(d)[1]               # (n_unitcells, )
        # plt.matshow(d)
        # print(p)
        rc = np.take(rc, p, axis=0)

        ## ## ## ## 
        
        r = rc.reshape([self.n_unitcells*self.n_atoms_unitcell, 3])
        r = self.put_in_box_m_(r[None], box[None])[0]

        return r






