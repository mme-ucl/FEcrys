from O.MM.ecm_basic import *
import glob
## ## ## ## ## 

class SingleComponent_proxy:
    def __init__(self, name):
        '''
        saves a lot of time initialising g_of_T when energies already evaluated
        '''
        dataset = load_pickle_(name)
        self.P = dataset['args_initialise_simulation']['P']
        self.n_atoms_mol = dataset['args_initialise_object']['n_atoms_mol']
        self.N = PDB_to_xyz_(dataset['args_initialise_object']['PDB']).shape[0]
        self.n_mol = self.N // self.n_atoms_mol
        self.T = dataset['args_initialise_simulation']['T']
        
        self.xyz   = 'None'
        self.boxes = dataset['MD dataset']['b']
        self.u     = dataset['MD dataset']['u']
        self.temperature = dataset['MD dataset']['T']
        self.n_DOF = 3*(self.N - 1)

        self.u_    = 'None' 

# 1 (atm) * (nm**3) = 0.0610193412507 kJ/mol
CONST_PV_to_kJ_per_mol = 0.0610193412507

all_lower_triangular_ = lambda boxes : all([np.abs(boxes[...,i,j]).max() < 1e-5 for i,j in zip([0,0,1],[1,2,2])])

class g_of_T:

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## helper methods for instantaneous enthalpy

    def PV_isotropic_(self, boxes, T):
        # when [r,V] is the microstate 
        kT = CONST_kB * T ; beta = 1.0 / kT
        V = np.linalg.det(boxes)
        PV_reduced = beta * CONST_PV_to_kJ_per_mol * self.P * V
        return PV_reduced # kT

    def PV_anisotropic_(self, boxes, T):
        # when [r,h] is the microstate
        kT = CONST_kB * T ; beta = 1.0 / kT
        assert all_lower_triangular_(boxes), 'box not lower-triangular'
        h11, h22, h33 = boxes[...,0,0], boxes[...,1,1], boxes[...,2,2]
        V = h11*h22*h33
        PV_reduced = beta * CONST_PV_to_kJ_per_mol * self.P * V
        ladJ_Vto6 = np.log(h22) + 2.0*np.log(h33)
        return PV_reduced + ladJ_Vto6  # kT

    def ladJ_Vto6_(self, boxes):
        # just to prevent re-evalauting everything again when only this is the differences
        assert all_lower_triangular_(boxes), 'box not lower-triangular'
        h22, h33 = boxes[...,1,1], boxes[...,2,2]
        ladJ_Vto6 = np.log(h22) + 2.0*np.log(h33)
        return ladJ_Vto6
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## if self.Tref_FE is a Helmholtz FE, this is used to convert it to Gibbs FE (default: isotropic box fluctuations)

    def set_Tref_g_(self, Tref_FE=None, version=1, bins:int=40):
        if Tref_FE is not None: self.Tref_FE = Tref_FE
        else: pass
        '''
        def log_gaussian_1D_(x, data):
            mu = np.mean(data)
            sd = ((data - mu)**2).mean()**0.5
            px = np.exp(-0.5*((x-mu)/sd)**2) / (sd*np.sqrt(2*np.pi))
            ln_px = np.log(px)
            return ln_px
        '''
        self.f2g_correction_params = {'version':version, 'bins':bins}

        def log_histogram_1D_(x, data, bins=40):
            h,ax = np.histogram(data, bins=bins, density=True)
            ax = ax[1:]-0.5*(ax[1]-ax[0])
            idx_bin = np.argmin((ax - x)**2)
            p_x = h[idx_bin]
            return np.log(p_x)
            
        #if bins is None: log_1D_model_ = lambda x, data: log_gaussian_1D_(x, data)
        #else:            log_1D_model_ = lambda x, data: log_histogram_1D_(x, data, bins=bins)
        log_1D_model_ = lambda x, data: log_histogram_1D_(x, data, bins=bins)

        b = np.array(self.NPT_systems[self.Tref].boxes)

        if version == 0:
            self.ln_pV = 0.0
        elif version == 1:
            self.ln_pV = log_1D_model_(np.linalg.det(self.Tref_box), data=np.linalg.det(b))
        else:
            self.ln_pV = log_1D_model_(self.Tref_box[0,0], data=b[:,0,0])
            self.ln_pV += log_1D_model_(self.Tref_box[1,1], data=b[:,1,1])
            self.ln_pV += log_1D_model_(self.Tref_box[2,2], data=b[:,2,2])
            if version == 3:
                self.ln_pV += log_1D_model_(self.Tref_box[1,0], data=b[:,1,0])
                self.ln_pV += log_1D_model_(self.Tref_box[2,0], data=b[:,2,0])
                self.ln_pV += log_1D_model_(self.Tref_box[2,1], data=b[:,2,1])
            else: pass

        PV_reduced = self.bT_to_reduced_PV_(self.Tref_box, self.Tref)
        self.Tref_f_to_g_correction = PV_reduced + self.ln_pV 
        self.Tref_g = self.Tref_FE + self.Tref_f_to_g_correction

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## init 

    def __init__(self,
                Tref : float = 300,
                Tref_FE : float = 0.0,        # f_crys or g_crys (not *_latt; NOT divided by n_mol)
                Tref_SE : float = 0.0,        # se_crys                      (NOT divided by n_mol)
                Tref_box : np.ndarray = None, # if None, FE assumed to be already Gibbs, and anisotropic

                paths_datasets_NPT : list = [],
                xyz_not_in_datasets : bool = False, # in the presence of self.evalautions that are loaded from save, CUT datasets save time in most scenarious.
                f2g_correction_params : dict = {'version':1, 'bins':40},
                ):
        ''' This whole class deals with only on polymorph. To get Tx, need at least two instances. '''
        
        self.Tref = Tref                                   # reference temperature in Kelvin
        self.Tref_FE = Tref_FE                             # crystal FE computed at the reference temperature
        self.Tref_SE = Tref_SE                             # crystal standard error computed at the reference temperature
        self.Tref_box = Tref_box                           # if the computed FE above is Helmholtz, this is not None
                                                           #     if not None: (3,3) fixed box used during NVT
        self.paths_datasets_NPT = paths_datasets_NPT       # list of paths (list of strings) to all the temperature replica NPT datasets
        self.xyz_not_in_datasets = xyz_not_in_datasets     # 

        self.f2g_correction_params = f2g_correction_params # if the computed FE above is Helmholtz, this is used inside set_Tref_g_
        
        ## ## ## ## 
        ## organising self.NPT_systems

        self.NPT_systems = {}
        import time
        for name in self.paths_datasets_NPT:
            if xyz_not_in_datasets:
                sc = SingleComponent_proxy(name)
            else:
                sc = SingleComponent.initialise_from_save_(name, verbose=False)
                time.sleep(5)
            self.NPT_systems[name] = sc
        self.NPT_systems = [self.NPT_systems[key] for key in self.NPT_systems.keys()]
        self.datasets_converged = np.array([TestConverged_1D(sc.u, verbose=False)() for sc in self.NPT_systems])
        self.P = self.NPT_systems[0].P # atm
        self.n_mol = self.NPT_systems[0].n_mol
        assert all([item.P == self.P for item in self.NPT_systems]), 'pressure not the same in all NPT datasets'
        assert all([item.n_mol == self.n_mol for item in self.NPT_systems]), 'number of molecules not the same in all NPT datasets'
        assert all([item.N == self.NPT_systems[0].N for item in self.NPT_systems]), 'number of atoms not the same in all NPT datasets'
        print(f'n_mol = {self.n_mol}')
        
        self.Ts = np.array([item.T for item in self.NPT_systems]) # K
        assert self.Tref in self.Ts, 'no NPT dataset found/provided (init arg #5), at the provided Tref (init arg #1)'
        assert len(np.unique(self.Ts)) == len(self.Ts), 'duplicated dataset(s) provided (init arg #5)'
        self.n_temperatures = len(self.Ts)
        sort_by_increasing_T = np.argsort(self.Ts)
        self.Ts = self.Ts[sort_by_increasing_T]
        self.datasets_converged = self.datasets_converged[sort_by_increasing_T]
        self.NPT_systems = dict(zip(self.Ts, [self.NPT_systems[i] for i in sort_by_increasing_T]))
        self.paths_datasets_NPT = np.array(self.paths_datasets_NPT)[sort_by_increasing_T]

        gR_ = lambda _bool : ['R','g'][np.array(_bool).astype(np.int32)]
        print('datasets_converged \n'+''.join([f'{a}K : {color_text_(b,gR_(b))} \n' for a,b in zip(self.Ts, self.datasets_converged )]))

        self.ind_ref = np.where(self.Ts == self.Tref)[0][0]
        print(f'Tref = {self.Tref} (self.ind_ref: {self.ind_ref})')

        ## ## ## ## 
        ## organising the FE through which the final FE output curve will pass at Tref

        if self.Tref_box is not None:
            assert self.Tref_box.shape == (3,3)
            print('Isotropic: Gibbs FEs computed are based on isotropic fluctuations of the box \n')
            self.bT_to_reduced_PV_ = self.PV_isotropic_
            self.set_Tref_g_(**self.f2g_correction_params)
            print(f'Gibbs FE at Tref obtained: {self.Tref_g } +/- {self.Tref_SE} kT. This will be used.')
            print(f'This originates from Helmholtz FE at Tref provided: {self.Tref_FE} +/- {self.Tref_SE} kT.')
        else:
            print('Anisotropic: Gibbs FEs computed are based on anisotropic fluctuations of the box \n')
            self.bT_to_reduced_PV_ = self.PV_anisotropic_
            self.Tref_g  = float(self.Tref_FE)
            print(f'Gibbs FE at Tref provided: {self.Tref_FE} +/- {self.Tref_SE} kT. This will be used.')
            self.Tref_f_to_g_correction = None

        ## ## ## ## 
        ## organising things for heat capacity plotting and _test_average_enthalpy_interpolator_

        self.gather_sampled_potential_energies_() # kT
        self.gather_sampled_enthalpies_() # kT
        self.gather_sampled_kinetic_energies_()   # kT

        print('')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## mostly for heat capacity:

    def gather_sampled_potential_energies_(self,):
        self.sampled_potential_energies = {}
        for T in self.Ts:
            self.sampled_potential_energies[T] = self.NPT_systems[T].u # kT

    def gather_sampled_enthalpies_(self,):
        self.sampled_enthalpies = {}
        for T in self.Ts:
            u  = self.NPT_systems[T].u[:,0]                             # (m,) kT
            pv = self.bT_to_reduced_PV_(self.NPT_systems[T].boxes, T=T) # (m,) kT
            self.sampled_enthalpies[T] = u + pv                         # (m,) kT

    def average_sampled_enthalpy_(self, T, m=None):
        if m is None: m = 0
        else: pass
        return self.sampled_enthalpies[T][-m:].mean()

    def gather_sampled_kinetic_energies_(self,):
        '''
        T(t) = (2./(self.n_DOF*CONST_kB)) * K(t) # T in Kelvin, and K(t) in kJ/mol
        K(t) = T(t) / (2./(self.n_DOF*CONST_kB))
             = T(t) * 0.5 * self.n_DOF * CONST_kB ; in kJ/mol
        beta*K(t) = T(t) * 0.5 * self.n_DOF / T   ; in kT
        '''
        self.sampled_kinetic_energies = {}
        for T in self.Ts:
            T_t = self.NPT_systems[T].temperature
            n_DOF = self.NPT_systems[T].n_DOF
            K_t = 0.5 * n_DOF * T_t / T
            self.sampled_kinetic_energies[T] = K_t # kT

    def total_energies_sampled_(self, T):
        return self.sampled_enthalpies[T] + self.sampled_kinetic_energies[T] # kT

    def clear_memory_(self,):
        del self.NPT_systems

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## FF evaluations

    @property
    def ANI(self,):
        if self.Tref_box is not None: return ''
        else:                         return '_ANI'

    def save_evaluations_(self, only_m_evalautions=None):
        name = self.paths_datasets_NPT[0].split('Temp')[0]+'Ts_'+ '_'.join([str(x) for x in self.Ts]) + self.ANI + '_enthalpy_evaluations'
        if only_m_evalautions is not None: name += f'_m{only_m_evalautions}'
        else: pass
        save_pickle_(self.evaluations, name)

    def load_evalautions_(self, only_m_evalautions=None, ani=''):
        '''
        the most unclear function in this file, but if there is a previous file found nothing is computed or defined here

        dont want to reveluate when the relevant evalautions were already done previusly
        checking if there already is a file that has at least the information that is asked in the args
        '''
        KEY = ani + '_enthalpy_evaluations'

        _name = self.paths_datasets_NPT[0].split('Temp')[0]+'Ts_' # 
        name = None # stays none if no file that has all self.Ts
        m = None    # stays none if not enough data in the found files, and if no file found
        
        possible_files = glob.glob(_name+'*'+KEY+'*')
        for file in possible_files:
            _Ts_in_file = file.split('Ts')[1].split(ani+'_enthalpy')[0][1:].split('_')
            _Ts_in_file = [float(x) for x in  _Ts_in_file if x!='ANI']
            if set(_Ts_in_file) & set(self.Ts.tolist()) == set(self.Ts.tolist()):
                name = file.split(ani+'_enthalpy')[0]+KEY
                if only_m_evalautions is not None:
                    try: 
                        _m = int(file.split(KEY)[1][2:]) 
                        if _m >= only_m_evalautions:
                            m = _m
                    except: pass
                else: pass
            else: pass
        
        if name is not None:
            if only_m_evalautions is not None:
                if m is not None:
                    assert m >= only_m_evalautions # just incase
                else:
                    only_m_evalautions = None
            else: pass
        else: 
            name = _name + '_'.join([str(x) for x in self.Ts]) + KEY

        if only_m_evalautions is not None: name += f'_m{m}'
        else: pass

        # another patch
        if ani=='' and 'ANI' in name: name = None
        else: pass

        try: 
            self.evaluations = load_pickle_(name)
            print(f'file name imported for self.evaluations: {name}')
            return True
        except: 
            print(f'tried, but could not find file: {name}')
            return False

    def compute_enthalpy_at_Ti_on_data_from_Tj_(self, Ti:float, Tj:float, m:int=None):
        if m is not None:
            r_j = self.NPT_systems[Tj].xyz[-m:]   # (m,N,3)
            b_j = self.NPT_systems[Tj].boxes[-m:] # (m,3,3)
            if len(b_j) != m: print(f'!! warning: sample size m={m} not avilable in dataset with T={Tj}')
            else: pass
        else:
            r_j = self.NPT_systems[Tj].xyz        # (?,N,3)
            b_j = self.NPT_systems[Tj].boxes      # (?,3,3)

        print(f'evalauting {len(b_j)} samples from T={Tj} at T={Ti}')

        pv = self.bT_to_reduced_PV_(b_j, Ti)                 # kT
        u  = self.NPT_systems[Ti].u_( r_j, b=b_j ).flatten() # kT

        return u + pv # kT
    
    ''' TODO: add this back in to the current version, so temperatures can be added without re-evaluating previous evalautions.
        # this needs to be changed first, probably add it as arg (path_incomplete_evaluation) in the next function.

        def fill_missing_evaluations_(self, path_incomplete_evaluations:str, m=None, save=True):
            """
            path_incomplete_evaluations : carefully specified by hand

            currently a way to add more temperatures without re-evaluating all other data again
            a better way would be automatically adding if missing (TODO)
            """
            self.evaluations = load_pickle_(path_incomplete_evaluations)
            KEYS = self.evaluations.keys()

            for i in range(self.n_temperatures):
                T_i = self.Ts[i]
                beta_i = 1 / (CONST_kB*T_i)
                for j in range(self.n_temperatures):
                    T_j = self.Ts[j]

                    if m is not None:
                        r_j = self.NPT_systems[T_j].xyz[-m:]   # (m,N,3)
                        b_j = self.NPT_systems[T_j].boxes[-m:] # (m,3,3)
                        if len(b_j) != m: print(f'! warning: sample size m={m} not avilable in dataset with T={T_j}')
                        else: pass
                    else:
                        r_j = self.NPT_systems[T_j].xyz   # (m,N,3) ; m = all datapoints
                        b_j = self.NPT_systems[T_j].boxes # (m,3,3) ; m = all datapoints

                    V_j = np.linalg.det(b_j)               # (m,)

                    PV_i_reduced = self.conversion_factor * beta_i * self.P * V_j

                    if (T_i, T_j) in KEYS: pass
                    else:
                        print(f'adding {(T_i, T_j)} ; m = {len(PV_i_reduced)}')
                        enthalpy_i_on_data_j = self.NPT_systems[T_i].u_( r_j, b=b_j ).flatten() + PV_i_reduced
                        self.evaluations[(T_i, T_j)] = enthalpy_i_on_data_j

            if save:
                self.save_evaluations_(only_m_evalautions=m) # already verbose
            else: pass
    '''

    def compute_all_evaluations_(self, m=None, try_loading_from_save=True, save=True):
        if m is not None:
            assert all([len(self.NPT_systems[T].u)>=m for T in self.Ts]), f'at least one dataset does not have {m} datapoints'
        else:
            m = np.min([len(self.NPT_systems[T].u) for T in self.Ts])
            print('')
            print(f'#### {m} points per state, can, or will be involved in enthalpy evaluations ####')
            print('')
        
        sign = None
        if try_loading_from_save:
            if self.load_evalautions_(m, ani=self.ANI): 
                print('try_loading_from_save : file found.')
            else: 
                if self.load_evalautions_(m, ani=''):
                    sign =  1.0
                    print('isotropic evalautions found, these will be converted and saved in a sperate file')
                elif self.load_evalautions_(m, ani='_ANI'):
                    sign = -1.0
                    print('anisotropic evalautions found, these will be converted and saved in a sperate file')
                else:
                    self.evaluations = {}
                    print(f'try_loading_from_save : file not found; running compute_all_evaluations_ with m = {m}')
        else:  
            self.evaluations = {}
            print(f'running compute_all_evaluations_ with m = {m}')

        keys = self.evaluations.keys()

        # another patch
        if sign is not None: keys = list(set(list([x for x in keys])) - set(['enthalpies converted']))
        else: pass
        ''' TODO (more book-keeping):
        an idea how to simplify the way self.evaluations array is managed:
            let:
                self.evaluations['u'][key_ij] = u_i(r_j) # the only expensive part (cannot be done with cut datasets)
                self.evaluations['pv'][key_ij] = beta_i * PV_j # without lad_j
                self.evaluations['ladj'][key_ij]  = ladj_j
            # although there is only one expensive part, for indexing (on the third-level) to match, better carry these together after computed once.

            if one of the top-level keys is missing, it gets added and self.evaluations resaved
            if one of the top-level keys is not needed, its not requested and not used

            if a second-level key (key_ij) is missing, it gets added (TODO above)
            if there are redundant key_ij not requested, they are not used

            # I will do this later, after Nov., probably in December.
            For now, seems like the patches are holding it away from errors at least.

            Convert all perviously saved files to the new format when settled on the simplest approach. 
                
            This text/note gets removed when its done.
        '''

        any_changes = False

        for Ti in self.Ts:
            for Tj in self.Ts:
                key_ij = (Ti, Tj)

                if key_ij in keys: 
                    if sign is not None:
                        self.check_convert_enthalpy_(Ti, Tj, m=m, sign=sign)
                        any_changes = True
                    else: pass
                else: 
                    print(f'adding {key_ij} to self.evaluations')
                    self.evaluations[key_ij] = self.compute_enthalpy_at_Ti_on_data_from_Tj_(Ti, Tj, m=m)
                    any_changes = True

        if save and any_changes: self.save_evaluations_(only_m_evalautions=m) # already verbose
        else: pass

        print('')
        print(f'#### maximum batch size possible for MBAR: {self.maximum_batch_size} points / state ####')
        print('')

    ## ## ## ## 

    def check_convert_enthalpy_(self, Ti, Tj, m=None, sign=None):
        '''
        this is here only for convenience for a case where 
        self.evaluations found are isotropic but the object wants them to be anisotropic,
        or the other way round. Dont want to evalaute the same potential energies again.
        '''
        if 'enthalpies converted' not in self.evaluations.keys():
            # added only once, error later if trying to convert more than once
            self.evaluations['enthalpies converted'] = {}
        else: pass

        key_ij = (Ti, Tj)

        assert key_ij not in self.evaluations['enthalpies converted'].keys(), '! this should not happen more than once'

        if m is not None:
            b_j = self.NPT_systems[Tj].boxes[-m:] # (m,3,3)
            if len(b_j) != m: print(f'!! warning: sample size m={m} not avilable in dataset with T={Tj}')
            else: pass
        else:
            b_j = self.NPT_systems[Tj].boxes      # (?,3,3)

        ladj = self.ladJ_Vto6_(b_j)

        print(f'converting enthalpy {key_ij} in self.evaluations. sign: {sign}')

        self.evaluations[key_ij] = self.evaluations[key_ij] + sign*ladj
        self.evaluations['enthalpies converted'][key_ij] = True # just to add the key

        # checking this was ok:
        m_check = 100
        err = np.abs(self.evaluations[key_ij][-m_check:] - self.compute_enthalpy_at_Ti_on_data_from_Tj_(Ti, Tj, m=m_check)).max()
        assert err < 1e-1, f'there is a likely mismatch, err={err} kT; ladj range, mean: [{np.abs(ladj).min()}, {np.abs(ladj).max()}], {np.abs(ladj).mean()}'

    ## ## ## ## 

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## MBAR fitting 

    def save_mbar_instance_(self, m):
        name = self.paths_datasets_NPT[0].split('Temp')[0]+'Ts_'+ '_'.join([str(x) for x in self.Ts]) + self.ANI + '_mbar_instance_m'+str(m)
        save_pickle_([self.mbar, self.mbar_res], name)

    def load_mbar_instance_(self, m):
        name = self.paths_datasets_NPT[0].split('Temp')[0]+'Ts_'+ '_'.join([str(x) for x in self.Ts]) + self.ANI + '_mbar_instance_m'+str(m)
        self.mbar, self.mbar_res = load_pickle_(name)

    def save_mbar_instance_shuffled_(self, m):
        name = self.paths_datasets_NPT[0].split('Temp')[0]+'Ts_'+ '_'.join([str(x) for x in self.Ts]) + self.ANI + '_mbar_instance_m'+str(m) + f'_shuffled_from_M{self.maximum_batch_size}'
        save_pickle_([self.mbar, self.mbar_res, self.evaluations_subsample_selection_indices], name)

    def load_mbar_instance_shuffled_(self, m):
        name = self.paths_datasets_NPT[0].split('Temp')[0]+'Ts_'+ '_'.join([str(x) for x in self.Ts]) + self.ANI + '_mbar_instance_m'+str(m) + f'_shuffled_from_M{self.maximum_batch_size}'
        self.mbar, self.mbar_res, self.evaluations_subsample_selection_indices = load_pickle_(name)
        assert len(self.evaluations_subsample_selection_indices.keys()) == self.n_temperatures, '! this should not print'

    ## ## ## ##

    @property
    def maximum_batch_size(self,):
        # M >= m
        return np.min([len(self.evaluations[(T, T)]) for T in self.Ts])
    
    def get_inds_select_(self, uii, m):
        if m < self.maximum_batch_size:
            inds_rand = None ; a=0
            print('(...')
            while inds_rand is None:
                inds_rand = find_split_indices_(uii, split_where=m, tol=0.0001, verbose=False)
                a+=1
            inds_select = inds_rand[:m] # error if not found but max recusion length will be reached anyway
            print(f'...): {a}')
        else:
            inds_select = np.arange(self.maximum_batch_size)
        return inds_select

    def _set_evaluations_subsample_selection_indices_SHUFFLED_(self, m):
        assert self.maximum_batch_size >= m
        self.evaluations_subsample_selection_indices = {}
        for Ti in self.Ts:
            self.evaluations_subsample_selection_indices[Ti]  = self.get_inds_select_(self.evaluations[(Ti,Ti)], m=m)

    def _set_evaluations_subsample_selection_indices_basic_(self, m):
        max_batch_size = int(self.maximum_batch_size)
        assert max_batch_size >= m
        self.evaluations_subsample_selection_indices = {}
        for Ti in self.Ts:
            ALL_indices = np.arange(len(self.evaluations[(Ti, Ti)]))
            self.evaluations_subsample_selection_indices[Ti] = np.array(ALL_indices[-m:])

    def set_evaluations_subsampled_(self,):
        self.evaluations_subsampled = {}
        for Tj in self.Ts:
            inds_select_j = self.evaluations_subsample_selection_indices[Tj]
            for Ti in self.Ts:
                key_ij = (Ti, Tj)
                enthalpy_i_on_data_j = self.evaluations[key_ij]
                #print(key_ij, enthalpy_i_on_data_j.shape, inds_select_j.shape, inds_select_j.min(), inds_select_j.max())
                self.evaluations_subsampled[key_ij] = np.array(enthalpy_i_on_data_j[inds_select_j])

    def compute_MBAR_(self, m=None, rerun=False, save=True, 
                      use_representative_subsets = False, # comment
                     ):
        assert hasattr(self, 'evaluations')

        if m is None: 
            # case 0: using all data, dont need to do anything
            m = self.maximum_batch_size # also x[-0:] = x
            self.evaluations_subsampled = dict(self.evaluations)
            self.SHUFFLED = False
            # case 0: done
        else:
            ''' testing MBAR with less data than previously evalauted in self.evalautions
                two options: use_representative_subsets True or False
                    True : each dataset randomised (self.SHUFFLE := True) to select representative batches of m datapount from each entire dataset
                    False: taking m point from the back of each dataset (self.SHUFFLE := False) 
            '''
            assert self.maximum_batch_size >= m, '!! sample-size m={m} is not available in self.evaluations'
            if use_representative_subsets:
                if rerun:
                    # case 1: v2 asked, and happy to run rew result
                    self._set_evaluations_subsample_selection_indices_SHUFFLED_(m)
                    self.SHUFFLED = True
                    # case 1: ok
                else:
                    # case 2 : v2 asked but trying to load one from before, rerun True if this does not load
                    try:
                        # case 2A: trying to load v2
                        self.load_mbar_instance_shuffled_(m)
                        self.SHUFFLED = True
                        # case 2A: v2 loaded ok, done
                    except:
                        # case 2B = case 1 
                        # rerun forced to True
                        rerun = True
                        # case 1:
                        self._set_evaluations_subsample_selection_indices_SHUFFLED_(m)
                        self.SHUFFLED = True
            else:
                # case 3: v2 not wanted, just do the v1 thing
                self._set_evaluations_subsample_selection_indices_basic_(m)
                self.SHUFFLED = False
                # case 3: ok
            # apply either options
            self.set_evaluations_subsampled_()

        # dont want to save self.evaluations_subsampled each time, just saving indices to recreate it every time loading the same thing

        self.Ns = np.array([self.evaluations_subsampled[(T_i,T_i)].shape[0] for T_i in self.Ts])
        self.Q = np.zeros([self.n_temperatures, self.Ns.sum()])
        assert self.Ns.sum() == self.n_temperatures * m, f'{self.Ns.sum()} != {self.n_temperatures * m}'

        for i in range(self.n_temperatures):
            Ti = self.Ts[i]
            Q_i = []
            for Tj in self.Ts:
                key_ij = (Ti, Tj)
                enthalpy_i_on_data_j = self.evaluations_subsampled[key_ij]
                Q_i.append(enthalpy_i_on_data_j)
            self.Q[i] = np.concatenate(Q_i, axis=0)

        if not rerun:
            try:
                if self.SHUFFLED: self.load_mbar_instance_shuffled_(m)
                else:             self.load_mbar_instance_(m)
                print('rerun : file found.')
            except:
                print(f'rerun : file not found; running compute_MBAR_ with m = {m}')
                self.compute_MBAR_(m=m, rerun=True, save=save)
        else:
            self.mbar = MBAR(self.Q, self.Ns, solver_protocol="robust")
            self.mbar_res = self.mbar.compute_free_energy_differences()
            if save:
                if self.SHUFFLED: self.save_mbar_instance_shuffled_(m)
                else:             self.save_mbar_instance_(m)
            else: pass

        self.mbar_Delta_f  = self.mbar_res['Delta_f']#[0,-1]
        self.mbar_dDelta_f = self.mbar_res['dDelta_f']#[0,-1]
        print('')
        
    '''
    def compute_MBAR_(self, m=0, rerun=False, save=True):
        assert hasattr(self, 'evaluations')

        if m is None: m = 0 # all data; x[-0:] = x
        else: pass

        self.Ns = np.array([self.evaluations[(T_i,T_i)][-m:].shape[0] for T_i in self.Ts])
        self.Q = np.zeros([self.n_temperatures, self.Ns.sum()])

        for i in range(self.n_temperatures):
            Ti = self.Ts[i]
            Q_i = []
            for Tj in self.Ts:
                key_ij = (Ti, Tj)
                enthalpy_i_on_data_j = self.evaluations[key_ij][-m:]

                if len(enthalpy_i_on_data_j) != m and m>0: print(f'! warning: sample-size m={m} not available in self.evaluations[{key_ij}] (dataset sampled at T={Tj})')
                else: pass

                Q_i.append(enthalpy_i_on_data_j)
            self.Q[i] = np.concatenate(Q_i, axis=0)
        
        if not rerun:
            try:
                self.load_mbar_instance_(m)
                print('rerun : file found.')
            except:
                print(f'rerun : file not found; running compute_MBAR_ with m = {m}')
                self.compute_MBAR_(m=m, rerun=True, save=save)
        else:
            self.mbar = MBAR(self.Q, self.Ns, solver_protocol="robust")
            self.mbar_res = self.mbar.compute_free_energy_differences()
            if save: self.save_mbar_instance_(m) # already verbose
            else: pass

        self.mbar_Delta_f  = self.mbar_res['Delta_f']#[0,-1]
        self.mbar_dDelta_f = self.mbar_res['dDelta_f']#[0,-1]
        print('')
    '''

    @property
    def mbar_sample_size(self,):
        return self.mbar.N_k[0] # same for all states at the moment

    @property
    def n_energy_evalautions(self,):
        m = self.mbar_sample_size
        n_states = len(self.mbar.N_k)
        return m * (n_states**2 - n_states)

    @property
    def _mbar(self,):
        return copy.deepcopy(self.mbar)
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## Gibbs FE
    
    def g_(self, T):
        ''' output: g_{crys}(T) for scalar T : continuous gibbs FE estimates as a function of temperature '''
        index = self.ind_ref
        arg = self.Q[index:index+2] * self.Ts[index:index+2].reshape([2,1]) / np.array([self.Tref,T]).reshape([2,1])
        self.res = self.mbar.compute_perturbed_free_energies(arg)

        FE = self.res['Delta_f'][0,1]  + self.Tref_g
        SE = self.res['dDelta_f'][0,1] + self.Tref_SE # in the case of f->g conversion; assuming no error added

        return FE, SE # g_crys, se_crys in kT # *_crys : not divided by n_mol 
    
    @property
    def g(self,):
        ''' g_{crys}(T \in self.Ts) : discrete gibbs FE estimates as a function of temperature '''
        FEs = self.mbar_Delta_f[self.ind_ref] + self.Tref_g
        SEs = self.mbar_dDelta_f[self.ind_ref] + self.Tref_SE
        return FEs, SEs

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## Average enthalpy [using u as the symbol for enthalpy here, not the usual potential energy]

    def av_u_(self,T):

        index = self.ind_ref

        u_ln = self.Q[index:index+2] * self.Ts[index:index+2].reshape([2,1]) / np.array([self.Tref,T]).reshape([2,1])
        
        A_in = self.Q[index:index+2] * self.Ts[index:index+2].reshape([2,1]) / T

        state_map = np.array([[0,1],[0,1]])

        _mbar = copy.deepcopy(self.mbar)
        res = _mbar.compute_expectations_inner(
                A_in,
                u_ln,
                state_map,
                return_theta=True,
                uncertainty_method=None,
                warning_cutoff=1.0e-10,
            )
        av_u = res['observables'][-1]

        return av_u 

    def _test_average_enthalpy_interpolator_(self, m=None):
        # it should pass through all the points where data was actually sampled
        interpolation = np.array([self.av_u_(T) for T in self.Ts])
        truth         = np.array([self.average_sampled_enthalpy_(T, m=m) for T in self.Ts])
        return np.abs(truth - interpolation).max() / self.n_mol

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## Curves: FE, enthalpy [using u as the symbol for enthalpy here, not the usual potential energy]

    def curve_(self, Tmin, Tmax, Tstride=100,  what='g'):
        ''' output : g_{crys}(T) curve between T=Tmin and T=Tmax '''
        if what == 'g':   # abs Gibbs FE
            function_ = lambda T : self.g_(T)
        elif what == 'u': # everage enthalpy (not average energy)
            function_ = lambda T : [self.av_u_(T), 0.0]
        elif what == 's': # entropy = u - g
            print('! curve_ : what="s" may be noisy because what="u" is used (what="u" is often noisy)')
            def function_(T):
                g, se = self.g_(T)
                av_u = self.av_u_(T)
                return av_u - g, se
        else: assert what in ['g', 'u', 's']

        Ts = np.linspace(Tmin, Tmax, Tstride)
        FEs = []
        SEs = []
        for T in Ts:
            FE, SE = function_(T)
            FEs.append(FE)
            SEs.append(SE)
        
        FEs_latt = np.array(FEs) / self.n_mol
        SEs_latt = np.array(SEs) / self.n_mol
        # Ts : grid for plotting (x-axis)

        return FEs_latt, SEs_latt, Ts

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    ## saving/loading final result

    @property
    def path_RES(self,):
        m = self.mbar_sample_size
        return self.paths_datasets_NPT[0].split('Temp')[0]+'Ts_'+'_'.join([str(x) for x in self.Ts])+f'_Tref_{self.Tref}{self.ANI}_RES_m{m}'

    def load_RES_(self, name_RES):
        if self.SHUFFLED: name_RES += '_shuffled'
        else: pass
        return load_pickle_(name_RES)

    def save_RES_(self, name_RES):
        if self.SHUFFLED: name_RES += '_shuffled'
        else: pass
        save_pickle_(self.RES, name_RES)

    def get_result_(self, Tmin=50, Tmax=800, Tstride=500, save=True):
        
        if self.f2g_correction_params == {'version':1, 'bins':40}: key = ''
        else: key = f'_f2g_setting_{self.f2g_correction_params["version"]}_{self.f2g_correction_params["bins"]}'

        name_RES = self.path_RES + f'_{Tmin}_{Tmax}_{Tstride}' + key

        try:
            RES = self.load_RES_(name_RES)
            #print(name_RES)
            #print(self.paths_datasets_NPT[self.ind_ref])
            #print(self.Tref_box)
            #print(f"{RES['ref']['Helmholtz_FE']} vs. {self.Tref_FE}")
            assert RES['ref']['Helmholtz_FE']  == self.Tref_FE
            #print(f"{RES['ref']['SE']} vs. {self.Tref_SE}")
            assert RES['ref']['SE']            == self.Tref_SE
            #print(f"{RES['ref']['Gibbs_FE']} vs. {self.Tref_g}")
            assert RES['ref']['Gibbs_FE']      == self.Tref_g
            self.RES = RES
            print('Found previously saved result with current settings. This is set to self.RES')

        except:
            print(f'Running new result. This will be set to self.RES' + (', and saved' if save else ''))
            RES = {}

            RES['ref']      = {}
            RES['curve']    = {}
            RES['discrete'] = {}

            g, s, grid  = self.curve_( Tmin, Tmax, Tstride=Tstride,  what='g')
            u, _, grid  = self.curve_( Tmin, Tmax, Tstride=Tstride,  what='u')
            RES['curve']['grid']         = grid
            RES['curve']['Gibbs_FE']     = g
            RES['curve']['Gibbs_SE']     = s
            RES['curve']['Enthalpy']     = u

            RES['discrete']['grid']      = self.Ts
            RES['discrete']['converged'] = np.array(self.datasets_converged).astype(np.int32)
            RES['discrete']['Gibbs_FE']  = self.g[0] / self.n_mol
            RES['discrete']['Gibbs_SE']  = self.g[1] / self.n_mol
            m = self.mbar_sample_size
            # Enthalpy : Enthalpy data used for mbar
            RES['discrete']['Enthalpy']          = np.array([self.average_sampled_enthalpy_(T,m=m) for T in self.Ts]) / self.n_mol
            RES['discrete']['Enthalpy_all_data'] = np.array([self.average_sampled_enthalpy_(T) for T in self.Ts])     / self.n_mol

            RES['ref']['Tref']                          = self.Tref
            RES['ref']['Helmholtz_FE']                  = self.Tref_FE
            RES['ref']['SE']                            = self.Tref_SE
            RES['ref']['Helmholtz_to_Gibbs_correction'] = self.Tref_f_to_g_correction # no error from correct assumed
            RES['ref']['Gibbs_FE']                      = self.Tref_g
            RES['ref']['Helmholtz_to_Gibbs_params']     = self.f2g_correction_params
            RES['ref']['n_mol']                         = self.n_mol

            self.RES = RES
            if save: self.save_RES_(name_RES)
            else: pass

        print('')
        
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## heat capacity:

    def cP_(self, Tmin=50, Tmax=800, Tstride=500, include_KE=False):
        if include_KE: book_ = lambda T : self.total_energies_sampled_(T)
        else:          book_ = lambda T : self.sampled_enthalpies[T]

        Ts = np.array(self.Ts)
        Es = np.array([ book_(T).mean()*T*CONST_kB for T in Ts ]) / self.n_mol

        lr = LineFit(Ts[:,np.newaxis], Es[:,np.newaxis])

        Ts_fine_grid = np.linspace(Tmin, Tmax, Tstride)
        Es_fine_grid = lr(Ts_fine_grid[:,np.newaxis])
        cP = np.round(lr.W[0][0], 4)

        print(r'$c_{P} \; / \; \text{kJ} \: \text{mol}^{-1} \text{K}^{-1} \text{molecule}^{-1} $'+f' = {cP}')

        # Es : kJ/mol lattice energy

        return [Ts,Es], [Ts_fine_grid, Es_fine_grid], cP

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

class LineFit:
    def __init__(self,
                X, # (m,dim_x) ; dim_x >= 1
                Y, # (m,dim_y) ; dim_y >= 1
                ):
        self.muX = X.mean(0,keepdims=True)
        self.muY = Y.mean(0,keepdims=True)
        m = X.shape[0]
        self.Cxx = (X-self.muX).T.dot(X-self.muX) / (m-1)
        self.Cyx = (Y-self.muY).T.dot(X-self.muX) / (m-1)
        self.W = self.Cyx.dot(np.linalg.inv(self.Cxx))
    def __call__(self, X):
        # X ; (m,dim_x) ; dim_x >= 1
        return (X-self.muX).dot(self.W) + self.muY

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## two-state BAR for NVT / NPT FE differences between similar macrostates [not included because not yet used in a publication]


