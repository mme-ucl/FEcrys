from O.interface import *
from O.NN.pgm_rb import *

from O.sym import *
from O.MM.Tx import *

from O.MM.bias import *

import shutil

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

'''
work in progress
'''

## ## ## ## ## ## 

cell_to_cell_str_ = lambda cell : ''.join([str(x) for x in cell])

class NEW_PROJECT:

    def __init__(self, name):
        self.molecules_folder = './O/MM/molecules'

        self.name = name

        self.folders = {
        'main' : f'{self.molecules_folder}/{name}',
        'misc' : f'{self.molecules_folder}/{name}/misc',
        'temp' : f'{self.molecules_folder}/{name}/temp',
        'data' : f'{self.molecules_folder}/{name}/data',
        'NPT'  : f'{self.molecules_folder}/{name}/data/NPT',
        }
        for folder in self.folders.values():
            if os.path.exists(folder): pass
            else:
                os.mkdir(folder)
                print(f'created folder: {folder}')
        
        self.files = {
            'single_mol_main' : f'{self.molecules_folder}/{name}/{name}_single_mol.pdb',      # should be 
            'single_mol_misc' : f'{self.molecules_folder}/{name}/misc/{name}_single_mol.pdb', # identical
        }

        try: 
            self.n_atoms_mol = mdtraj.load(self.files['single_mol_main']).n_atoms
        except:
            print('next: please run add_single_molecule_pdb_(PDB_single_mol)')

        self.n_atoms_unitcell = {}

        self.check_create_supercells_(100, checking=True)

        self.files['GAFF_gro'] = f'{self.molecules_folder}/{self.name}/misc/GAFF_{self.name}.gro'
        self.files['GAFF_itp'] = f'{self.molecules_folder}/{self.name}/misc/GAFF_{self.name}.itp'

        self.files['OPLS_gro'] = f'{self.molecules_folder}/{self.name}/misc/OPLS_{self.name}.gro'
        self.files['OPLS_itp'] = f'{self.molecules_folder}/{self.name}/misc/OPLS_{self.name}.itp'

        self.files['TMFF_gro'] = f'{self.molecules_folder}/{self.name}/misc/tmFF_{self.name}.gro'
        self.files['TMFF_itp'] = f'{self.molecules_folder}/{self.name}/misc/tmFF_{self.name}.itp'

    def check_unitcells_(self,):
        list_forms = []
        a_unitcell_found = False
        for item in Path(self.folders['main']).iterdir():
            if 'unitcell' in item.name:
                a_unitcell_found = True
                form = item.name.split('_')[1]
                list_forms.append(form)
                self.files[form+'_unitcell'] = f'{self.molecules_folder}/{self.name}/{self.name}_{form}_unitcell.pdb'
                N = PDB_to_xyz_(self.files[form+'_unitcell']).shape[0]
                n_atoms_unicell = N // self.n_atoms_mol
                assert n_atoms_unicell == N / self.n_atoms_mol, f'unitcell file {self.files[form+"_unitcell"]} with inconsistent number of atoms'
                self.n_atoms_unitcell[form] = n_atoms_unicell
            else: pass
        self.list_forms = list(set(list_forms))

        if a_unitcell_found: print(f'list_forms:\n {self.list_forms}')
        else:                print('no unit cells found')

        return a_unitcell_found
    
    def add_single_molecule_pdb_(self, PDB_single_mol:str):
        
        traj = mdtraj.load(PDB_single_mol)
        traj.save_pdb(f'{self.molecules_folder}/{self.name}/temp/{self.name}_single_mol_temp.pdb')

        self.n_atoms_mol = traj.n_atoms
        print(f'n_atoms_mol = {self.n_atoms_mol}')

        # convert multiframe pdb (default from mdtraj) to single frame
        pdb_in = open(f'{self.molecules_folder}/{self.name}/temp/{self.name}_single_mol_temp.pdb', 'r')
        pdb_out = open(self.files['single_mol_misc'], 'w')
        
        for line in pdb_in:
            if 'TER' in line or 'ENDMDL' in line: pass
            else: pdb_out.write(line)
        pdb_out.close()

        shutil.copy2(self.files['single_mol_misc'], self.files['single_mol_main'])   

        assert mdtraj.load(self.files['single_mol_main']).n_atoms == self.n_atoms_mol
        
        # check if a unitcell was added:
        if self.check_unitcells_(): pass
        else: print('next: please run add_unitcell_(PDB_unitcell, name_of_the_form)')

    def add_unitcell_(self, PDB_unitcell:str, form:str):
        self.files[form+'_unitcell_temp'] = f'{self.molecules_folder}/{self.name}/{self.name}_{form}_unitcell_temp.pdb'
        self.files[form+'_unitcell'] = f'{self.molecules_folder}/{self.name}/{self.name}_{form}_unitcell.pdb'

        if os.path.exists(self.files[form+'_unitcell']): 
            print(f'add_unitcell_() : output file already found')

        else:
            traj = mdtraj.load(PDB_unitcell)
            traj.save_pdb(self.files[form+'_unitcell_temp'])
            reorder_atoms_unitcell_(PDB     = self.files[form+'_unitcell_temp'],
                                    PDB_ref = self.files['single_mol_main'],
                                    n_atoms_mol=self.n_atoms_mol,
                                    )
            shutil.copy2(f'{self.molecules_folder}/{self.name}/{self.name}_{form}_unitcell_temp_reordered.pdb', 
                        self.files[form+'_unitcell'])  

            print(f"file saved: {self.files[form+'_unitcell']}")

        if self.check_create_supercells_(max_n_mol = 100, checking=True): pass
        else: print('next : please run check_create_supercells_(max_n_mol)')

    ## ## ## ## 

    def form_cell_to_ideal_supercell_PDB_(self, form, cell):
        return f'{self.molecules_folder}/{self.name}/{self.name}_{form}_unitcell_cell'+cell_to_cell_str_(cell)+'.pdb'

    def check_create_supercells_(self, max_n_mol = 40, checking=False):
        self.check_unitcells_()
        self.supercell_details = {}
        a_supercell_found = False
        for form in self.list_forms:
            self.supercell_details[form] = {}
            unitcell = self.files[form+'_unitcell']
            supercells = get_unitcell_stack_order_(PDB_to_box_(unitcell),
                          n_mol_unitcell=self.n_atoms_unitcell[form], top_n=100)
            for n_mol in  supercells.keys():
                if n_mol <= max_n_mol:
                    cell = supercells[n_mol]
                    supercell = self.form_cell_to_ideal_supercell_PDB_(form, cell)
                    if os.path.exists(supercell):
                        a_supercell_found = True
                    else:
                        if checking: pass 
                        else: supercell_from_unitcell_(unitcell, cell=cell)
                    self.supercell_details[form][n_mol] = cell
                    cell_str = cell_to_cell_str_(cell)
                    self.files[form+f'_ideal_supercell_{n_mol}'] =  f'{self.molecules_folder}/{self.name}/{self.name}_{form}_unitcell_cell{cell_str}.pdb'
                else: pass

        return a_supercell_found
    
    @property
    def list_n_mol_recommended(self,):
        sets = [set([x for x in value.keys()]) for value in self.supercell_details.values()]
        list_n_mol_recommended  = sorted(list(sets[0].intersection(*sets[1:])))

        if len(list_n_mol_recommended) > 0:
            # print(f'self.list_n_mol_recommended = {list_n_mol_recommended}')
            return list_n_mol_recommended
        else: 
            print('Note: there is not one single n_mol that is applicable to all supercells')
            return None

    ## ## ## ## 

    def check_ideal_supercells_(self, min_n_mol=1, max_n_mol=40):
        boxes = []
        for form in self.list_forms:
            list_n_mol_possible = [x for x in self.supercell_details[form].keys() if x <= max_n_mol]
            list_n_mol = []
            for n_mol in list_n_mol_possible:
                if min_n_mol <= n_mol <=  max_n_mol:
                    cell = self.supercell_details[form][n_mol]
                    cell_str = cell_to_cell_str_(cell)
                    PDB_supercell = self.files[form+f'_ideal_supercell_{n_mol}']
                    boxes.append(PDB_to_box_(PDB_supercell))
                    list_n_mol.append(n_mol)
                else:
                    pass
                    #print(f'form {form}, no supercell with : (min_n_mol={min_n_mol}) <= (n_mol={ n_mol}) <= (max_n_mol={max_n_mol})')
            print(f'form {form} with considered list_n_mol = {list_n_mol}')
        boxes = np.stack(boxes)
        cutoff_max = np.min([boxes[...,i,i].min() for i in range(3)]) * 0.5
        print(f'maximum PME cutoff (d_max) possible with current set of list_n_mol selections: {cutoff_max.round(3)}nm')
        #print(f'PME cutoff used: {PME_cutoff}nm')

    ## ## ## ## 

    def create_NPT_subfolder_(self, NPT_subfolder_name):
        os.mkdir(f'./O/MM/molecules/{self.name}/data/NPT/{NPT_subfolder_name}')

    def FFname_form_cell_T_to_eqm_supercell_PDB_(self, 
                                                 FFname, form, cell, T, key='',
                                                 ):
        return f'{self.molecules_folder}/{self.name}/{self._name}_{FFname.lower()}_equilibrated_form_{form}_Cell_{cell_to_cell_str_(cell)}_Temp_{T}{key}.pdb'

    ## ## ## ## 
    ## ## ## ## 

    def check_parametrise_with_GAFF_automatic_(self, test=False):
        sc = SingleComponent(PDB = self.files['single_mol_main'],
                    name = self.name,
                    n_atoms_mol = self.n_atoms_mol,
                    FF_class = GAFF,
                    )
        self.add_single_molecule_pdb_(self.files['single_mol_main']) # to remove the last lines of the pdb file added in the above step
        print('ok')

        if test:
            print('(test = True) : testing if there are any errors:')
            sc.initialise_system_()
            sc.initialise_simulation_()
        else: pass

    def check_parametrise_with_GAFF_(self, gro_file=None, itp_file=None, test=False):
        files_found = os.path.exists(self.files['GAFF_gro']) and os.path.exists(self.files['GAFF_itp'])
        if None in [gro_file, itp_file]:
            assert files_found, 'please provide files: gro_file, itp_file; for the GAFF force field'
        else:
            if files_found: pass
            else:
                shutil.copy2(gro_file, self.files['GAFF_gro'])
                shutil.copy2(itp_file, self.files['GAFF_itp'])
        print('ok')

        if test:
            print('(test = True) : testing if there are any errors:')
            sc = SingleComponent(PDB = self.files['single_mol_main'],
                        name = self.name,
                        n_atoms_mol = self.n_atoms_mol,
                        FF_class = GAFF_general,
                        )
            sc.initialise_system_()
            sc.initialise_simulation_()
        else: pass

    def check_parametrise_with_OPLS_(self, gro_file=None, itp_file=None, test=False):
        files_found = os.path.exists(self.files['OPLS_gro']) and os.path.exists(self.files['OPLS_itp'])
        if None in [gro_file, itp_file]:
            assert files_found, 'please provide files: gro_file, itp_file; for the OPLS force field'
        else:
            if files_found: pass
            else:
                shutil.copy2(gro_file, self.files['OPLS_gro'])
                shutil.copy2(itp_file, self.files['OPLS_itp'])
        print('ok')

        if test:
            print('(test = True) : testing if there are any errors:')
            sc = SingleComponent(PDB = self.files['single_mol_main'],
                        name = self.name,
                        n_atoms_mol = self.n_atoms_mol,
                        FF_class = OPLS_general,
                        )
            sc.initialise_system_()
            sc.initialise_simulation_()
        else: pass

    def check_parametrise_with_TMFF_(self, gro_file=None, itp_file=None, test=False):
        files_found = os.path.exists(self.files['TMFF_gro']) and os.path.exists(self.files['TMFF_itp'])
        if None in [gro_file, itp_file]:
            assert files_found, 'please provide files: gro_file, itp_file; for the TMFF force field'
        else:
            if files_found: pass
            else:
                shutil.copy2(gro_file, self.files['TMFF_gro'])
                shutil.copy2(itp_file, self.files['TMFF_itp'])
        print('ok')

        if test:
            print('(test = True) : testing if there are any errors:')
            sc = SingleComponent(PDB = self.files['single_mol_main'],
                        name = self.name,
                        n_atoms_mol = self.n_atoms_mol,
                        FF_class = tmFF,
                        )
            sc.initialise_system_()
            sc.initialise_simulation_()
        else: pass

class PIPELINE(NEW_PROJECT):
    def __init__(self, name):
        '''
        name = name of the molecule
        '''
        super().__init__(name)

    def set_FF_(self, FF_class):
        '''
        choosing force field
        '''
        list_possible_FF_class = [GAFF, GAFF_general, OPLS_general, tmFF]
        print('checking forcefield is set up')
        if FF_class == list_possible_FF_class[0]:
            self.check_parametrise_with_GAFF_automatic_()
        elif FF_class == list_possible_FF_class[1]:
            self.check_parametrise_with_GAFF_()
        elif FF_class == list_possible_FF_class[2]:
            self.check_parametrise_with_OPLS_()
        elif FF_class == list_possible_FF_class[3]:
            self.check_parametrise_with_TMFF_()
        else: assert FF_class in list_possible_FF_class, '! please check the FF_class is supported'
        self.FF_class = FF_class

    def set_d_cut_(self, d_cut):
        '''
        choosing non-bonded cutoff distance in the force field
        '''
        self.d_cut = d_cut
        self.KEY = f'_lr{self.d_cut}'

    def set_ind_rO_(self, ind_rO):
        '''
        choosing cartesian block atom 1
        '''
        self.ind_rO = ind_rO

    def set_option_(self, option):
        '''
        choosing cartesian block atoms 2 and 3
        '''
        self.option = option

    def set_symmetry_adjustment_(self, method):
        '''
        method should have 
        args:
            path_dataset, form, T
        kwargs:
            checking
        '''
        self.symmetry_adjustment_ = method

    #######################
    ## copy from project_settings.py

    def NPT_(   self, 
                _list_forms : list,
                _Ts : list,
                _list_n_mol : dict,
                NPT_subfolder = '',

                T_init = None, # faster to warm up a supercell that was extracted from a warm NPT simulation
                n_frames_warmup = 50000,
                n_frames_sample = 200000,
                n_frames_stride = 50,
                save_equilibrated_supercell = True, # turn off when rerunning a dataset for which NVT was already ran and used, to keep consistent PDB for that dataset.

                checking = False, # just to check the files were not lost
                checking_check_only_path = True,

                checking_resave_datasets_with_CUT_xyz = False, # for storage
                checking_try_plotting_equlibrated_cell = False,

                overwrite = False,
                rerun_unconverged = False,
                
            ):
        assert hasattr(self, 'FF_class'),   'please set_FF_'
        assert hasattr(self, 'd_cut'),      'please set_d_cut_'
        assert hasattr(self, 'ind_rO'),     'please set_ind_rO_'

        PATH = f'{self.molecules_folder}/{self.name}'
        FF_name = self.FF_class.FF_name

        average_energies = {}
        for form in _list_forms:
            for T in _Ts:
                for n_mol in _list_n_mol[form]:

                    cell = self.supercell_details[form][n_mol] ; cell_str = cell_to_cell_str_(cell)
                    verbose_message = f'form={form}, cell={cell} (n_mol={n_mol}), T={T}, FF={FF_name}'

                    if T_init is not None:
                        PDB_initial = self.FFname_form_cell_T_to_eqm_supercell_PDB_(FF_name, form=form, cell=cell, T=T_init, key=self.KEY)
                        assert os.path.exists(PDB_initial), '! no equilibrated supercell found for: form={form}, cell={cell} (n_mol={n_mol}), T={T_init}, FF={FF_name}'
                        if not checking: print(f'NPT simulation initialised from supercell that was already equilibrated at T={T_init}')
                        else: pass
                    else:
                        PDB_initial = self.form_cell_to_ideal_supercell_PDB_(form, cell)
                        assert os.path.exists(PDB_initial), f'! no ideal supercell found for: form={form}, n_mol={n_mol}'
                        if not checking: print(f'NPT simulation initialised from ideal supercell')
                        else: pass

                    path_NPT_dataset = f'{PATH}/data/NPT/{NPT_subfolder}/{self.name}_{FF_name.lower()}_NPT_dataset_form_{form}_Cell_{cell_str}_Temp_{T}' + self.KEY
                    path_equilibrated_supercell = self.FFname_form_cell_T_to_eqm_supercell_PDB_(FF_name, form, cell, T, key=self.KEY)

                    if checking:
                        print('checking NPT simulation:', verbose_message)
                        print(f'dataset: {path_NPT_dataset}')
                        print('path exists:', os.path.exists(path_NPT_dataset))

                        print(f'checking path_equilibrated_supercell: {path_equilibrated_supercell}')
                        print('path exists:', os.path.exists(path_equilibrated_supercell))

                        if checking_check_only_path: pass
                        else:
                            dataset = load_pickle_(path_NPT_dataset)
                            u = np.array(dataset['MD dataset']['u']) / n_mol
                            u_mean = u.mean()
                            
                            converged = TestConverged_1D(u)() ; average_energies[(form, T, n_mol)] = [u_mean, converged]
                            
                            print(f'average lattice potential energy: {u_mean} kT')
                            print(f'number of samples: {u.shape[0]}')

                            if checking_resave_datasets_with_CUT_xyz:
                                print('shrinking down the dataset file: (full file backed up seperately)')
                                dataset['MD dataset']['xyz'] = dataset['MD dataset']['xyz'][:50]
                                save_pickle_(dataset, name=path_NPT_dataset)
                            else: pass

                            if checking_try_plotting_equlibrated_cell:
                                try:
                                    path_ideal_supercell = self.form_cell_to_ideal_supercell_PDB_(form, cell)
                                    plot_box_lengths_angles_histograms_(dataset['MD dataset']['b'], 
                                                                        b0 = PDB_to_box_(path_ideal_supercell), 
                                                                        b1 = PDB_to_box_(path_equilibrated_supercell),
                                                                        )
                                except: print('could not find equilibrated supercell extracted from this NPT dataset')
                            else: pass
                            del dataset
                    else: pass

                    if not checking or (rerun_unconverged and not average_energies[(form, T, n_mol)][1]):
                        print('starting NPT simulation:', verbose_message)
                        print(f'dataset: {path_NPT_dataset}')

                        if rerun_unconverged: _overwrite = True # all this rerun_unconverged variations is becase of 075 with non-opls (..)
                        else:  _overwrite = bool(overwrite)

                        if os.path.exists(path_NPT_dataset) and not _overwrite:
                            print(f'overwrite = {_overwrite} and dataset found; skipping')
                        else:
                            #if 1==1:
                            import logging
                            logging.basicConfig(level=logging.ERROR)

                            try:
           
                                assert os.path.exists(f'{PATH}/data/NPT/{NPT_subfolder}')
                                print(color_text_(f'\n starting trajectory, this will be saved into folder {PATH}/data/NPT/{NPT_subfolder} \n', 'I'))

                                if rerun_unconverged:
                                    sc = SingleComponent.initialise_from_save_(path_NPT_dataset)
                                    if overwrite: sc.set_arrays_blank_()
                                    else: pass
                                    # sc.args_initialise_object['PDB'] = form_cell_to_ideal_supercell_PDB_(form, cell)

                                else:
                                    sc = SingleComponent(PDB=PDB_initial, n_atoms_mol=self.n_atoms_mol, name=self.name, FF_class=self.FF_class)
                                    sc.initialise_system_(PME_cutoff=self.d_cut, nonbondedMethod=app.PME)
                                    sc.initialise_simulation_(timestep_ps = 0.002, P = 1, T = T) # ps, atm, K
                                    sc.simulation.step(n_frames_warmup * n_frames_stride)
                                
                                sc.run_simulation_(n_frames_sample, n_frames_stride)

                                sc.save_simulation_data_(path_NPT_dataset)

                                u = np.array(sc.u) / n_mol
                                converged = TestConverged_1D(u)()
                                u_mean = u.mean()
                                print(f'average lattice potential energy: {u_mean} kT')
                                average_energies[(form, T, n_mol)] = [u_mean, converged]

                                if save_equilibrated_supercell:
                                    print('equilibrated supercell PDB is being extracted and saved:')
                                    sc._xyz = tidy_crystal_xyz_(sc.xyz, sc.boxes, n_atoms_mol=sc.n_atoms_mol, ind_rO=self.ind_rO)
                                    index = get_index_average_box_automatic_(sc.boxes)
                                    r1 = sc.xyz[index] ; b1 = sc.boxes[index]
                                    r1 = sc.minimise_xyz_(r1, b=b1, verbose=True)
                                    sc.save_pdb_(r1, b=b1, name=path_equilibrated_supercell)
                                else: pass
                                
                                del sc
                                print('finished NPT simulation:', verbose_message)
                            #else:
                            except:
                                logging.exception(f'equilibration problem: {verbose_message}')
                                print(f'equilibration problem: {verbose_message}')
                    else: pass

                    print('\n ################################################################################## \n')
        print('done')
        return average_energies

    def NVT_(   self,
                _list_forms : list,
                _Ts : list,
                _list_n_mol : dict,

                n_frames_warmup = 5000,
                n_frames_sample = 500000,
                n_frames_stride = 50,

                checking = False,
                checking_check_only_path = True,
                checking_resave_datasets_with_CUT_xyz = False, # for storage
                checking_check_box_match_pdb = True,

                overwrite = False, # checking False
            ):
        assert hasattr(self, 'FF_class'),   'please set_FF_'
        assert hasattr(self, 'd_cut'),      'please set_d_cut_'

        PATH = f'{self.molecules_folder}/{self.name}'
        FF_name = self.FF_class.FF_name

        average_energies = {}

        for form in _list_forms:
            for T in _Ts:
                for n_mol in _list_n_mol[form]:
                    
                    cell = self.supercell_details[form][n_mol] ; cell_str = cell_to_cell_str_(cell)
                    verbose_message = f'form={form}, cell={cell} (n_mol={n_mol}), T={T}, FF={FF_name}'
                
                    path_equilibrated_supercell = self.FFname_form_cell_T_to_eqm_supercell_PDB_(FF_name, form, cell, T, key=self.KEY)
                    path_NVT_dataset = f'{PATH}/data/{self.name}_{FF_name.lower()}_NVT_dataset_form_{form}_Cell_{cell_str}_Temp_{T}' + self.KEY

                    if checking:
                        print('checking NVT simulation:', verbose_message)
                        print(f'dataset: {path_NVT_dataset}')
                        print('path exists:', os.path.exists(path_NVT_dataset))

                        if checking_check_only_path: pass
                        else:
                            dataset = load_pickle_(path_NVT_dataset)
                            u = np.array(dataset['MD dataset']['u']) / n_mol
                            u_mean = u.mean()

                            converged = TestConverged_1D(u)() ; average_energies[(form, T, n_mol)] = [u_mean, converged]

                            print(f'average lattice potential energy : {u_mean} kT')
                            print(f'number of samples: {u.shape[0]}')

                            if checking_resave_datasets_with_CUT_xyz:
                                print('shrinking down the dataset file: (full file backed up seperately)')
                                dataset['MD dataset']['xyz'] = dataset['MD dataset']['xyz'][:50]
                                save_pickle_(dataset, name=path_NVT_dataset)
                            else: pass

                            if checking_check_box_match_pdb:
                                try:
                                    boxes_constant = dataset['MD dataset']['b']
                                    err = np.abs(PDB_to_box_(path_equilibrated_supercell)[np.newaxis,...] -  boxes_constant).max()
                                    print(f'checking_check_box_match_pdb; err = {err}')
                                except: print('could not find equilibrated supercell related to this NVT dataset')
                            else: pass

                            del dataset
                    else:
                        print('starting NVT simulation:', verbose_message)
                        print(f'dataset: {path_NVT_dataset}')

                        if os.path.exists(path_NVT_dataset) and not overwrite:
                            print(f'overwrite = {overwrite} and dataset found; skipping')
                        else:

                            print(color_text_(f'\n starting trajectory, this will be saved into folder {PATH}/data \n', 'I'))
            
                            sc = SingleComponent(PDB = path_equilibrated_supercell, 
                                                 n_atoms_mol = self.n_atoms_mol, 
                                                 name = self.name,
                                                 FF_class = self.FF_class)
                            sc.initialise_system_(PME_cutoff=self.d_cut, nonbondedMethod=app.PME)
                            sc.initialise_simulation_(timestep_ps = 0.002, P = None, T = T)
                            sc.simulation.step(n_frames_warmup * n_frames_stride)
                            sc.run_simulation_(n_frames_sample, n_frames_stride)
                            sc.save_simulation_data_(path_NVT_dataset)

                            u = np.array(sc.u) / n_mol
                            converged = TestConverged_1D(u)()
                            u_mean = u.mean()
                            print(f'average lattice potential energy: {u_mean} kT')
                            average_energies[(form, T, n_mol)] = [u_mean, converged]

                            del sc
                            print('finished NVT simulation:', verbose_message)

                    print('\n ################################################################################## \n')
        print('done')
        return average_energies

    def PGM_(   self,
                _list_forms : list,
                _Ts : list,
                _list_n_mol : dict,

                n_training_batches = 20000,
                learning_rate = 0.001,
                n_layers = 4,
                n_att_heads= 4,
                symmetry_adjust_MD_data = True,

                checking = False, # just to check the files were not lost
                checking_load_datasets = False,

                PGM_key = '',
                NVT = True,
                NPT_subfolder = '',

                truncate_data : int = None,
            ):
        assert hasattr(self, 'FF_class'),   'please set_FF_'
        assert hasattr(self, 'd_cut'),      'please set_d_cut_'
        assert hasattr(self, 'ind_rO'),     'please set_ind_rO_'
        assert hasattr(self, 'option'),     'please set_option_'
        if symmetry_adjust_MD_data:
            assert hasattr(self, 'symmetry_adjustment_'), 'please run set_symmetry_adjustment_ to enable symmetry_adjust_MD_data'
            symmetry_adjustment_ = self.symmetry_adjustment_
        else: symmetry_adjustment_ = lambda *args, **kwargs : ''

        PATH = f'{self.molecules_folder}/{self.name}'
        FF_name = self.FF_class.FF_name
        evaluation = {}

        for form in _list_forms:
            for T in _Ts:
                for n_mol in _list_n_mol[form]:

                    cell = self.supercell_details[form]['supercells'][n_mol] ; cell_str = cell_to_cell_str_(cell)
                    verbose_message = f'form={form}, cell={cell} (n_mol={n_mol}), T={T}, FF={FF_name}'
                    n_mol_unitcell = int(self.supercell_details[form]['n_mol_unitcell'])

                    if NVT: path_dataset = f'{PATH}/data/{self.name}_{FF_name.lower()}_NVT_dataset_form_{form}_Cell_{cell_str}_Temp_{T}' + self.KEY
                    else:   path_dataset = f'{PATH}/data/NPT/{NPT_subfolder}/{self.name}_{FF_name.lower()}_NPT_dataset_form_{form}_Cell_{cell_str}_Temp_{T}' + self.KEY

                    sr_KEY = symmetry_adjustment_(path_dataset, form=form, T=T, checking=checking)

                    if NVT: PGM_instance_name = f'{self.name}_{FF_name.lower()}_form_{form}_Cell_{cell_str}_Temp_{T}' + self.KEY + sr_KEY + PGM_key
                    else:   PGM_instance_name = f'rb_{self.name}_{FF_name.lower()}_form_{form}_Cell_{cell_str}_Temp_{T}' + self.KEY + sr_KEY + PGM_key

                    print(f'PGM instance name: {PGM_instance_name}')

                    if checking and not os.path.exists(path_dataset + sr_KEY): pass
                    else: path_dataset += sr_KEY 

                    if not os.path.exists(path_dataset):
                        print('!! dataset related this training run was not found')
                        print(path_dataset)
                        path_dataset = 0
                        assert checking, '(checking = False) = training; this dataset is needed'
                    else: pass

                    if checking: 
                        if checking_load_datasets: pass
                        else: path_dataset = 0
                    else: pass
                    if NVT:
                        nn = NN_interface_sc_multimap(
                                        name = PGM_instance_name,
                                        paths_datasets = [path_dataset,],
                                        running_in_notebook = False,
                                        training = not checking,
                                        model_class = PGMcrys_v1,
                                        )
                    else:
                        nn = NN_interface_sc_multimap_rb(name = PGM_instance_name,
                                                        paths_datasets=[path_dataset,],
                                                        running_in_notebook = False,
                                                        training = not checking,
                                                        )
                        
                    if checking:
                        print('loading PGM instance:', verbose_message)
                        nn.load_misc_()
                        nn.solve_BAR_using_pymbar_(rerun=False)
                        evaluation[(form, T, n_mol)] = nn

                        time_min = nn.training_time
                        print(f'training and evaluation duration: {time_min // 60} hours and {time_min % 60} minutes')
                    else:
                        print('starting PGM run:', verbose_message)

                        if truncate_data is not None: nn.nns[0].truncate_data_(truncate_data)
                        else: pass
                        
                        nn.set_ic_map_step1(ind_root_atom=self.ind_rO, option=self.option)
                        nn.set_ic_map_step2(check_PES=True)
                        nn.set_ic_map_step3(n_mol_unitcells = [n_mol_unitcell])
                        print(f'learning_rate = {learning_rate}')
                        nn.set_model(n_layers = n_layers, learning_rate=learning_rate, n_att_heads=n_att_heads, evaluation_batch_size=5000)
                        nn.set_trainer(n_batches_between_evaluations=50)
                        print(f'training for n_batches = {n_training_batches}')
                        nn.train(n_batches = n_training_batches, save_misc = True, save_BAR = True,
                                test_inverse = False, evaluate_on_training_data = False)
                        nn.load_misc_()
                        nn.solve_BAR_using_pymbar_(rerun=True)
                        nn.save_samples_(20000)
                        nn.save_model_()

                        print('finished PGM run:', verbose_message)
                    print('\n ################################################################################## \n')
                    del nn
        print('done')
        return evaluation

    def MBAR_(  self,
                form : str,
                Tref_FEref_SEref : list, # lattice
                # Tref, FEref, SEref = Tref_FEref_SEref
                #     Tref  : Kelvin
                #     FEref : lattice FE at Tref (in kT) 
                #     SEref : lattice FE standard error at Tref (kn kT)

                list_Ts : list,
                batch_size : int,
                
                n_mol_NPT : int,
                NPT_subfolder = '',

                FEref_SEref_is_Helmholtz = True,
                Tref_box = None, # in case PDB file missing or replaced, can find the box inside nn.model.ic_maps and provide it here

                clear_memory = True,
                xyz_not_in_datasets = False,
                get_result = True,

                f2g_correction_params = {'version':1, 'bins':40},
                use_representative_subsets = False,
            ):
        
        PATH = f'{self.molecules_folder}/{self.name}'
        FF_name = self.FF_class.FF_name

        Tref, FEref, SEref = Tref_FEref_SEref
        
        cell = self.supercell_details[form]['supercells'][n_mol_NPT] ; cell_str = cell_to_cell_str_(cell)
        list_dataset_names = [f'{PATH}/data/NPT/{NPT_subfolder}/{self.name}_{FF_name.lower()}_NPT_dataset_form_{form}_Cell_{cell_str}_Temp_{T}' + self.KEY for T in list_Ts]

        if not FEref_SEref_is_Helmholtz:
            if Tref_box is not None: print('! ignoring Tref_box because FEref_SEref_is_Helmholtz = False')
            else: pass
        else:
            if Tref_box is None:
                print('taking Tref_box from PDB, because FEref_SEref_is_Helmholtz = True, and Tref_box = None')
                Tref_box = PDB_to_box_(self.FFname_form_cell_T_to_eqm_supercell_PDB_(FF_name, form, cell, Tref, key=self.KEY))
            else : pass

        curve = g_of_T(   

                    Tref = Tref,
                    Tref_FE = FEref * n_mol_NPT,
                    Tref_SE = SEref * n_mol_NPT,
                    Tref_box = Tref_box,

                    paths_datasets_NPT = list_dataset_names,
                    xyz_not_in_datasets = xyz_not_in_datasets, # bool # True if xyz (only) in the dataset was cut short to save memory (when self.evalautions can be loaded)

                    f2g_correction_params =  f2g_correction_params,
                    )
            
        curve.compute_all_evaluations_(m=batch_size)
        if clear_memory: curve.clear_memory_()
        else: pass
        curve.compute_MBAR_(m=batch_size, use_representative_subsets=use_representative_subsets)
        print(color_text_(f'mbar : number of FF evaluations involved: {curve.n_energy_evalautions}','I'))
        print(f'average lattice enthalpy interpolation; maximum error: {curve._test_average_enthalpy_interpolator_(m=batch_size)}kT')
        
        if get_result: curve.get_result_(Tmin=50, Tmax=800, Tstride=500, save=True)
        else: pass

        return curve

####################################################################################################################
## plotting:

form_to_color_ = lambda form : dict(zip(list_forms, [f'C{i}' for i in range(len(list_forms))]))[form]

def plot_box_lengths_angles_histograms_(boxes, b0 = None, b1=None):
    cell_NPT = cell_lengths_and_angles_(boxes)
    if b0 is None: b0 = np.array(boxes[0])
    else: b0 = np.array(b0).reshape([3,3])
    if b1 is None: b1 = np.array(b0)
    else: b1 = np.array(b1).reshape([3,3])

    cell_in = cell_lengths_and_angles_(b0)
    cell_out = cell_lengths_and_angles_(b1)
    
    fig, ax = plt.subplots(1,2, figsize=(8,2), dpi=100)
    Max0 = max([
        plot_1D_histogram_(cell_NPT[:,0], bins=30, ax=ax[0], return_max_y=True, color='C0'),
        plot_1D_histogram_(cell_NPT[:,1], bins=30, ax=ax[0], return_max_y=True, color='C1'),
        plot_1D_histogram_(cell_NPT[:,2], bins=30, ax=ax[0], return_max_y=True, color='C2'),
        ])
    for i in range(0,3):
        ax[0].plot([cell_in[i]]*2, [0,Max0], color='C'+str(i), linestyle='dotted', linewidth=1.5)
        ax[0].plot([cell_out[i]]*2, [0, Max0], color='C'+str(i), linewidth=1.5)
    Max1 = max([
        plot_1D_histogram_(cell_NPT[:,3], bins=30, ax=ax[1], return_max_y=True, color='C0'),
        plot_1D_histogram_(cell_NPT[:,4], bins=30, ax=ax[1], return_max_y=True, color='C1'),
        plot_1D_histogram_(cell_NPT[:,5], bins=30, ax=ax[1], return_max_y=True, color='C2'),
        ])
    for i in range(3,6):
        ax[1].plot([cell_in[i]]*2, [0,Max1], color='C'+str(i-3), linestyle='dotted', linewidth=1.5)
        ax[1].plot([cell_out[i]]*2, [0,Max1], color='C'+str(i-3), linewidth=1.5)

    ax[0].set_xlabel('box vector lengths / nm')
    ax[1].set_xlabel('box vector angles / degrees')
    
    plt.tight_layout()
    plt.show()

def plot_PGM_results_(evaluation, window=3, plot_raw_errors = True, figsize=(10,4), dpi=None, **kwargs):
    keys = evaluation.keys()
    Ts = np.array(list(set([key[1] for key in keys])))
    Ts = np.sort(Ts)
    T2index = dict(zip(Ts,np.arange(len(Ts))))
    fig, ax = plt.subplots(1, max([len(Ts), 2]), figsize=figsize, dpi=dpi)
    for key in keys:
        print(f'form = {key[0]}, T = {key[1]}, n_mol = {key[2]}')
        _ax = ax[T2index[key[1]]]
        evaluation[key].plot_result_(n_mol = key[-1],
                                                    colors=[form_to_color_(key[0])]*4,
                                                    window=window,
                                                    ax = _ax,
                                                    plot_raw_errors = plot_raw_errors,
                                                    **kwargs,
                                                    )
        _ax.set_title(f'T = {key[1]}K', size=13)
    plt.tight_layout()
    return ax
    #plt.show()

def plot_curves_(curves,
                 y_lim = [-1,10],  # kT
                 x_lim = [50,800], # K
                 y_lim_enthalpy = None, # default None (same for both subplots)
                 show_pure_mbar_error = True,
                 circle_unconverged_enthalpies = True,
                 units_kJ_per_mol = False,
                 loc_forms = 'upper center',
                 loc_ax0 = 'upper right',
                 loc_ax1 = 'upper right',
                 subplots_args = [2,1], # subplots_args= [1,2],
                 subplots_kwargs = {'dpi': 150}, # subplots_kwargs= {'dpi': 300, 'figsize':(8,4)}
                 key_min = None,
                 capsize = 5,
                 y_stride = 2, # int
                 title_size = 12,
                 label_size = 12,
                ):
    if all([val.RES['ref']['Helmholtz_to_Gibbs_correction'] == None for val in curves.values()]): no_Helmholtz = True
    else: no_Helmholtz = False

    if y_lim_enthalpy is None: y_lim_enthalpy = y_lim
    else: pass
    colors = {}
    
    fig, ax = plt.subplots(*subplots_args, **subplots_kwargs) # 2,1, dpi=dpi)
    
    keys = [_ for _ in curves.keys()] # list_forms
    n_states = len(keys)
    
    #### #### #### #### 
    ## Tref plotting Gibbs and Helmholtz FE differences at Tref:
    
    Tref    = curves[keys[0]].RES['ref']['Tref']
    n_mol   = curves[keys[0]].RES['ref']['n_mol']
    assert all([curves[key].RES['ref']['Tref'] == Tref for key in keys])
    assert all([curves[key].RES['ref']['n_mol'] == n_mol for key in keys])
    
    f_Tref   = np.array([curves[key].RES['ref']['Helmholtz_FE'] for key in keys]) / n_mol
    g_Tref   = np.array([curves[key].RES['ref']['Gibbs_FE'] for key in keys])     / n_mol
    se_Tref  = np.array([curves[key].RES['ref']['SE'] for key in keys])           / n_mol

    if key_min is None: ind_min  = np.argmin(f_Tref)
    else: ind_min = np.where(np.array([_ for _ in curves.keys()]) == key_min)[0].astype(np.int32)[0]

    if units_kJ_per_mol:
        f_Tref  *= (Tref*CONST_kB)
        g_Tref  *= (Tref*CONST_kB)
        se_Tref *= (Tref*CONST_kB)
    else: pass

    color_0 = 'red'
    color_1 = 'blue'
    for i in range(n_states):
        if i == 0: label_0 = r'$T_{\text{ref}}$'+' Helmholtz'
        else:      label_0 = None
        if i == 0: label_1 = r'$T_{\text{ref}}$'+' Gibbs' #Helmholtz + PV_correction'
        else:      label_1 = None
        if no_Helmholtz: label_0 = None ; color_0 = color_1 
        else: pass
        ax[0].scatter([Tref],  [f_Tref[i] - f_Tref[ind_min]], color=color_0, s=5, zorder=100, label=label_0)
        ax[0].errorbar([Tref], [f_Tref[i] - f_Tref[ind_min]], [se_Tref[i] + se_Tref[ind_min]], color=color_0, fmt="", capsize=capsize, zorder=100)
        ax[0].scatter([Tref],  [g_Tref[i] - g_Tref[ind_min]], color=color_1, s=2.5, zorder=100, label=label_1)
        #plt.errorbar([Tref], [g_Tref[i] - g_Tref[ind_min]], [se_Tref[i] + se_Tref[ind_min]], color=color_1, fmt="", capsize=capsize, zorder=5)

    colors[r'$T_{\text{ref}}$'+' Helmholtz'] = color_0 
    colors[r'$T_{\text{ref}}$'+' Gibbs'] = color_1
    
    #### #### #### #### 
    ## discrete plotting: (generality: tricky if temperature grids are different; long process)
    #                     (NPT temperature grids different can be e.g., if one form melted or changed state during NPT, but not other forms)
    set_Ts_disc = set([]) # unique unique discrete temperatures exists overall
    for key in keys:
       set_Ts_disc = set_Ts_disc | set(curves[key].RES['discrete']['grid'])
    n_discrete_temperatures = len(set_Ts_disc) # how many unique discrete temperatures overall
    # FE (g), err (se), av. enthalpy (u)
    arr_disc_g      = np.zeros([n_states, n_discrete_temperatures]) + np.nan
    arr_disc_se     = np.zeros([n_states, n_discrete_temperatures]) + np.nan
    arr_disc_u      = np.zeros([n_states, n_discrete_temperatures]) + np.nan
    arr_disc_u_full = np.zeros([n_states, n_discrete_temperatures]) + np.nan
    arr_disc_con    = np.zeros([n_states, n_discrete_temperatures]) + np.nan
    Ts_disc = np.array(list(set_Ts_disc))
    for i in range(n_states):
        key = keys[i]
        inds_fill = np.array([np.where(Ts_disc==T)[0][0] for T in curves[key].RES['discrete']['grid']])

        if units_kJ_per_mol: C = curves[key].RES['discrete']['grid'] * CONST_kB
        else: C = 1.0

        arr_disc_g[i,inds_fill]      = curves[key].RES['discrete']['Gibbs_FE']                * C
        arr_disc_se[i,inds_fill]     = curves[key].RES['discrete']['Gibbs_SE']                * C
        arr_disc_u[i,inds_fill]      = curves[key].RES['discrete']['Enthalpy']                * C
        arr_disc_u_full[i,inds_fill] = curves[key].RES['discrete']['Enthalpy_all_data']       * C
        arr_disc_con[i,inds_fill]    = curves[key].RES['discrete']['converged']

        del C
    
    # discrete Gibbs FE differences:
    color_0 = 'black'
    color_1 = 'black'
    for i in range(n_states):
        for j in range(n_discrete_temperatures):
            if i == 0 and j==0 : label_0 = r'$T_{\text{NPT}}$'+' Gibbs'
            else:                label_0 = None
            y     = float(arr_disc_g[i][j] - arr_disc_g[ind_min][j])
            y_err = float(arr_disc_se[i][j] + arr_disc_se[ind_min][j])
            ax[0].scatter( [Ts_disc[j]], [y], color=color_0, s=1, label=label_0)
            ax[0].errorbar([Ts_disc[j]], [y], [y_err], color=color_0, fmt="", capsize=capsize, zorder=5)

    #ax[0].legend(loc="upper right", fontsize=7)

    colors[r'$T_{\text{NPT}}$'+' Gibbs'] = color_0 
    
    # discrete enthalpy differences:
    color_0 = 'black' # partial
    color_1 = 'red' # full
    for i in range(n_states):
        for j in range(n_discrete_temperatures):
            if i == 0 and j==0 : label_0 = r'$T_{\text{NPT}}$ '+ 'eval. batches'
            else:                label_0 = None
            if i == 0 and j==0 : label_1 = r'$T_{\text{NPT}}$ '+ 'full datasets'
            else:                label_1 = None

            y     = float(arr_disc_u[i][j] - arr_disc_u[ind_min][j])
            ax[1].scatter( [Ts_disc[j]], [y], color=color_0, zorder=1, s=10, label=label_0)
            y     = float(arr_disc_u_full[i][j] - arr_disc_u_full[ind_min][j])
            ax[1].scatter( [Ts_disc[j]], [y], color=color_1, zorder=0, s=10, label=label_1)

            if circle_unconverged_enthalpies:
                if arr_disc_con[i][j] == 1: pass
                else:
                    ax[1].plot( [Ts_disc[j]], [y], 'o',
                            markersize=15, markerfacecolor='none', markeredgecolor='red', markeredgewidth=1,
                            )
            else: pass

    ax[1].legend(loc=loc_ax1, fontsize=7)

    # plot grey bands:
    for T in Ts_disc:
        ax[0].plot([T]*2, y_lim, color='black', alpha=0.1, zorder=-100)
        ax[1].plot([T]*2, y_lim_enthalpy, color='black', alpha=0.1, zorder=-100)
    
    #### #### #### #### 
    ## continuous curve plotting:
    # any instance of g_of_T (called curve here) can use any grid for the continuous curve, so this has to be consistent; much easier to plot.
    Ts_curve = curves[keys[0]].RES['curve']['grid']
    assert all([np.abs(curves[key].RES['curve']['grid'] - Ts_curve).max()==0 for key in keys])
    
    #if units_kJ_per_mol:
    #    list_curve_g  = [curves[key].RES['curve']['Gibbs_FE'] * curves[key].RES['curve']['grid'] * CONST_kB for key in keys] 
    #    list_curve_se = [curves[key].RES['curve']['Gibbs_SE'] * curves[key].RES['curve']['grid'] * CONST_kB for key in keys] 
    #    list_curve_u  = [curves[key].RES['curve']['Enthalpy'] * curves[key].RES['curve']['grid'] * CONST_kB for key in keys]
    #    list_PGM_se   = [(curves[key].RES['ref']['SE'] / curves[key].RES['ref']['n_mol']) * curves[key].RES['ref']['Tref'] * CONST_kB  for key in keys]
    #else:
    list_curve_g  = [curves[key].RES['curve']['Gibbs_FE'] for key in keys] 
    list_curve_se = [curves[key].RES['curve']['Gibbs_SE'] for key in keys] 
    list_curve_u  = [curves[key].RES['curve']['Enthalpy'] for key in keys]
    list_PGM_se   = [curves[key].RES['ref']['SE'] / curves[key].RES['ref']['n_mol']  for key in keys]
    
    # continuous Gibbs FE differences:
    for i in range(n_states):
        color = f'C{i}'
        delta_g    = np.array(list_curve_g[i] - list_curve_g[ind_min])
        delta_g_se = np.array(list_curve_se[i] + list_curve_se[ind_min])
        delta_g_se_PGM = float(list_PGM_se[i] + list_PGM_se[ind_min])

        if units_kJ_per_mol:
            ax[0].fill_between(Ts_curve,
                               (delta_g - delta_g_se) * Ts_curve*CONST_kB,
                               (delta_g + delta_g_se) * Ts_curve*CONST_kB, alpha=0.4, color=color)
        else: 
            ax[0].fill_between(Ts_curve, delta_g - delta_g_se, delta_g + delta_g_se, alpha=0.4, color=color)

        if show_pure_mbar_error:
            if units_kJ_per_mol:
                ax[0].plot(Ts_curve,
                           (delta_g - (delta_g_se - delta_g_se_PGM)) * Ts_curve*CONST_kB,
                           linestyle='dotted', color=color, alpha=1, linewidth=1)
                ax[0].plot(Ts_curve,
                           (delta_g + (delta_g_se - delta_g_se_PGM)) * Ts_curve*CONST_kB,
                           linestyle='dotted', color=color, alpha=1, linewidth=1)
            else:
                ax[0].plot(Ts_curve, delta_g - (delta_g_se - delta_g_se_PGM), linestyle='dotted', color=color, alpha=1, linewidth=1)
                ax[0].plot(Ts_curve, delta_g + (delta_g_se - delta_g_se_PGM), linestyle='dotted', color=color, alpha=1, linewidth=1)
        else: pass

    # continuous enthalpy differences:
    for i in range(n_states):
        color   = form_to_color_(keys[i]) # f'C{i}'
        delta_u = np.array(list_curve_u[i] - list_curve_u[ind_min])
        if units_kJ_per_mol:
            ax[1].plot(Ts_curve, delta_u * Ts_curve*CONST_kB, alpha=1.0, color= color, linestyle='--')
        else:
            ax[1].plot(Ts_curve, delta_u, alpha=1.0, color= color, linestyle='--')

    ax[0].set_yticks(np.arange(-50,50,y_stride)[1:])
    ax[1].set_yticks(np.arange(-50,50,y_stride)[1:])

    ax[0].set_ylim(y_lim)
    ax[0].set_xlim(x_lim)
    ax[0].set_xlabel(r'$T \; / \; \text{K}$')
    
    ax[1].set_ylim(y_lim_enthalpy)
    ax[1].set_xlim(x_lim)
    ax[1].set_xlabel(r'$T \; / \; \text{K}$')
    
    ax[0].set_title('FEs', size=title_size)
    ax[1].set_title('average enthalpies', size=title_size)
    if units_kJ_per_mol:
        ax[0].set_ylabel(r'$\Delta G_{\text{latt}} \; / \;  \text{kJ mol}^{-1}$', size=label_size)
        ax[1].set_ylabel(r'$\Delta \langle U + PV \rangle_{\text{latt}} \; / \;  \text{kJ mol}^{-1}$', size=label_size)
    else:
        ax[0].set_ylabel(r'$\Delta G_{\text{latt}}$  / kT', size=label_size)
        ax[1].set_ylabel(r'$\Delta \langle U + PV \rangle_{\text{latt}}$  / kT', size=label_size)

    ####
    lines = []
    for i in range(n_states):
        line, = ax[0].plot([],[], color=f'C{i}', label=f'form {keys[i]}')
        lines.append(line)
    forms_legend = ax[0].legend(handles=lines, loc=loc_forms, fontsize=7)
    ax[0].add_artist(forms_legend)

    lines = []
    for _key in colors:
        if no_Helmholtz and 'Helm' in _key: pass
        else:
            line, = ax[0].plot([],[], color=colors[_key], label=_key)
            lines.append(line)
        
    other_legend = ax[0].legend(handles=lines, loc=loc_ax0, fontsize=7)
    ax[0].add_artist(other_legend)
    ####

    plt.tight_layout()

    return ax

def plot_curves_under_(curves,
                       ax,
                       ind_min,
                       alpha = 0.8,
                       color = 'black',
                       linewidth = 1,
                       dashes = [1,5],
                       units_kJ_per_mol=True,
                       plot_ax1 = False,
                       alpha_ax1 = 0.8,
                ):

    keys = [_ for _ in curves.keys()] # list_forms
    n_states = len(keys)
    
    Ts_curve = curves[keys[0]].RES['curve']['grid']
    assert all([np.abs(curves[key].RES['curve']['grid'] - Ts_curve).max()==0 for key in keys])
    
    list_curve_g  = [curves[key].RES['curve']['Gibbs_FE'] for key in keys] 
    list_curve_se = [curves[key].RES['curve']['Gibbs_SE'] for key in keys] 
    list_curve_u  = [curves[key].RES['curve']['Enthalpy'] for key in keys]
    list_PGM_se   = [curves[key].RES['ref']['SE'] / curves[key].RES['ref']['n_mol']  for key in keys]
    
    # continuous Gibbs FE differences:
    for i in range(n_states):
        delta_g    = np.array(list_curve_g[i] - list_curve_g[ind_min])
        delta_g_se = np.array(list_curve_se[i] + list_curve_se[ind_min])
        delta_g_se_PGM = float(list_PGM_se[i] + list_PGM_se[ind_min])

        if units_kJ_per_mol:
            ax[0].plot(Ts_curve, (delta_g - delta_g_se) * Ts_curve*CONST_kB, color=color, alpha=alpha, zorder=-10, linewidth=linewidth, dashes=dashes)
            ax[0].plot(Ts_curve, (delta_g + delta_g_se) * Ts_curve*CONST_kB, color=color, alpha=alpha, zorder=-10, linewidth=linewidth, dashes=dashes)
        else: 
            ax[0].plot(Ts_curve, (delta_g - delta_g_se), color=color, alpha=alpha, zorder=-10, linewidth=linewidth, dashes=dashes)
            ax[0].plot(Ts_curve, (delta_g + delta_g_se), color=color, alpha=alpha, zorder=-10, linewidth=linewidth, dashes=dashes)

    if plot_ax1:
        for i in range(n_states):
            delta_u = np.array(list_curve_u[i] - list_curve_u[ind_min])
            if units_kJ_per_mol:
                ax[1].plot(Ts_curve, delta_u * Ts_curve*CONST_kB, color=color, alpha=alpha_ax1, linestyle='--')
            else:
                ax[1].plot(Ts_curve, delta_u                    , color=color, alpha=alpha_ax1, linestyle='--')
    else: pass

    return ax

####################################################################################################################
