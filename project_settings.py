from O.interface import *
from O.sym import *

from O.MM.bias import *

from O.MM.Tx import *

from O.NN.pgm_rb import *

"""
work in process, will be updated over time
"""

## ## ## ##

''' cif -> pdb with standardised order and naming of atoms in each molecule of a unitcell
# mercury : cif -> pdb
# mercury output : './O/MM/molecules/smz/Form_V.pdb'
reorder_atoms_unitcell_(PDB='./O/MM/molecules/smz/Form_V.pdb', PDB_ref='./O/MM/molecules/smz/smz_single_mol.pdb', n_atoms_mol=n_atoms_mol)
# -> saved: ./O/MM/molecules/smz/Form_V_reordered.pdb
# renamed by hand: Form_V_reordered.pdb -> smz_V_unitcell.pdb
# also saved : smz_V_unitcell_cell111.pdb == smz_V_unitcell.pdb
'''

#############################################

list_make_global = ['mol_name',                 # supervised
                    'PATH',                     # supervised
                    'single_molecule_PDB',
                    'n_atoms_mol',              # supervised
                    'FF_class',                 # supervised
                    'FF_name',
                    'list_Forms',               # supervised
                    'PME_cutoff',               # supervised
                    'KEY',
                    'ind_rO',                   # supervised
                    'option',                   # supervised
                    'supercell_details',        # supervised
                    'steps_followed',           # supervised
                    'symmetry_reduction_step_', # supervised
                    'NPT_subfolder',
                   ]

''' supercell_details:
to populate entries 'n_mol_unitcell' and 'supercells' in the supercell_details:
# repeat for each Form in list_Forms:
Form = 'I'
n_mol_unitcell = PDB_to_xyz_(Form_cell_to_ideal_supercell_PDB_(Form,[1,1,1])).shape[0]/n_atoms_mol
assert n_mol_unitcell == int(n_mol_unitcell)
n_mol_unitcell = int(n_mol_unitcell)
supercell_details[Form]['n_mol_unitcell'] = n_mol_unitcell 
top_n = 10 # adjust top_n to have supercell shapes that can be considered.
supercells = get_unitcell_stack_order_(PDB_to_box_(Form_cell_to_ideal_supercell_PDB_(Form,[1,1,1])),
                          n_mol_unitcell=n_mol_unitcell, top_n=top_n)
list_n_mol ; list of n_mol that will be actually used from the 'supercells' part
'''

#############################################
## general:

cell_to_cell_str_ = lambda cell : ''.join([str(x) for x in cell])
Form_cell_to_ideal_supercell_PDB_ = lambda Form, cell : PATH+f'{mol_name}_{Form}_unitcell_cell'+cell_to_cell_str_(cell)+'.pdb'
FFname_Form_cell_T_to_eqm_supercell_PDB_ = lambda FFname, Form, cell, T, key='' : f'{PATH}/{mol_name}_{FFname.lower()}_equilibrated_Form_{Form}_Cell_{cell_to_cell_str_(cell)}_Temp_{T}{key}.pdb'

def check_ideal_supercells_(min_n_mol = 1):
    boxes = []
    for Form in list_Forms:
        list_n_mol = supercell_details[Form]['list_n_mol']
        print(f'Form {Form} with considered list_n_mol = {list_n_mol}')
        for n_mol in list_n_mol:
            if n_mol >= min_n_mol:
                cell = supercell_details[Form]['supercells'][n_mol] ; cell_str = cell_to_cell_str_(cell)
                PDB_unitcell = f'{PATH}/{mol_name}_{Form}_unitcell.pdb'
                PDB_supercell = f'{PATH}/{mol_name}_{Form}_unitcell_cell{cell_str}.pdb'
                assert os.path.exists(PDB_unitcell)
                if not os.path.exists(PDB_supercell):
                    PDB_supercell = supercell_from_unitcell_(PDB_unitcell, cell=cell, save_output=True)[-1]
                else: pass
                boxes.append(PDB_to_box_(PDB_supercell))
            else:
                print(f'Form {Form} : {n_mol} < {min_n_mol}')

    boxes = np.stack(boxes)
    cutoff_max = np.min([boxes[...,i,i].min() for i in range(3)]) * 0.5
    print(f'maximum PME cutoff possible with current set of list_n_mol selections: {cutoff_max.round(3)}nm')
    print(f'PME cutoff used: {PME_cutoff}nm')

def NPT_(
            _list_Forms : list,
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

    average_energies = {}

    for Form in _list_Forms:
        for T in _Ts:
            for n_mol in _list_n_mol[Form]:

                cell = supercell_details[Form]['supercells'][n_mol] ; cell_str = cell_to_cell_str_(cell)
                verbose_message = f'Form={Form}, cell={cell} (n_mol={n_mol}), T={T}, FF={FF_name}'

                if T_init is not None:
                    PDB_initial = FFname_Form_cell_T_to_eqm_supercell_PDB_(FF_name, Form=Form, cell=cell, T=T_init, key=KEY)
                    assert os.path.exists(PDB_initial), '! no equilibrated supercell found for: Form={Form}, cell={cell} (n_mol={n_mol}), T={T_init}, FF={FF_name}'
                    if not checking: print(f'NPT simulation initialised from supercell that was already equilibrated at T={T_init}')
                    else: pass
                else:
                    PDB_initial = Form_cell_to_ideal_supercell_PDB_(Form, cell)
                    assert os.path.exists(PDB_initial), f'! no ideal supercell found for: Form={Form}, n_mol={n_mol}'
                    if not checking: print(f'NPT simulation initialised from ideal supercell')
                    else: pass

                path_NPT_dataset = f'{PATH}/data/NPT/{NPT_subfolder}/{mol_name}_{FF_name.lower()}_NPT_dataset_Form_{Form}_Cell_{cell_str}_Temp_{T}' + KEY
                path_equilibrated_supercell = FFname_Form_cell_T_to_eqm_supercell_PDB_(FF_name, Form, cell, T, key=KEY)

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
                        
                        converged = TestConverged_1D(u)() ; average_energies[(Form, T, n_mol)] = [u_mean, converged]
                        
                        print(f'average lattice potential energy: {u_mean} kT')
                        print(f'number of samples: {u.shape[0]}')

                        if checking_resave_datasets_with_CUT_xyz:
                            print('shrinking down the dataset file: (full file backed up seperately)')
                            dataset['MD dataset']['xyz'] = dataset['MD dataset']['xyz'][:50]
                            save_pickle_(dataset, name=path_NPT_dataset)
                        else: pass

                        if checking_try_plotting_equlibrated_cell:
                            try:
                                path_ideal_supercell = Form_cell_to_ideal_supercell_PDB_(Form, cell)
                                plot_box_lengths_angles_histograms_(dataset['MD dataset']['b'], 
                                                                    b0 = PDB_to_box_(path_ideal_supercell), 
                                                                    b1 = PDB_to_box_(path_equilibrated_supercell),
                                                                    )
                            except: print('could not find equilibrated supercell extracted from this NPT dataset')
                        else: pass
                        del dataset
                else: pass

                if not checking or (rerun_unconverged and not average_energies[(Form, T, n_mol)][1]):
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
                            assert FF_class.FF_name == FF_name, ''
                            
                            print(color_text_(f'\n starting trajectory, this will be saved into folder {PATH}/data/NPT/{NPT_subfolder} \n', 'I'))

                            if rerun_unconverged:
                                sc = SingleComponent.initialise_from_save_(path_NPT_dataset)
                                if overwrite: sc.set_arrays_blank_()
                                else: pass
                                # sc.args_initialise_object['PDB'] = Form_cell_to_ideal_supercell_PDB_(Form, cell)

                            else:
                                sc = SingleComponent(PDB=PDB_initial, n_atoms_mol=n_atoms_mol, name=mol_name, FF_class=FF_class)
                                sc.initialise_system_(PME_cutoff=PME_cutoff, nonbondedMethod=app.PME)
                                sc.initialise_simulation_(timestep_ps = 0.002, P = 1, T = T) # ps, atm, K
                                sc.simulation.step(n_frames_warmup * n_frames_stride)
                            
                            sc.run_simulation_(n_frames_sample, n_frames_stride)

                            sc.save_simulation_data_(path_NPT_dataset)

                            u = np.array(sc.u) / n_mol
                            converged = TestConverged_1D(u)()
                            u_mean = u.mean()
                            print(f'average lattice potential energy: {u_mean} kT')
                            average_energies[(Form, T, n_mol)] = [u_mean, converged]

                            if save_equilibrated_supercell:
                                print('equilibrated supercell PDB is being extracted and saved:')
                                sc._xyz = tidy_crystal_xyz_(sc.xyz, sc.boxes, n_atoms_mol=sc.n_atoms_mol, ind_rO=ind_rO)
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

def NVT_( 
            _list_Forms : list,
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
    average_energies = {}

    for Form in _list_Forms:
        for T in _Ts:
            for n_mol in _list_n_mol[Form]:
                
                cell = supercell_details[Form]['supercells'][n_mol] ; cell_str = cell_to_cell_str_(cell)
                verbose_message = f'Form={Form}, cell={cell} (n_mol={n_mol}), T={T}, FF={FF_name}'
            
                path_equilibrated_supercell = FFname_Form_cell_T_to_eqm_supercell_PDB_(FF_name, Form, cell, T, key=KEY)
                path_NVT_dataset = f'{PATH}/data/{mol_name}_{FF_name.lower()}_NVT_dataset_Form_{Form}_Cell_{cell_str}_Temp_{T}' + KEY

                assert FF_class.FF_name == FF_name, ''

                if checking:
                    print('checking NVT simulation:', verbose_message)
                    print(f'dataset: {path_NVT_dataset}')
                    print('path exists:', os.path.exists(path_NVT_dataset))

                    if checking_check_only_path: pass
                    else:
                        dataset = load_pickle_(path_NVT_dataset)
                        u = np.array(dataset['MD dataset']['u']) / n_mol
                        u_mean = u.mean()

                        converged = TestConverged_1D(u)() ; average_energies[(Form, T, n_mol)] = [u_mean, converged]

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
                        assert FF_class.FF_name == FF_name, ''

                        print(color_text_(f'\n starting trajectory, this will be saved into folder {PATH}/data \n', 'I'))
        
                        sc = SingleComponent(PDB=path_equilibrated_supercell, n_atoms_mol=n_atoms_mol, name=mol_name, FF_class=FF_class)
                        sc.initialise_system_(PME_cutoff=PME_cutoff, nonbondedMethod=app.PME)
                        sc.initialise_simulation_(timestep_ps = 0.002, P = None, T = T)
                        sc.simulation.step(n_frames_warmup * n_frames_stride)
                        sc.run_simulation_(n_frames_sample, n_frames_stride)
                        sc.save_simulation_data_(path_NVT_dataset)

                        u = np.array(sc.u) / n_mol
                        converged = TestConverged_1D(u)()
                        u_mean = u.mean()
                        print(f'average lattice potential energy: {u_mean} kT')
                        average_energies[(Form, T, n_mol)] = [u_mean, converged]

                        del sc
                        print('finished NVT simulation:', verbose_message)

                print('\n ################################################################################## \n')
    print('done')
    return average_energies

def PGM_(
        _list_Forms : list,
        _Ts : list,
        _list_n_mol : dict,

        n_training_batches = 20000,
        learning_rate = 0.001,
        symmetry_reduction_step_ = None,

        checking = False, # just to check the files were not lost
        checking_load_datasets = False,

        PGM_key = '',
        NVT = True,
        NPT_subfolder = '',

        truncate_data : int = None,
        ):
    ''' no GUI '''
    evaluation = {}

    for Form in _list_Forms:
        for T in _Ts:
            for n_mol in _list_n_mol[Form]:

                cell = supercell_details[Form]['supercells'][n_mol] ; cell_str = cell_to_cell_str_(cell)
                verbose_message = f'Form={Form}, cell={cell} (n_mol={n_mol}), T={T}, FF={FF_name}'
                n_mol_unitcell = int(supercell_details[Form]['n_mol_unitcell'])

                if NVT: path_dataset = f'{PATH}/data/{mol_name}_{FF_name.lower()}_NVT_dataset_Form_{Form}_Cell_{cell_str}_Temp_{T}' + KEY
                else:   path_dataset = f'{PATH}/data/NPT/{NPT_subfolder}/{mol_name}_{FF_name.lower()}_NPT_dataset_Form_{Form}_Cell_{cell_str}_Temp_{T}' + KEY

                if symmetry_reduction_step_ is None: sr_KEY = ''
                else:                                sr_KEY = symmetry_reduction_step_(path_dataset, Form=Form, T=T, checking=checking)

                if NVT: PGM_instance_name = f'{mol_name}_{FF_name.lower()}_Form_{Form}_Cell_{cell_str}_Temp_{T}' + KEY + sr_KEY + PGM_key
                else:   PGM_instance_name = f'rb_{mol_name}_{FF_name.lower()}_Form_{Form}_Cell_{cell_str}_Temp_{T}' + KEY + sr_KEY + PGM_key

                print(f'PGM instance name: {PGM_instance_name}')

                if checking and not os.path.exists(path_dataset + sr_KEY): pass
                else: path_dataset += sr_KEY 

                if not os.path.exists(path_dataset):
                    print('!! dataset related this training run was not found')
                    #print(path_dataset)
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
                    evaluation[(Form, T, n_mol)] = nn

                    time_min = nn.training_time
                    print(f'training and evaluation duration: {time_min // 60} hours and {time_min % 60} minutes')
                else:
                    print('starting PGM run:', verbose_message)

                    if truncate_data is not None: nn.nns[0].truncate_data_(truncate_data)
                    else: pass
                    
                    nn.set_ic_map_step1(ind_root_atom=ind_rO, option=option)
                    nn.set_ic_map_step2(check_PES=True)
                    nn.set_ic_map_step3(n_mol_unitcells = [n_mol_unitcell])
                    print(f'learning_rate = {learning_rate}')
                    nn.set_model(n_layers = 4, learning_rate=learning_rate, n_att_heads=4, evaluation_batch_size=5000)
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

def MBAR_(
            Form : str,
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
    
    Tref, FEref, SEref = Tref_FEref_SEref
    
    cell = supercell_details[Form]['supercells'][n_mol_NPT] ; cell_str = cell_to_cell_str_(cell)
    list_dataset_names = [f'{PATH}/data/NPT/{NPT_subfolder}/{mol_name}_{FF_name.lower()}_NPT_dataset_Form_{Form}_Cell_{cell_str}_Temp_{T}' + KEY for T in list_Ts]

    if not FEref_SEref_is_Helmholtz:
        if Tref_box is not None: print('! ignoring Tref_box because FEref_SEref_is_Helmholtz = False')
        else: pass
    else:
        if Tref_box is None:
            print('taking Tref_box from PDB, because FEref_SEref_is_Helmholtz = True, and Tref_box = None')
            Tref_box = PDB_to_box_(FFname_Form_cell_T_to_eqm_supercell_PDB_(FF_name, Form, cell, Tref, key=KEY))
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

###################################################################################################################
## supercell_details:

def symmetry_reduction_step_blank_(path_NVT_dataset, Form, T, checking=True):
    return ''

#################################
supercell_details_veliparib = {
            'list_Forms' : ['I', 'II'],
            'I':{'PDB_unitcell' : None,
                    'n_mol_unitcell' : 8,
                    'supercells':{       8: [1, 1, 1], # y
                                        16: [1, 2, 1], # y        #[2, 1, 1],
                                        24: [1, 3, 1], # y        #[3, 1, 1],
                                        32: [2, 2, 1], # y
                                        40: [1, 5, 1],
                                        48: [2, 3, 1],
                                },
                    'list_n_mol': None,
                },
            'II':{'PDB_unitcell'     : None,
                    'n_mol_unitcell' : 3,
                    'supercells':{     3: [1, 1, 1],
                                       6: [1, 2, 1],
                                       9: [1, 3, 1], # y
                                      12: [2, 2, 1], # y
                                      15: [1, 5, 1],
                                      18: [3, 2, 1], # y         #[2, 3, 1],
                                      21: [1, 7, 1],
                                      24: [2, 2, 2], # y
                                      27: [3, 3, 1], # y
                                      30: [2, 5, 1],
                                      36: [2, 3, 2],
                                      42: [2, 7, 1],
                                      45: [3, 5, 1],
                                      48: [2, 4, 2],
                                },
                    'list_n_mol': None,
                },
        }

#################################
supercell_details_smz = {
            'list_Forms' : ['I', 'II', 'III', 'IV', 'V'],
            'I':{'PDB_unitcell ' : None,
                  'n_mol_unitcell' : 8,
                  'supercells':{  8: [1, 1, 1],
                                 16: [1, 2, 1], #
                                 24: [1, 3, 1], ##
                                 32: [2, 2, 1], ###
                                 40: [1, 5, 1],
                                 48: [2, 3, 1],},
                    'list_n_mol' : None,
                  },
             'II':{'PDB_unitcell' : None,
                  'n_mol_unitcell' : 8,
                   'supercells':{ 8: [1, 1, 1],
                                 16: [2, 1, 1], #
                                 24: [3, 1, 1], ##
                                 32: [2, 2, 1], ###
                                 40: [5, 1, 1],
                                 48: [3, 2, 1],},
                    'list_n_mol' : None,
                  },
             'III':{'PDB_unitcell' : None,
                  'n_mol_unitcell' : 4,
                   'supercells':{ 4: [1, 1, 1],
                                  8: [1, 2, 1],
                                 12: [1, 3, 1],
                                 16: [2, 2, 1], #
                                 20: [1, 5, 1],
                                 24: [2, 3, 1], ##
                                 28: [1, 7, 1],
                                 32: [2, 2, 2], ###
                                 36: [3, 3, 1],
                                 40: [2, 5, 1],
                                 48: [2, 3, 2],},
                    'list_n_mol' : None,
                   },
             'IV':{'PDB_unitcell' : None,
                  'n_mol_unitcell' : 4,
                  'supercells':{  4: [1, 1, 1],
                                  8: [1, 2, 1],
                                 12: [1, 3, 1],
                                 16: [1, 4, 1], #
                                 20: [1, 5, 1],
                                 24: [2, 3, 1], ##
                                 28: [1, 7, 1],
                                 32: [2, 4, 1], ###
                                 36: [3, 3, 1],
                                 40: [2, 5, 1],
                                 48: [2, 3, 2],},
                    'list_n_mol' : None,
                  },
             'V':{'PDB_unitcell' : None,
                  'n_mol_unitcell' : 8,
                  'supercells':{  8: [1, 1, 1],
                                 16: [1, 2, 1],
                                 24: [1, 3, 1],
                                 32: [1, 2, 2],
                                 40: [1, 5, 1],
                                 48: [1, 3, 2],},
                    'list_n_mol' : None,
                  },
            }

def symmetry_reduction_step_smz_(path_NVT_dataset, Form, T, checking=True):

    if T >= 250: sr_KEY = '_sym_randomised'
    else:        sr_KEY = '_sym_reduced'

    print(f'SR KEY {sr_KEY}')

    if checking: pass
    else:
        if os.path.exists(path_NVT_dataset + sr_KEY):
            print(f'this {sr_KEY} dataset aready exists')
        else:
            sr = DatasetSymmetryReduction(path_dataset=path_NVT_dataset)
            sr.set_ABCD_(ind_rO, option=option)

            if sr_KEY == '_sym_randomised':
                sr.sort_methyl_([-1])
            else:
                if   Form == 'I'  : sr.sort_methyl_([2,2,1,1,1,1,2,2], offsets=[np.pi])
                elif Form == 'II' : sr.sort_methyl_([2,1], offsets=[0])
                elif Form == 'III': sr.sort_methyl_([2,1], offsets=[np.pi])
                elif Form == 'IV' : sr.sort_methyl_([2,1], offsets=[np.pi]) # FIX should be pi, was 0
                elif Form == 'V'  : sr.sort_methyl_([1,1,2,2], offsets=[np.pi]) # fixing now
                else:  assert Form in list_Forms, f'symmetry_reduction_step_smz for Form {Form} was not yet defined'

            sr.sort_n2_([28, 29], [-1])

            sr.check_energy_(1000)
            sr.save_sym_reduced_dataset_(key = sr_KEY)

    return sr_KEY

#################################
supercell_details_mivebresib = {
            'list_Forms' : ['I', 'II', 'III',],
            'I':{'PDB_unitcell' : None,
                  'n_mol_unitcell': 8,
                  'supercells':{    8: [1, 1, 1],
                                   16: [1, 2, 1],
                                   24: [1, 3, 1],
                                   32: [1, 2, 2],
                               },
                  'list_n_mol' : None,
                },
            'II':{'PDB_unitcell' : None,
                  'n_mol_unitcell' : 2,
                  'supercells':{ #2: [1, 1, 1],
                                 #4: [2, 1, 1],
                                 #6: [3, 1, 1],
                                 8: [4, 1, 1],
                                 10: [5, 1, 1],
                                 12: [3, 2, 1],
                                 14: [7, 1, 1],
                                 16: [4, 2, 1],
                                 18: [3, 3, 1],
                                 20: [5, 2, 1],
                                 24: [6, 2, 1],
                                 28: [7, 2, 1],
                                 30: [5, 3, 1],
                                 32: [4, 2, 2],
                               },
                  'list_n_mol' : None,
                },
            'III':{'PDB_unitcell' : None,
                  'n_mol_unitcell' : 8,
                  'supercells':{ 8: [1, 1, 1],
                                 16: [2, 1, 1],
                                 24: [3, 1, 1],
                                 32: [2, 2, 1],
                               },
                  'list_n_mol' : None,
                },
            }

def symmetry_reduction_step_miv_(path_dataset, Form, T, checking=True):
    if FF_name == 'GAFF': sr_KEY = '_sym_adjusted'
    else: sr_KEY = '_sym_reduced'
    if checking: pass
    else: 
        if os.path.exists(path_dataset + sr_KEY):
            print(f'this {sr_KEY} dataset aready exists')
        else:
            sr = DatasetSymmetryReduction(path_dataset=path_dataset)
            sr.set_ABCD_(ind_rO, option=option)

            if Form == 'I':
                sr.sort_methyl_(lookup_indices = [[3,3,3,3,0,0,0,0], [-1]], offsets=[np.pi, 0])
                if FF_name == 'GAFF': sr.sort_n2_([6,7], [-1])
                else: pass
            elif Form == 'II':
                sr.sort_methyl_(lookup_indices = [[2,1], [-1]],             offsets=[np.pi, 0])
                if FF_name == 'GAFF': sr.sort_n2_([6,7], [-1])
                else: pass
            elif Form == 'III':
                sr.sort_methyl_(lookup_indices = [[3,3,3,3,0,0,0,0], [-1]], offsets=[np.pi, 0])
                if FF_name == 'GAFF': sr.sort_n2_([6,7], [-1])
                else: pass
            else: assert Form in list_Forms
            sr.check_energy_(1000)
            sr.save_sym_reduced_dataset_(key = sr_KEY)

    return sr_KEY 
#################################
'''
tmff abt072:

Form D -> new packing Form (E) found at 600K. Not new, this is the same as C (D->C transition happened).
Form M -> new conformational Form prevalent at 450K and 500K : most energetically stable across all Forms.
'''

supercell_details_abt072 = {
            'list_Forms' : ['C', 'D', 'F', 'M', 'E'],
            'C':{'PDB_unitcell'  : None,
                'n_mol_unitcell' : 4,
                'supercells':{   4: [1, 1, 1],   # 4
                                 8: [2, 1, 1],   # 8
                                12: [3, 1, 1],   # 12
                                16: [2, 2, 1],   # 16
                                20: [5, 1, 1],
                                24: [3, 2, 1],
                                28: [7, 1, 1],
                                32: [2, 2, 2],

                                96: [4, 3, 2],
                            },
                'list_n_mol': None, #[4,8,12,16,24,32],

                },
            'D':{'PDB_unitcell':None,
                'n_mol_unitcell' : 4,
                'supercells':{   4: [1, 1, 1],   # 4
                                 8: [1, 2, 1],   # 8
                                12: [1, 3, 1],   # 12
                                16: [2, 2, 1],   # 16
                                20: [1, 5, 1],
                                24: [2, 3, 1],
                                28: [1, 7, 1],
                                32: [2, 2, 2],

                                96: [3, 4, 2],
                            },
                'list_n_mol': None, #[4,8,12,16,24,32],
                },
            'F':{'PDB_unitcell':None,
                'n_mol_unitcell' : 8,
                'supercells':{   8: [1, 1, 1],   # 8
                                16: [1, 2, 1],   # 16
                                24: [1, 3, 1],
                                32: [2, 2, 1],

                                96: [2, 3, 2],
                            },
                'list_n_mol': None, #[8,16,24,32],
                },
            'M':{'PDB_unitcell':None,
                'n_mol_unitcell' : 4,
                'supercells':{   4: [1, 1, 1],   # 4
                                 8: [2, 1, 1],   # 8
                                12: [3, 1, 1],   # 12
                                16: [2, 2, 1],   # 16
                                20: [5, 1, 1],
                                24: [3, 2, 1],
                                28: [7, 1, 1],
                                32: [2, 2, 2],

                                96: [4, 3, 2],
                            },
                'list_n_mol': None, #[4,8,12,16,24,32],
                },
            'E':{'PDB_unitcell' : None,
                'n_mol_unitcell' : 16,
                'supercells':{
                                16: [1, 1, 1],  
                            },
                'list_n_mol': None, #[4,8,12,16,24,32],
                },
            }

def symmetry_reduction_step_abt072_(path_NVT_dataset, Form, T, checking=True):
    sr_KEY = '_sym_reduced'
    if checking or sr_KEY != '': pass
    else: 
        if os.path.exists(path_NVT_dataset + sr_KEY):
            print(f'this {sr_KEY} dataset aready exists')
        else:
            sr = DatasetSymmetryReduction(path_dataset=path_NVT_dataset)
            sr.set_ABCD_(ind_rO, option=option)

            if Form == 'C':
                lookup_indices_methyl = [0,3]
                lookup_indices_trimethyl = [0,3]
            elif Form in ['D','E']:
                lookup_indices_methyl = [3,3,0,0]
                lookup_indices_trimethyl = [0,0,3,3]
            elif Form == 'F':
                lookup_indices_methyl = [0,0,0,0,3,3,3,3]
                lookup_indices_trimethyl = [0,0,0,0,3,3,3,3]
            else:
                assert Form == 'M', f'symmetry_reduction_step_ : Form {Form} not recognised'
                lookup_indices_methyl = [0,0,0,0]
                lookup_indices_trimethyl = [0,0,3,3]

            sr.sort_methyl_(lookup_indices = lookup_indices_methyl)
            sr.sort_trimethyl_(lookup_indices = lookup_indices_trimethyl)
            sr.check_energy_(1000)
            sr.save_sym_reduced_dataset_(key = sr_KEY)

    return sr_KEY 

####################################################################################################################
# steps:
####################################################################################################################
## sulfamerazine (smz):

def smz_opls_():
    # smz_opls is all smz, not showing gaff here yet (those results are much older and incomplete)

    PME_cutoff = 0.5 ; KEY = f'' # all smz was done with 0.5nm as the cut-off; blank KEY.

    mol_name = 'smz'
    PATH = DIR_main+f'MM/molecules/{mol_name}/'
    NPT_subfolder = 'nmol24_opls'

    single_molecule_PDB = PATH+f'{mol_name}_single_mol.pdb'
    n_atoms_mol = 30

    FF_class = OPLS # previously OPLS_general
    FF_name = FF_class.FF_name

    supercell_details = dict(supercell_details_smz)
    list_Forms = list(supercell_details['list_Forms'])
    for Form in list_Forms:
        supercell_details[Form]['PDB_unitcell'] = PATH+f'{mol_name}_{Form}_unitcell.pdb'
    supercell_details['I']['list_n_mol']   = [16, 24, 32]
    supercell_details['II']['list_n_mol']  = [16, 24, 32]
    supercell_details['III']['list_n_mol'] = [16, 24, 32]
    supercell_details['IV']['list_n_mol']  = [16, 24, 32]
    supercell_details['V']['list_n_mol']   = [16, 24, 32]

    ind_rO = 10 ; option = 0

    #### #### #### #### #### #### #### #### #### #### 
    steps_followed = ''

    # NPT simulations:
    steps_followed += f'''
        {color_text_('# NPT','B')}:
    '''
    steps_followed += '''
    # these NPT simulations were used for:
    #     getting equlibrated supercells for NVT (at 200 and 300 K)
    #     for MBAR_ (90k out of 100k frames from each trejectory in the 'high-data regime' in the Figure; most accurate)
    #     for MBAR_ based on another NPT_subfolder where these trajectories were strided to plot the 'low-data regime'

        NPT_(
        _list_Forms = ['I', 'II', 'III', 'IV', 'V'],
        _Ts         = [200, 250, 300, 350, 400, 450, 500, 225],
        _list_n_mol = {'I': [24], 'II': [24], 'III': [24], 'IV': [24], 'V': [24]},
        NPT_subfolder = NPT_subfolder,

        n_frames_warmup=10000,
        n_frames_sample=100000,
        n_frames_stride=50,
        save_equilibrated_supercell = True,

        checking = True, # set False if re-running
        )

    # these two relatively long NPT simulations were only ran to test PGMcrys_v1_rb
        NPT_(
                    _list_Forms = ['I','II'],
                    _Ts = [300],
                    _list_n_mol = {'I': [24], 'II': [24]},
                    NPT_subfolder = '',

                    n_frames_warmup = 10000,
                    n_frames_sample = 600000,
                    n_frames_stride = 50,
                    save_equilibrated_supercell = False,

                    T_init = 300,

                    checking = True, # set False if re-running
        )
    '''

    # NVT simulations:
    steps_followed += f'''
        {color_text_('# NVT','G')}:
    '''
    steps_followed += '''
    # 200K was only included for the 'self-consitency check', but was also more accurate due to the lower PGM error at 200K compared to 300K.
        NVT_(
        _list_Forms  = ['I', 'II', 'III', 'IV', 'V'],
        _Ts  = [200, 300],
        _list_n_mol = {'I': [24], 'II': [24], 'III': [24], 'IV': [24], 'V': [24]},

        n_frames_warmup = 10000,
        n_frames_sample = 600000,
        n_frames_stride = 50,

        checking = True, # set False if re-running
        )
    '''

    # PGM runs:
    steps_followed += f'''
        {color_text_('# PGM','G')}:
    '''
    steps_followed += '''
    # the main results were based on these training runs:
        PGM_(
            _list_Forms = ['I', 'II', 'III', 'IV', 'V'],
            _Ts         = [300],
            _list_n_mol = {'I': [24], 'II': [24], 'III': [24], 'IV': [24], 'V': [24]},

            n_training_batches = 15000,
            learning_rate = 0.001,
            symmetry_reduction_step_ = symmetry_reduction_step_,

            checking = True, # set False if re-running
            PGM_key = '_ext',
            )

    # the 'self-consitency check' runs:
        PGM_(
            _list_Forms = ['I', 'II', 'III', 'IV', 'V'],
            _Ts         = [200],
            _list_n_mol = {'I': [24], 'II': [24], 'III': [24], 'IV': [24], 'V': [24]},

            n_training_batches = 10000,
            learning_rate = 0.001,
            symmetry_reduction_step_ = symmetry_reduction_step_,

            checking = True, # set False if re-running
            PGM_key = '_ext',
            )

    # just testing PGMcrys_v1_rb related to further work
        PGM_(
            _list_Forms = ['I','II'],
            _Ts         = [300],
            _list_n_mol = {'I': [24], 'II': [24]},

            n_training_batches = 15000,
            learning_rate = 0.001,
            symmetry_reduction_step_ = symmetry_reduction_step_,

            checking = True, # set False if re-running
            PGM_key = '_ext_setting_2',
            NVT = False,
            )
    '''

    # MBAR:
    steps_followed += f'''
        {color_text_('# MBAR','P')}:
    '''
    steps_followed += '''

        n_mol_Tref = 24
        evaluation = PGM_(
                        _list_Forms = ['I', 'II', 'III', 'IV', 'V'],
                        _Ts         = [200, 300],
                        _list_n_mol = {'I': [n_mol_Tref], 'II': [n_mol_Tref], 'III': [n_mol_Tref], 'IV': [n_mol_Tref], 'V': [n_mol_Tref]},
                        symmetry_reduction_step_ = symmetry_reduction_step_,
                        checking = True,
                        PGM_key = '_ext'
                        NVT = True,
                        ) 

        n_mol_NPT = 24
        
        Tref = 300

        # 'high-data regime'
        curves300_90k = {}
        for Form in ['I', 'II', 'III', 'IV', 'V']:
            curves300_90k[Form] = MBAR_(
                            Form = Form,
                            Tref_FEref_SEref = [Tref,
                                                evaluation[(Form, Tref, n_mol_Tref)].nns[0].BAR_V_FE / n_mol_Tref, # absolute_lattice_FE
                                                evaluation[(Form, Tref, n_mol_Tref)].nns[0].BAR_V_SE / n_mol_Tref, # absolute_lattice_SE
                                                ], # notebook 
                            FEref_SEref_is_Helmholtz = True,
                            Tref_box = None, 
                                # correct PDB files need to be in that folder (save_equilibrated_supercell = True),
                                # when FEref_SEref_is_Helmholtz = True and Tref_box = None.   
                                
                            list_Ts = [200, 250, 300, 350, 400, 450, 500],
                            batch_size = 90000,
                            n_mol_NPT = n_mol_NPT,
                            NPT_subfolder = NPT_subfolder,
                            clear_memory = True,
                            xyz_not_in_datasets = True, # set False if running for the first time
                            )

        # 'low-data regime'
        curves300_10k = {}
        for Form in ['I', 'II', 'III', 'IV', 'V']:
            curves300_10k[Form] = MBAR_(
                            Form = Form,
                            Tref_FEref_SEref = [Tref,
                                                evaluation[(Form, Tref, n_mol_Tref)].nns[0].BAR_V_FE / n_mol_Tref, # absolute_lattice_FE
                                                evaluation[(Form, Tref, n_mol_Tref)].nns[0].BAR_V_SE / n_mol_Tref, # absolute_lattice_SE
                                                ], # notebook 
                            FEref_SEref_is_Helmholtz = True,
                            Tref_box = None, 
                            list_Ts = [200, 250, 300, 350, 400, 450, 500],
                            batch_size = 10000,
                            n_mol_NPT =  n_mol_NPT,
                            NPT_subfolder = 'nmol24_opls_stride8',
                            clear_memory = True,
                            xyz_not_in_datasets = True, # set False if running for the first time
                            )
        
        Tref = 200
        curves200_90k = {}
        curves200_10k = {}
        ... similar to above, shown in the smz notebook
    '''

    #symmetry_reduction_step_ = lambda path_NVT_dataset, Form, T, checking: ''
    # ^ not the blank one, because this was actually needed at both 200K and 300K (at 300K only needed in two of the Forms)
    symmetry_reduction_step_ = symmetry_reduction_step_smz_

    # make global all the constants
    for var_name in list_make_global:
        assert var_name in locals(), f'could not find {var_name}'
        globals()[var_name] = locals()[var_name]

####################################################################################################################
## MIV:

def mivebresib_tmFF_lr_():
    PME_cutoff = 0.5 ; KEY = f'_lr{PME_cutoff}'

    mol_name = 'mivebresib'
    PATH = DIR_main+f'MM/GAFF_sc/{mol_name}/'
    NPT_subfolder = 'tmff_lr'

    single_molecule_PDB = PATH+f'{mol_name}_single_mol.pdb'
    n_atoms_mol = 51

    FF_class = tmFF
    FF_name = FF_class.FF_name

    supercell_details = dict(supercell_details_mivebresib)
    list_Forms = list(supercell_details['list_Forms'])
    for Form in list_Forms:
        supercell_details[Form]['PDB_unitcell'] = PATH+f'{mol_name}_{Form}_unitcell.pdb'
    supercell_details['I']['list_n_mol'] = [16]
    supercell_details['II']['list_n_mol'] = [16]
    supercell_details['III']['list_n_mol'] = [16]

    ind_rO = 22 ; option = 0
    symmetry_reduction_step_ = symmetry_reduction_step_miv_
    
    #### #### #### #### #### #### #### #### #### #### 
    steps_followed = ''

    steps_followed += f'''{color_text_('MIV with tmFF (0.5nm) (PART 1):','I')}'''
    steps_followed += '''
    # steps done before this .py file was made (more difficult to track back what happened)

        # collecting the main set of NPT data for MBAR
        NPT_(
        _list_Forms = ['I','II','III'],
        _Ts = [200, 250, 300, 350, 400, 450, 500, 550], # i think 600K was here also but forgetting why it was not used for MBAR
        _list_n_mol = {'I':[16],'II':[16],'III':[16],},
        NPT_subfolder = NPT_subfolder,
        n_frames_warmup = 50000,   # this may be too much equilibration but just in case this was used
        n_frames_sample = 200000,  # only the second half of this trajectory was used for MBAR, this was too long.
        n_frames_stride = 50,      # this was used, but could x2 this and /2 n_frames_sample (as in abt72)
        save_equilibrated_supercell = True,

        checking = True, # set False if re-running
        )

        # running NVT for PGM (NVT version) training data:
        NVT_(
        _list_Forms  = ['I', 'II', 'III'],
        _Ts  = [200, 300], # 400K was also ran for 500k/10k = 50ns but PGM did not get a good enough error-bar with this amount of data
        _list_n_mol = {'I':[16],'II':[16],'III':[16],},

        n_frames_warmup = 5000,
        n_frames_sample = 500000,
        n_frames_stride = 50,

        checking = True, # set False if re-running
        checking_check_only_path = False,
        )
        
        # running NVT for PGM (NVT version) training data, showing that higher temperatures with PGM can be also useful with enough data.
        NVT_(
        _list_Forms  = ['I', 'II', 'III'],
        _Ts  = [400], # 400K was ran again with double the data to better converge PGM result (the main result in the notebook)
        _list_n_mol = {'I':[16],'II':[16],'III':[16],},

        n_frames_warmup = 5000,
        n_frames_sample = 1000000,
        n_frames_stride = 50,

        checking = True, # set False if re-running
        checking_check_only_path = False,
        )
    
        ## training PGM (NVT version) on the NVT datasets (only one Tref needed, but 3 were done to for a consistency-check)
        PGM_(
            _list_Forms = ['I','II','III'],
            _Ts         = [200, 300, 400],
            _list_n_mol = {'I':[16],'II':[16],'III':[16],},

            n_training_batches = 20000, # all training+evaluation runs are long (~8.5 hours) because learning rate was half the usual
            learning_rate = 0.0005,
            symmetry_reduction_step_ = symmetry_reduction_step_,
            PGM_key = '',
            NVT = True,

            checking = False, # set False if re-running
            )
        
        MBAR step shown in the notebook (the main result because it turned out the most cost-effective).

    '''

    steps_followed += f'''{color_text_('MIV with tmFF (0.5nm) (PART 2):','I')}'''
    steps_followed += '''
    # steps done using this .py file (it was possible to be more organised)

    # The following steps turned out mostly useful for a consitency-check related to further testing the NPT version of PGM:
    # (these are non-GUI steps)

        # ran long NPT simulations at Tref=200K for training data
        # (hoping to minimise PGM error-bar with more data)
        NPT_(
        _list_Forms = ['I','II','III'],
        _Ts = [200],
        _list_n_mol = {'I':[16],'II':[16],'III':[16],},
        NPT_subfolder = '',
        n_frames_warmup = 20000,
        n_frames_sample = 800000,
        n_frames_stride = 50,
        save_equilibrated_supercell = False,
        )

        ## training PGM (NPT version) on the Tref=200K data
        PGM_(
            _list_Forms = ['I','II','III'],
            _Ts         = [200],
            _list_n_mol = {'I':[16],'II':[16],'III':[16],},

            n_training_batches = 20000,
            learning_rate = 0.0005,
            symmetry_reduction_step_ = symmetry_reduction_step_,

            checking = False, # set False if re-running
            PGM_key = '_ext_setting_2',
            NVT = False,
            )

        # adding 225K and 275K data for MBAR 
        # (hoping to better connect with Tref=200K ensembles; minimising MBAR error) 
        NPT_(
        _list_Forms = ['I','II','III'],
        _Ts = [225, 275],
        _list_n_mol = {'I':[16],'II':[16],'III':[16],},
        NPT_subfolder = NPT_subfolder,
        n_frames_warmup = 20000,
        n_frames_sample = 100000,
        n_frames_stride = 50,
        save_equilibrated_supercell = False,
        )
    
        # recomputed *(brute force) the enthalpy evaluations for MBAR, and fit MBAR
        # to include the two new temperatures (225K, 275K)
        Tref = 200
        n_mol_Tref = 16
        n_mol_NPT = 16

        curves = {}
        for Form in list_Forms:
            curves[Form] = MBAR_(
                            Form = Form,
                            Tref_FEref_SEref = [Tref,0.,0.,
                                                ], # setting this to 0.0, as this only effects the RES output (recomputed properly in notebook)
                            list_Ts = [200, 225, 250, 275, 300, 350, 400, 450, 500, 550],
                            FEref_SEref_is_Helmholtz = False,
                            batch_size = 100000,
                            n_mol_NPT = n_mol_NPT,
                            NPT_subfolder = NPT_subfolder,
                            clear_memory = True,
                            xyz_not_in_datasets = False,
                            )

        # recomputed *(brute force) the enthalpy evaluations for MBAR, and fit MBAR
        # these enthalpies are missing the box Jacobean term.
        curves = {}
        for Form in list_Forms:
            curves[Form] = MBAR_(
                            Form = Form,
                            Tref_FEref_SEref = [Tref,0.,0., 
                                                ], # when using dummy Tref FEs like 0.0, Tref can also be anything here from the list below
                            list_Ts = [200, 225, 250, 275, 300, 350, 400, 450, 500, 550],
                            FEref_SEref_is_Helmholtz = True,
                            batch_size = 100000,
                            n_mol_NPT = n_mol_NPT,
                            NPT_subfolder = NPT_subfolder,
                            clear_memory = True,
                            xyz_not_in_datasets = False,
                            )

        # * TODO: minimise the brute force re-evalautions in Tx.py (i.e., compute enthalpy of a supercell only once)

        NB: All datasets were cut to save memory before moving to a GUI computer for running the notebook. Full datasets backed up.

    '''

    #### #### #### #### #### #### #### #### #### #### 
    # make global all the constants
    for var_name in list_make_global:
        assert var_name in locals(), f'could not find {var_name}'
        globals()[var_name] = locals()[var_name]

def mivebresib_gaff_lr_():
    PME_cutoff = 0.5 ; KEY = f'_lr{PME_cutoff}'

    mol_name = 'mivebresib'
    PATH = DIR_main+f'MM/GAFF_sc/{mol_name}/'
    NPT_subfolder = 'gaff_lr'

    single_molecule_PDB = PATH+f'{mol_name}_single_mol.pdb'
    n_atoms_mol = 51

    FF_class = GAFF
    FF_name = FF_class.FF_name

    supercell_details = dict(supercell_details_mivebresib)
    list_Forms = list(supercell_details['list_Forms'])
    for Form in list_Forms:
        supercell_details[Form]['PDB_unitcell'] = PATH+f'{mol_name}_{Form}_unitcell.pdb'
    supercell_details['I']['list_n_mol'] = [16]
    supercell_details['II']['list_n_mol'] = [16]
    supercell_details['III']['list_n_mol'] = [16]

    ind_rO = 22 ; option = 0
    symmetry_reduction_step_ = symmetry_reduction_step_miv_
    
    #### #### #### #### #### #### #### #### #### #### 
    steps_followed = ''

    #### #### #### #### #### #### #### #### #### #### 
    # make global all the constants
    for var_name in list_make_global:
        assert var_name in locals(), f'could not find {var_name}'
        globals()[var_name] = locals()[var_name]

""" i need to sort this out, some of this will be put as notebooks, two more needed VEL and ABT072
###################################################################################################################
## veliparib:

def veliparib_tmFF_sr_():
    PME_cutoff = 0.36 ; KEY = ''
    # short, default in paper3

    mol_name = 'veliparib'
    PATH = DIR_main+f'MM/molecules/{mol_name}/'
    NPT_subfolder = 'tmff_sr'

    single_molecule_PDB = PATH+f'{mol_name}_single_mol.pdb'
    n_atoms_mol = 34
    FF_class = velff
    FF_name = FF_class.FF_name

    supercell_details = dict(supercell_details_veliparib)
    list_Forms = supercell_details['list_Forms']
    for Form in list_Forms:
        supercell_details[Form]['PDB_unitcell'] = PATH+f'{mol_name}_{Form}_unitcell.pdb'
    supercell_details['I']['list_n_mol'] = [8, 16, 24, 32]
    supercell_details['II']['list_n_mol'] = [9, 12, 18, 24, 27]

    ind_rO = 5 ; option = 0

    #### #### #### #### #### #### #### #### #### #### 
    steps_followed = ''

    # all NPT simulations:
    steps_followed += f'''
        {color_text_('# NPT','B')}:
    '''
    _list_n_mol = dict(zip(list_Forms, [supercell_details[Form]['list_n_mol'] for Form in list_Forms]))
    steps_followed += f'''
        NPT_(
            _list_Forms = {list_Forms},
            _Ts         = [300],
            _list_n_mol = {_list_n_mol},
            NPT_subfolder  = '{NPT_subfolder}',

            T_init = None,
            n_frames_warmup = 20000,
            n_frames_sample = 50000,
            save_equilibrated_supercell = True,

            checking = True, # set False if re-running
            )
    '''
    _list_n_mol = dict(zip(list_Forms, [[24]]*len(list_Forms)))
    steps_followed += f'''
        NPT_(
            _list_Forms = {list_Forms},
            _Ts         = [100],
            _list_n_mol = {_list_n_mol},
            NPT_subfolder  = '{NPT_subfolder}',

            T_init = None,
            n_frames_warmup = 20000,
            n_frames_sample = 100000,
            save_equilibrated_supercell = True,

            checking = True, # set False if re-running
            )
    '''
    steps_followed += f'''
        NPT_(
            _list_Forms = {list_Forms},
            _Ts         = [200, 225, 250, 300, 350, 400, 450, 500, 550, 600],
            _list_n_mol = {_list_n_mol},
            NPT_subfolder  = '{NPT_subfolder}',

            T_init = 300,
            n_frames_warmup = 20000,
            n_frames_sample = 150000,
            save_equilibrated_supercell = True,

            checking = True, # set False if re-running
            )
    '''
    # T = 225K was added later.
    # save_equilibrated_supercell = True ; Could not find. These were not saved for temperatures at which NVT was not ran.

    #### #### #### #### #### #### 
    # all NVT simulations: 
    steps_followed += f'''
        {color_text_('# NVT','G')}:
    '''
    _list_n_mol = dict(zip(list_Forms, [supercell_details[Form]['list_n_mol'] for Form in list_Forms]))
    steps_followed += f'''
        NVT_( 
            _list_Forms = {list_Forms},
            _Ts         = [300],
            _list_n_mol = {_list_n_mol},

            n_frames_warmup = 1000,
            n_frames_sample = "n_mol * 18750",

            checking = True, # set False if re-running
            )
    '''
    _list_n_mol = dict(zip(list_Forms, [[24]]*len(list_Forms)))
    steps_followed += f'''
        NVT_(
            _list_Forms = {list_Forms},
            _Ts         = [100, 200],
            _list_n_mol = {_list_n_mol},

            n_frames_warmup = 1000,
            n_frames_sample = 24 * 18750,

            checking = True, # set False if re-running
            )
    '''

    #### #### #### #### #### #### 
    # all PGM runs:
    def symmetry_reduction_step_(path_dataset, Form, T, checking=True):
        '''
        just the methyl group because this did not rotate at 100K and slow at 200K
        '''
        if T > 150: sr_KEY = '_sym_reduced' # it was not used for 100 in this case because methyl groups did not rotate
        else:       sr_KEY = ''

        if checking or sr_KEY != '': pass
        else:
            if os.path.exists(path_dataset + sr_KEY):
                print(f'this {sr_KEY} dataset aready exists')
            else:
                sr = DatasetSymmetryReduction(path_dataset=path_dataset)
                sr.set_ABCD_(ind_rO, option=option)
                sr.sort_methyl_(lookup_indices = [0])
                sr.check_energy_(1000)
                sr.save_sym_reduced_dataset_(key = sr_KEY)

        return sr_KEY 

    steps_followed += f'''
        {color_text_('# PGM','G')}:
    '''
    _list_n_mol = dict(zip(list_Forms, [supercell_details[Form]['list_n_mol'] for Form in list_Forms]))
    steps_followed += f'''
        PGM_(
            _list_Forms = {list_Forms},
            _Ts         = [300],
            _list_n_mol = {_list_n_mol},

            n_training_batches = "min([n_mol * (15625//25),20000])",
            learning_rate = 0.001, # with 27 and 32 molecules, 0.0005 was used instead.
            symmetry_reduction_step_ = symmetry_reduction_step_blank_, # was not used here.

            checking = True, # set False if re-running
            )
    '''
    ''' * found this later by running: [this can be used to find most details about a previously trained model]
    for key in evaluation.keys():
        evaluation[key].load_model_()
        print(key, evaluation[key].model.learning_rate)
    '''
    _list_n_mol = dict(zip(list_Forms, [[24]]*len(list_Forms)))
    steps_followed += f'''
        PGM_(
            _list_Forms = {list_Forms},
            _Ts         = [100, 200, 300],
            _list_n_mol = {_list_n_mol},

            n_training_batches = "min([n_mol * (15625//25),20000])",
            learning_rate = 0.001, # * for temperatures 200 and 300, 0.0008 was used instead
            symmetry_reduction_step_ = symmetry_reduction_step_,

            checking = True, # set False if re-running
            )
    '''

    #### #### #### #### #### #### 
    # MBAR:
    steps_followed += f'''
        {color_text_('# MBAR','P')}:
    '''
    n_mol_Tref = 24
    _list_n_mol = dict(zip(list_Forms, [[n_mol_Tref]]*len(list_Forms)))
    steps_followed += f'''
        Tref = 200
        n_mol_Tref = {n_mol_Tref}
        n_mol_NPT  = 24
        evaluation = PGM_(
                    _list_Forms = {list_Forms},
                    _Ts         = [Tref],
                    _list_n_mol = {_list_n_mol},
                    n_training_batches = "info not needed for evalulation",
                    symmetry_reduction_step_ = symmetry_reduction_step_,
                    checking = True,
                    )
        curves = {"{}"}
        #for Form in list_Forms:
        Form = 'I'
        curves[Form] = MBAR_(
                        Form = Form,
                        Tref_FEref_SEref = [Tref,
                                            evaluation[(Form, Tref, n_mol_Tref)].nns[0].BAR_V_FE / n_mol_Tref, # absolute_lattice_FE
                                            evaluation[(Form, Tref, n_mol_Tref)].nns[0].BAR_V_SE / n_mol_Tref, # absolute_lattice_SE
                                            ],

                        list_Ts = [200, 225, 250, 300, 350, 400, 450, 500, 550, 600],
                        batch_size = 50000,
                        n_mol_NPT = n_mol_NPT,
                        NPT_subfolder  = '{NPT_subfolder}',
                        clear_memory = True,
                        xyz_not_in_datasets = True, # set False if running for the first time
                        )

        Form = 'II'
        curves[Form] = MBAR_(
                        Form = Form,
                        Tref_FEref_SEref = [Tref,
                                            evaluation[(Form, Tref, n_mol_Tref)].nns[0].BAR_V_FE / n_mol_Tref, # absolute_lattice_FE
                                            evaluation[(Form, Tref, n_mol_Tref)].nns[0].BAR_V_SE / n_mol_Tref, # absolute_lattice_SE
                                            ],
                        list_Ts = [200, 225, 250, 300, 350, 400, 450, 500, 550], # Form II melted at 600K (with PME_cutoff={PME_cutoff}nm)
                        batch_size = 50000,
                        n_mol_NPT = n_mol_NPT,
                        NPT_subfolder  = '{NPT_subfolder}',
                        clear_memory = True,
                        xyz_not_in_datasets = True, # set False if running for the first time
                        )
        ax = plot_curves_(curves, y_lim=[-1,4], y_lim_enthalpy=[-1,6])
    '''
    # make global
    for var_name in list_make_global:
        assert var_name in locals(), f'could not find {var_name}'
        globals()[var_name] = locals()[var_name]


def veliparib_tmFF_lr_():
    PME_cutoff = 0.72 ; KEY = f'_lr{PME_cutoff}'

    mol_name = 'veliparib'
    PATH = DIR_main+f'MM/molecules/{mol_name}/'
    NPT_subfolder = 'tmff_lr'

    single_molecule_PDB = PATH+f'{mol_name}_single_mol.pdb'
    n_atoms_mol = 34

    FF_class = velff
    FF_name = FF_class.FF_name

    supercell_details = dict(supercell_details_veliparib)
    list_Forms = supercell_details['list_Forms']
    for Form in list_Forms:
        supercell_details[Form]['PDB_unitcell'] = PATH+f'{mol_name}_{Form}_unitcell.pdb'
    supercell_details['I']['list_n_mol'] = [48]
    supercell_details['II']['list_n_mol'] = [48]

    ind_rO = 5 ; option = 0

    #### #### #### #### #### #### #### #### #### #### 
    steps_followed = ''

    steps_followed += f'''
        {color_text_('# NPT','B')}:
    '''
    steps_followed += '''
    NPT_(
        _list_Forms = ['I', 'II'],
        _Ts         = [100, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600],
        _list_n_mol = {'I': [48], 'II': [48]},
        NPT_subfolder = NPT_subfolder,

        T_init = None,
        n_frames_warmup = 50000,
        n_frames_sample = 100000,
        n_frames_stride = 50,
        save_equilibrated_supercell = True,

        checking  = True, # set False if re-running,
    )
    '''
    #### #### #### #### #### #### 
    steps_followed += f'''
        {color_text_('# NVT','G')}:
    '''
    steps_followed += '''
        NVT_(
            _list_Forms = ['I', 'II'],
            _Ts         = [100, 200, 300],
            _list_n_mol = {'I': [48], 'II': [48]},

            n_frames_warmup = 5000,
            n_frames_sample = 800000, # ???
            n_frames_stride = 50,

            checking = True, # set False if re-running
            )
    '''
    #### #### #### #### #### #### 
    def  symmetry_reduction_step_(path_dataset, Form, T, checking=True):
        '''
        just the methyl group because this did not rotate at 100K and slow at 200K
        '''
        sr_KEY = '_sym_reduced'
        if checking or sr_KEY != '': pass
        else:
            if os.path.exists(path_dataset + sr_KEY):
                print(f'this {sr_KEY} dataset aready exists')
            else:
                sr = DatasetSymmetryReduction(path_dataset=path_dataset)
                sr.set_ABCD_(ind_rO, option=option)
                sr.sort_methyl_(lookup_indices = [0])
                sr.check_energy_(1000)
                sr.save_sym_reduced_dataset_(key = sr_KEY)
 
        return sr_KEY 

    steps_followed += f'''
        {color_text_('# PGM','G')}:
    '''
    _list_n_mol = dict(zip(list_Forms, [supercell_details[Form]['list_n_mol'] for Form in list_Forms]))
    steps_followed += '''
        PGM_(
            _list_Forms = ['I', 'II'],
            _Ts         = [200],
            _list_n_mol = {'I': [48], 'II': [48]},

            n_training_batches = 0,
            learning_rate = 0.001,
            symmetry_reduction_step_ = symmetry_reduction_step_,

            PGM_key = '_ext',
            checking = True, # set False if re-running

            )
    '''

    # make global
    for var_name in list_make_global:
        assert var_name in locals(), f'could not find {var_name}'
        globals()[var_name] = locals()[var_name]

####################################################################################################################
## abt72:
def abt072_GAFF_lr_():
    PME_cutoff = 0.72 ; KEY = f'_lr{PME_cutoff}'

    mol_name = 'abt72'
    PATH = DIR_main+f'MM/molecules/{mol_name}/'
    NPT_subfolder = 'gaff_lr'

    single_molecule_PDB = PATH+f'{mol_name}_single_mol.pdb'
    n_atoms_mol = 60

    FF_class = GAFF
    FF_name = FF_class.FF_name

    supercell_details = dict(supercell_details_abt072)
    list_Forms = list(supercell_details['list_Forms'])
    for Form in list_Forms:
        supercell_details[Form]['PDB_unitcell'] = PATH+f'{mol_name}_{Form}_unitcell.pdb'
    supercell_details['C']['list_n_mol'] = [16]
    supercell_details['D']['list_n_mol'] = [16]
    supercell_details['F']['list_n_mol'] = [16]
    supercell_details['M']['list_n_mol'] = [16]

    ind_rO = 20 ; option = None

    #### #### #### #### #### #### #### #### #### #### 
    steps_followed = ''

    # all NPT simulations:
    steps_followed += f'''
        {color_text_('# NPT','B')}:
    '''

    Ts = [200, 250, 300, 350, 400, 450, 500, 550, 600]
    _list_n_mol = dict(zip(list_Forms, [[16]]*len(list_Forms)))
    steps_followed += f'''
        NPT_(
            _list_Forms = {list_Forms},
            _Ts         = {Ts},
            _list_n_mol = {_list_n_mol},
            NPT_subfolder = '{NPT_subfolder}',

            T_init = None,
            n_frames_warmup = 20000,
            n_frames_sample = 50000,
            n_frames_stride = 50,
            save_equilibrated_supercell = True,

            checking  = True, # set False if re-running
        )
    '''

    symmetry_reduction_step_ = 'add'

    # make global
    for var_name in list_make_global:
        assert var_name in locals(), f'could not find {var_name}'
        globals()[var_name] = locals()[var_name]


def abt072_GAFF_sr_():
    PME_cutoff = 0.43 ; KEY = f'_lr{PME_cutoff}'

    mol_name = 'abt72'
    PATH = DIR_main+f'MM/molecules/{mol_name}/'
    NPT_subfolder = 'gaff_sr'

    single_molecule_PDB = PATH+f'{mol_name}_single_mol.pdb'
    n_atoms_mol = 60

    FF_class = GAFF
    FF_name = FF_class.FF_name


    supercell_details = dict(supercell_details_abt072)
    list_Forms = list(supercell_details['list_Forms'])
    for Form in list_Forms:
        supercell_details[Form]['PDB_unitcell'] = PATH+f'{mol_name}_{Form}_unitcell.pdb'
    supercell_details['C']['list_n_mol'] = [8,16]
    supercell_details['D']['list_n_mol'] = [8,16]
    supercell_details['F']['list_n_mol'] = [8,16]
    supercell_details['M']['list_n_mol'] = [8,16]

    ind_rO = 20 ; option = None

    #### #### #### #### #### #### #### #### #### #### 
    steps_followed = ''

    # all NPT simulations:
    steps_followed += f'''
        {color_text_('# NPT','B')}:
    '''

    Ts = [200, 250, 300, 350, 400, 450, 500, 550, 600]
    _list_n_mol = dict(zip(list_Forms, [[16]]*len(list_Forms)))
    steps_followed += f'''
        run_NPT_(
                _list_Forms = {list_Forms},
                _Ts         = {Ts},
                _list_n_mol = {_list_n_mol},
                NPT_subfolder = '{NPT_subfolder}',

                T_init = None,
                n_frames_warmup = 20000,
                n_frames_sample = 50000,
                n_frames_stride = 50,
                save_equilibrated_supercell = True,

                checking  = True, # set False if re-running
    )
    '''

    symmetry_reduction_step_ = 'add'

    # make global
    for var_name in list_make_global:
        assert var_name in locals(), f'could not find {var_name}'
        globals()[var_name] = locals()[var_name]


def abt072_tmFF_lr_():
    PME_cutoff = 0.72 ; KEY = f'_lr{PME_cutoff}'

    mol_name = 'abt72'
    PATH = DIR_main+f'MM/molecules/{mol_name}/'
    NPT_subfolder = 'tmff_lr'

    single_molecule_PDB = PATH+f'{mol_name}_single_mol.pdb'
    n_atoms_mol = 60

    FF_class = tmFF
    FF_name = FF_class.FF_name

    supercell_details = dict(supercell_details_abt072)
    list_Forms = list(supercell_details['list_Forms'])
    for Form in list_Forms:
        supercell_details[Form]['PDB_unitcell'] = PATH+f'{mol_name}_{Form}_unitcell.pdb'
    supercell_details['C']['list_n_mol'] = [16]
    supercell_details['D']['list_n_mol'] = [16]
    supercell_details['F']['list_n_mol'] = [16]
    supercell_details['M']['list_n_mol'] = [16]

    supercell_details['E']['list_n_mol'] = [16]

    ind_rO = 20 ; option = None
  
    #### #### #### #### #### #### #### #### #### #### 
    steps_followed = ''

    # all NPT simulations:
    steps_followed += f'''
        {color_text_('# NPT','B')}:
    '''

    Ts = [200, 250, 300, 350, 400, 450, 500, 550, 600]
    _list_n_mol = dict(zip(list_Forms, [[16]]*len(list_Forms)))
    steps_followed += f'''
        NPT_(
                _list_Forms = {list_Forms},
                _Ts         = {Ts},
                _list_n_mol = {_list_n_mol},
                NPT_subfolder = '{NPT_subfolder}',

                T_init = None,
                n_frames_warmup = 25000,
                n_frames_sample = 50000,
                n_frames_stride = 200,
                save_equilibrated_supercell = True,

                checking  = True, # set False if re-running
    )
    '''
    steps_followed += f'''
        NPT_(
                _list_Forms = {list_Forms},
                _Ts         = {Ts},
                _list_n_mol = {_list_n_mol},
                NPT_subfolder = '{NPT_subfolder}',

                T_init = None,
                n_frames_warmup = 'not needed here',
                n_frames_sample = 50000,
                n_frames_stride = 400,
                save_equilibrated_supercell = True,

                checking  = True, # set False if re-running
                checking_check_only_path = False,
                rerun_unconverged = True,
                overwrite = True,
    )
    '''

    steps_followed += f'''
        {color_text_('# MBAR','P')}:

        curves = {"{}"}
        Tref = 200
        n_mol_NPT = 16
        for Form in list_Forms:
            curves[Form] = MBAR_(
                            Form = Form,
                            Tref_FEref_SEref = [Tref,
                                                0.0,
                                                0.0,
                                                ],
                            list_Ts = [200, 250, 300, 350, 400, 450, 500, 550, 600],
                            batch_size = 50000,
                            n_mol_NPT = n_mol_NPT,
                            NPT_subfolder  = '{NPT_subfolder}',
                            clear_memory = True,
                            xyz_not_in_datasets = False,
                            get_result = True, # delete this after checking
                        )
    '''

    steps_followed += f'''
        {color_text_('# PGM','G')}:
    '''
    symmetry_reduction_step_ = symmetry_reduction_step_abt072_

    _Ts = [200]
    _list_n_mol = dict(zip(list_Forms, [[16]]*len(list_Forms)))
    steps_followed += f'''
        PGM_(
            _list_Forms = {list_Forms},
            _Ts         = {_Ts},
            _list_n_mol = {_list_n_mol},

            n_training_batches = 10000,
            learning_rate = 0.0005,
            symmetry_reduction_step_ = symmetry_reduction_step_abt072_,

            checking = True, # set False if re-running
            )
    '''

    # make global
    for var_name in list_make_global:
        assert var_name in locals(), f'could not find {var_name}'
        globals()[var_name] = locals()[var_name]

"""
####################################################################################################################
## plotting:

Form_to_color_ = lambda Form : dict(zip(list_Forms, [f'C{i}' for i in range(len(list_Forms))]))[Form]

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

def plot_PGM_results_(evaluation, window=3, plot_raw_errors = True, figsize=(10,4)):
    keys = evaluation.keys()
    Ts = np.array(list(set([key[1] for key in keys])))
    Ts = np.sort(Ts)
    T2index = dict(zip(Ts,np.arange(len(Ts))))
    fig, ax = plt.subplots(1, max([len(Ts), 2]), figsize=figsize)
    for key in keys:
        print(f'Form = {key[0]}, T = {key[1]}, n_mol = {key[2]}')
        _ax = ax[T2index[key[1]]]
        evaluation[key].plot_result_(n_mol = key[-1],
                                                    colors=[Form_to_color_(key[0])]*4,
                                                    window=window,
                                                    ax = _ax,
                                                    plot_raw_errors = plot_raw_errors,
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
                 loc_Forms = 'upper center',
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
    
    keys = [_ for _ in curves.keys()] # list_Forms
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
    #                     (NPT temperature grids different can be e.g., if one Form melted or changed state during NPT, but not other Forms)
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
        color   = Form_to_color_(keys[i]) # f'C{i}'
        delta_u = np.array(list_curve_u[i] - list_curve_u[ind_min])
        if show_pure_mbar_error:
            if units_kJ_per_mol:
                ax[1].plot(Ts_curve, delta_u * Ts_curve*CONST_kB, alpha=1.0, color= color, linestyle='--')
            else:
                ax[1].plot(Ts_curve, delta_u, alpha=1.0, color= color, linestyle='--')
        else: pass

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
        line, = ax[0].plot([],[], color=f'C{i}', label=f'Form {keys[i]}')
        lines.append(line)
    Forms_legend = ax[0].legend(handles=lines, loc=loc_Forms, fontsize=7)
    ax[0].add_artist(Forms_legend)

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
                       linewidth = 1,
                       dashes = [1,5],
                       units_kJ_per_mol=True,
                ):

    keys = [_ for _ in curves.keys()] # list_Forms
    n_states = len(keys)
    
    Ts_curve = curves[keys[0]].RES['curve']['grid']
    assert all([np.abs(curves[key].RES['curve']['grid'] - Ts_curve).max()==0 for key in keys])
    
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
            ax[0].plot(Ts_curve, (delta_g - delta_g_se) * Ts_curve*CONST_kB, color='black', alpha=alpha, zorder=-10, linewidth=linewidth, dashes=dashes)
            ax[0].plot(Ts_curve, (delta_g + delta_g_se) * Ts_curve*CONST_kB, color='black', alpha=alpha, zorder=-10, linewidth=linewidth, dashes=dashes)
        else: 
            ax[0].plot(Ts_curve, (delta_g - delta_g_se), color='black', alpha=alpha, zorder=-10, linewidth=linewidth, dashes=dashes)
            ax[0].plot(Ts_curve, (delta_g + delta_g_se), color='black', alpha=alpha, zorder=-10, linewidth=linewidth, dashes=dashes)

    return ax

####################################################################################################################

""" 
'''
The same as run_NPT but with restrained torsion during simulation, 
saved in a different NPT_subfolder. Used only in veliparib, tmFF, PME_cutoff 0.36 (short) ; prevented Form II melting at 600K.

The main difference from run_NPT highlighted between ####################
'''
def run_NPT_with_restraint_(
            _list_Forms : list,
            _Ts : list,
            _list_n_mol : dict,
            NPT_subfolder = '',

            T_init = None,
            n_frames_warmup = 50000,
            n_frames_sample = 200000,
            save_equilibrated_supercell = True, # turn off when rerunning a dataset 

            inds_torsion_restrained = [26, 20, 18, 15],
            restraint_centres = [np.pi],
            restraint_width = 29.0,

            checking = False, # just to check the files were not lost
            ):
    ''' no GUI '''

    for Form in _list_Forms:
        for T in _Ts:
            for n_mol in _list_n_mol[Form]:

                cell = supercell_details[Form]['supercells'][n_mol] ; cell_str = cell_to_cell_str_(cell)
                verbose_message = f'Form={Form}, cell={cell} (n_mol={n_mol}), T={T}, FF={FF_name}'
                print('starting NPT simulation:', verbose_message)

                if T_init is not None:
                    PDB_initial = FFname_Form_cell_T_to_eqm_supercell_PDB_(FF_name, Form=Form, cell=cell, T=T_init, key=KEY)
                    assert os.path.exists(PDB_initial), '! no equilibrated supercell found for: Form={Form}, cell={cell} (n_mol={n_mol}), T={T_init}, FF={FF_name}'
                    print(f'starting NPT simulation from supercell that was already equilibrated at T={T_init}')
                else:
                    PDB_initial = Form_cell_to_ideal_supercell_PDB_(Form, cell)
                    assert os.path.exists(PDB_initial), f'! no ideal supercell found for: Form={Form}, n_mol={n_mol}'
                    print(f'starting NPT simulation from ideal supercell')

                path_NPT_dataset = f'{PATH}/data/NPT/{NPT_subfolder}/{mol_name}_{FF_name.lower()}_NPT_dataset_Form_{Form}_Cell_{cell_str}_Temp_{T}' + KEY
                path_equilibrated_supercell = FFname_Form_cell_T_to_eqm_supercell_PDB_(FF_name, Form, cell, T, key=KEY)

                if checking:
                    print(f'checking path path_NPT_dataset: {path_NPT_dataset}')
                    print('path exists:', os.path.exists(path_NPT_dataset))
                    print(f'checking path path_equilibrated_supercell: {path_equilibrated_supercell}')
                    print('path exists:', os.path.exists(path_equilibrated_supercell))
                
                else:
                    try:
                        assert FF_class.FF_name == FF_name, ''
                        sc = SingleComponent(PDB=PDB_initial, n_atoms_mol=n_atoms_mol, name=mol_name, FF_class=FF_class)
                        sc.initialise_system_(PME_cutoff=PME_cutoff, nonbondedMethod=app.PME)

                        ####################
                        bias = BIAS(sc)
                        bias.add_bias_(WALLS,
                                    name = 'walls1', # can be any text (if more than one torsion restraints can use different names)
                                    inds_torsion     = inds_torsion_restrained,
                                    means            = restraint_centres, 
                                        # called 'means' because these are average positions in a restrained simulation
                                        # position around which what torsional angle to spend the most time
                                    width_percentage = restraint_width,   
                                        # % of 2*pi interval allowed for the torsion to explore (freely without bias)
                                        # the 'walls' preventing exploration beyond this region 
                                        # should coincide with the natural FE barriers of the given torsion
                                        # # this is necessary to maximise the sample size, since all biased data is discarded during saving (below)
                                    )
                        ####################
                        
                        sc.initialise_simulation_(timestep_ps = 0.002, P = 1, T = T) # ps, atm, K
                        sc.simulation.step(n_frames_warmup * 50)
                        sc.run_simulation_(n_frames_sample, 50)

                        ####################
                        bias.save_simulation_data_zero_bias_(path_NPT_dataset)
                        ####################

                        TestConverged_1D(sc.u)

                        if save_equilibrated_supercell:
                            print('equilibrated supercell PDB is being extracted and saved:')
                            sc._xyz = tidy_crystal_xyz_(sc.xyz, sc.boxes, n_atoms_mol=sc.n_atoms_mol, ind_rO=ind_rO)
                            index = get_index_average_box_automatic_(sc.boxes)
                            r1 = sc.xyz[index] ; b1 = sc.boxes[index]
                            r1 = sc.minimise_xyz_(r1, b=b1, verbose=True)
                            sc.save_pdb_(r1, b=b1, name=path_equilibrated_supercell)
                        else: pass
                        
                        del sc
                        print('finished NPT simulation:', verbose_message)
                    except:
                        print('equilibration problem:', verbose_message)

                print('##################################################################################')
    print('done')
"""



