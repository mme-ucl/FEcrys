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
            list_n_mol = [x for x in self.supercell_details[form].keys() if x <= max_n_mol]
            print(f'form {form} with considered list_n_mol = {list_n_mol}')
            for n_mol in list_n_mol:
                if min_n_mol <= n_mol <=  max_n_mol:
                    cell = self.supercell_details[form][n_mol]
                    cell_str = cell_to_cell_str_(cell)
                    PDB_supercell = self.files[form+f'_ideal_supercell_{n_mol}']
                    boxes.append(PDB_to_box_(PDB_supercell))
                else:
                    pass
                    #print(f'form {form}, no supercell with : (min_n_mol={min_n_mol}) <= (n_mol={ n_mol}) <= (max_n_mol={max_n_mol})')

        boxes = np.stack(boxes)
        cutoff_max = np.min([boxes[...,i,i].min() for i in range(3)]) * 0.5
        print(f'maximum PME cutoff (d_max) possible with current set of list_n_mol selections: {cutoff_max.round(3)}nm')
        #print(f'PME cutoff used: {PME_cutoff}nm')

    ## ## ## ## 

    def create_NPT_subfolder_(self, NPT_subfolder_name):
        os.mkdir(f'./O/MM/molecules/{self.name}/data/NPT/{NPT_subfolder_name}')

    def FFname_Form_cell_T_to_eqm_supercell_PDB_(self, 
                                                 FFname, Form, cell, T, key='',
                                                 ):
        return f'{self.molecules_folder}/{self.name}/{self._name}_{FFname.lower()}_equilibrated_Form_{Form}_Cell_{cell_to_cell_str_(cell)}_Temp_{T}{key}.pdb'

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
        super().__init__(name)

    def set_FF_(self, FF_class):
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


        '''
        fix the chosen cutoff and n_mol settings
        '''

    '''
    run the functions from project settings from here using self.constants .. (currently as globals in those functions)
    TODO: copy them here and test them on a new project
    '''

