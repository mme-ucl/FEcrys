.. _api-O-MM-ff_setup:

O.MM.ff_setup
=============

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py>`__

.. warning:: Module docstring pending.


Classes and functions
---------------------

``change_charges_itp_top_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L6>`__

.. code-block:: python

   def change_charges_itp_top_(path_top_or_itp_file_in: str, path_top_or_itp_file_out: str, n_atoms_mol: int, replacement_charges: np.ndarray=None, neutralise_charge: bool=True)

.. warning:: Docstring pending.


``change_n_mol_top_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L77>`__

.. code-block:: python

   def change_n_mol_top_(path_top_file_in: str, path_top_file_out: str, replace_n_mol: int, verbose=True)

.. warning:: Docstring pending.


``methods_for_permutation`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L127>`__

.. code-block:: python

   class methods_for_permutation

.. warning:: Docstring pending.


``methods_for_permutation._current_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L133>`__

.. code-block:: python

   def _current_r_(self)

.. warning:: Docstring pending.


``methods_for_permutation._current_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L139>`__

.. code-block:: python

   def _current_v_(self)

.. warning:: Docstring pending.


``methods_for_permutation._current_F_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L145>`__

.. code-block:: python

   def _current_F_(self)

.. warning:: Docstring pending.


``methods_for_permutation._system_mass_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L151>`__

.. code-block:: python

   def _system_mass_(self)

.. warning:: Docstring pending.


``methods_for_permutation._set_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L155>`__

.. code-block:: python

   def _set_r_(self, r)

.. warning:: Docstring pending.


``methods_for_permutation._set_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L159>`__

.. code-block:: python

   def _set_v_(self, v)

.. warning:: Docstring pending.


``methods_for_permutation.forward_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L163>`__

.. code-block:: python

   def forward_atom_index_(self, inds)

.. warning:: Docstring pending.


``methods_for_permutation.inverse_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L167>`__

.. code-block:: python

   def inverse_atom_index_(self, inds)

.. warning:: Docstring pending.


``methods_for_permutation.set_arrays_blank_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L173>`__

.. code-block:: python

   def set_arrays_blank_(self)

.. warning:: Docstring pending.


``methods_for_permutation.save_frame_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L184>`__

.. code-block:: python

   def save_frame_(self)

.. warning:: Docstring pending.


``methods_for_permutation.run_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L194>`__

.. code-block:: python

   def run_simulation_(self, n_saves, stride_save_frame: int=100, verbose_info: str='')

.. warning:: Docstring pending.


``methods_for_permutation.run_simulation_w_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L206>`__

.. code-block:: python

   def run_simulation_w_(self, n_saves, stride_save_frame: int=100, verbose_info: str='')

.. warning:: Docstring pending.


``methods_for_permutation.u_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L220>`__

.. code-block:: python

   def u_(self, r, b=None)

.. rubric:: Docstring

.. code-block:: text

   speed up evaluation also # ..


``itp2FF`` (class)
^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L248>`__

.. code-block:: python

   class itp2FF()

.. warning:: Docstring pending.


``itp2FF.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L249>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``itp2FF.itp_mol`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L254>`__

.. code-block:: python

   def itp_mol(self)

.. warning:: Docstring pending.


``itp2FF.itp_mol_adjusted_charges`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L258>`__

.. code-block:: python

   def itp_mol_adjusted_charges(self)

.. warning:: Docstring pending.


``itp2FF.top_crys`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L263>`__

.. code-block:: python

   def top_crys(self)

.. warning:: Docstring pending.


``itp2FF.gro_mol`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L267>`__

.. code-block:: python

   def gro_mol(self)

.. warning:: Docstring pending.


``itp2FF.pdb_mol`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L271>`__

.. code-block:: python

   def pdb_mol(self)

.. warning:: Docstring pending.


``itp2FF.single_mol_pdb`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L275>`__

.. code-block:: python

   def single_mol_pdb(self)

.. warning:: Docstring pending.


``itp2FF._single_mol_pdb_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L278>`__

.. code-block:: python

   def _single_mol_pdb_file_(self)

.. warning:: Docstring pending.


``itp2FF.single_mol_pdb_permuted`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L282>`__

.. code-block:: python

   def single_mol_pdb_permuted(self)

.. warning:: Docstring pending.


``itp2FF.single_mol_permutations`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L286>`__

.. code-block:: python

   def single_mol_permutations(self)

.. warning:: Docstring pending.


``itp2FF.set_pemutation_to_match_topology_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L289>`__

.. code-block:: python

   def set_pemutation_to_match_topology_(self)

.. warning:: Docstring pending.


``itp2FF.a_step_after_initialise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L345>`__

.. code-block:: python

   def a_step_after_initialise_(self)

.. warning:: Docstring pending.


``itp2FF.initialise_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L348>`__

.. code-block:: python

   def initialise_FF_(self, neuralise_net_charge=False, replacement_charges=None)

.. rubric:: Docstring

.. code-block:: text

   run this only after (n_mol and n_atoms_mol) defined in __init__ of SingleComponent


``itp2FF.set_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L373>`__

.. code-block:: python

   def set_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   run this just before self.system initialisation


``itp2FF.reset_n_mol_top_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L379>`__

.. code-block:: python

   def reset_n_mol_top_(self)

.. warning:: Docstring pending.


``OPLS_general`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L405>`__

.. code-block:: python

   class OPLS_general()

.. warning:: Docstring pending.


``OPLS_general.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L406>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``OPLS_general.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L416>`__

.. code-block:: python

   def FF_name(self)

.. warning:: Docstring pending.


``GAFF_general`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L419>`__

.. code-block:: python

   class GAFF_general()

.. warning:: Docstring pending.


``GAFF_general.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L420>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``GAFF_general.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L430>`__

.. code-block:: python

   def FF_name(self)

.. warning:: Docstring pending.


``remove_force_by_names_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L435>`__

.. code-block:: python

   def remove_force_by_names_(system, names: list, verbose=True)

.. warning:: Docstring pending.


``remove_force_by_names_.remove_force_by_name_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L436>`__

.. code-block:: python

   def remove_force_by_name_(_name)

.. warning:: Docstring pending.


``_get_pairs_mol_inner_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L452>`__

.. code-block:: python

   def _get_pairs_mol_inner_(single_mol_pdb_file, n=3)

.. rubric:: Docstring

.. code-block:: text

   n atoms in a row is n-1 bonds

   n = 3 for this velff because nrexcl = 2
       within 2 bonds away removed
       within 3 bonds away kept (1-2-3-4 ; 1-4 kept)


``_get_pairs_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L489>`__

.. code-block:: python

   def _get_pairs_(remove_mol_ij, n_mol)

.. warning:: Docstring pending.


``custom_LJ_force_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L501>`__

.. code-block:: python

   def custom_LJ_force_(sc, C6_C12_types_dictionary)

.. rubric:: Docstring

.. code-block:: text

   LJ : all Lennard-Jones


``custom_C_force_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L570>`__

.. code-block:: python

   def custom_C_force_(sc)

.. rubric:: Docstring

.. code-block:: text

   C : all Coulombic


``tmFF`` (class)
^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L647>`__

.. code-block:: python

   class tmFF()

.. rubric:: Docstring

.. code-block:: text

   tailor-made FF


``tmFF.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L649>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``tmFF.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L659>`__

.. code-block:: python

   def FF_name(self)

.. warning:: Docstring pending.


``tmFF.a_step_after_initialise_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L662>`__

.. code-block:: python

   def a_step_after_initialise_(self)

.. warning:: Docstring pending.


``tmFF.recast_NB_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L691>`__

.. code-block:: python

   def recast_NB_(self, verbose=True)

.. warning:: Docstring pending.


``tmFF.corrections_to_ff_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L706>`__

.. code-block:: python

   def corrections_to_ff_(self, verbos=True)

.. warning:: Docstring pending.


``velff`` (class)
^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L709>`__

.. code-block:: python

   class velff()

.. rubric:: Docstring

.. code-block:: text

   keeping seperate to load pickled files


``velff.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L711>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``velff.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/ff_setup.py#L721>`__

.. code-block:: python

   def FF_name(self)

.. warning:: Docstring pending.
