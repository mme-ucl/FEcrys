.. _api-O-MM-sc_system:

O.MM.sc_system
==============

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py>`__

.. warning:: Module docstring pending.


Classes and functions
---------------------

``SingleComponent`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L10>`__

.. code-block:: python

   class SingleComponent(PDB: str, n_atoms_mol: int, name: str, FF_name: str='GAFF', atom_order_PDB_match_itp=False, FF_class=None)

.. warning:: Docstring pending.


``SingleComponent.__new__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L11>`__

.. code-block:: python

   def __new__(cls, PDB, n_atoms_mol, name, FF_name='GAFF', atom_order_PDB_match_itp=False, FF_class=None)

.. warning:: Docstring pending.


``SingleComponent.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L28>`__

.. code-block:: python

   def __init__(self, PDB: str, n_atoms_mol: int, name: str, FF_name: str='GAFF', atom_order_PDB_match_itp=False, FF_class=None)

.. warning:: Docstring pending.


``SingleComponent.set_rdkit_mol_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L100>`__

.. code-block:: python

   def set_rdkit_mol_(self)

.. rubric:: Docstring

.. code-block:: text

   not used in this class, just to check atom indices


``SingleComponent.print`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L107>`__

.. code-block:: python

   def print(self, *text, verbose=False)

.. warning:: Docstring pending.


``SingleComponent.initialise_system_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L112>`__

.. code-block:: python

   def initialise_system_(self, PME_cutoff: float=None, removeCMMotion: bool=True, nonbondedMethod=app.PME, custom_EwaldErrorTolerance=0.0001, constraints=None, SwitchingFunction_factor=0.95)

.. warning:: Docstring pending.


``SingleComponent.initialise_simulation_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L215>`__

.. code-block:: python

   def initialise_simulation_(self, rbv=None, minimise=True, T=300.0, timestep_ps=0.0005, collision_rate=1, P=None, barostat_type=2, barostat_1_scaling=[True, True, True], stride_barostat=25, custom_integrator=None)

.. warning:: Docstring pending.


``SingleComponent.save_simulation_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L302>`__

.. code-block:: python

   def save_simulation_data_(self, path_and_name: str)

.. rubric:: Docstring

.. code-block:: text

   simulation timescale = len(u) * stride_save_frame * timestep_ps


``SingleComponent.load_simulation_data_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L325>`__

.. code-block:: python

   def load_simulation_data_(path_and_name)

.. warning:: Docstring pending.


``SingleComponent.initialise_from_save_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L329>`__

.. code-block:: python

   def initialise_from_save_(path_and_name: str, resume_simulation=True, verbose=True)

.. warning:: Docstring pending.


``concatenate_datasets_`` (function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L353>`__

.. code-block:: python

   def concatenate_datasets_(paths_datasets: list, remove_warmup=None, stride=1)

.. warning:: Docstring pending.


``LJ`` (class)
^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L388>`__

.. code-block:: python

   class LJ()

.. warning:: Docstring pending.


``LJ.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L390>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``LJ.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L395>`__

.. code-block:: python

   def FF_name(self)

.. warning:: Docstring pending.


``LJ._single_mol_pdb_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L399>`__

.. code-block:: python

   def _single_mol_pdb_file_(self)

.. warning:: Docstring pending.


``LJ.initialise_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L402>`__

.. code-block:: python

   def initialise_FF_(self)

.. warning:: Docstring pending.


``LJ.top_file_gmx_crys`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L411>`__

.. code-block:: python

   def top_file_gmx_crys(self)

.. warning:: Docstring pending.


``LJ.reset_n_mol_top_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L414>`__

.. code-block:: python

   def reset_n_mol_top_(self)

.. warning:: Docstring pending.


``LJ.set_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L429>`__

.. code-block:: python

   def set_FF_(self)

.. warning:: Docstring pending.


``TIP4P`` (class)
^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L436>`__

.. code-block:: python

   class TIP4P()

.. rubric:: Docstring

.. code-block:: text

   mixin class for SingleComponent. Methods relevant only for using TIP4P are here.


``TIP4P.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L440>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``TIP4P.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L445>`__

.. code-block:: python

   def FF_name(self)

.. warning:: Docstring pending.


``TIP4P._single_mol_pdb_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L449>`__

.. code-block:: python

   def _single_mol_pdb_file_(self)

.. warning:: Docstring pending.


``TIP4P.initialise_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L452>`__

.. code-block:: python

   def initialise_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   run this after (n_mol and n_atoms_mol) defined in __init__ of SingleComponent


``TIP4P.top_file_gmx_crys`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L468>`__

.. code-block:: python

   def top_file_gmx_crys(self)

.. warning:: Docstring pending.


``TIP4P.reset_n_mol_top_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L471>`__

.. code-block:: python

   def reset_n_mol_top_(self)

.. warning:: Docstring pending.


``TIP4P.set_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L487>`__

.. code-block:: python

   def set_FF_(self)

.. warning:: Docstring pending.


``TIP4P.add_v_sites_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L494>`__

.. code-block:: python

   def add_v_sites_(self, r)

.. warning:: Docstring pending.


``TIP4P.remove_v_sites_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L510>`__

.. code-block:: python

   def remove_v_sites_(self, r)

.. warning:: Docstring pending.


``TIP4P._current_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L527>`__

.. code-block:: python

   def _current_r_(self)

.. warning:: Docstring pending.


``TIP4P._current_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L533>`__

.. code-block:: python

   def _current_v_(self)

.. warning:: Docstring pending.


``TIP4P._current_F_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L539>`__

.. code-block:: python

   def _current_F_(self)

.. warning:: Docstring pending.


``TIP4P._set_r_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L544>`__

.. code-block:: python

   def _set_r_(self, r)

.. warning:: Docstring pending.


``TIP4P._set_v_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L548>`__

.. code-block:: python

   def _set_v_(self, v)

.. warning:: Docstring pending.


``TIP4P.forward_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L552>`__

.. code-block:: python

   def forward_atom_index_(self, inds)

.. warning:: Docstring pending.


``TIP4P._system_mass_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L557>`__

.. code-block:: python

   def _system_mass_(self)

.. warning:: Docstring pending.


``TIP4P.inverse_atom_index_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L561>`__

.. code-block:: python

   def inverse_atom_index_(self, inds)

.. warning:: Docstring pending.


``GAFF`` (class)
^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L566>`__

.. code-block:: python

   class GAFF()

.. rubric:: Docstring

.. code-block:: text

   mixin class for SingleComponent. Methods relevant only for using GAFF are here.

   attribes needed before running self.initialise_FF_

   self.misc_dir
   self.name
   .. add


``GAFF.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L576>`__

.. code-block:: python

   def __init__(self)

.. warning:: Docstring pending.


``GAFF.FF_name`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L581>`__

.. code-block:: python

   def FF_name(self)

.. warning:: Docstring pending.


``GAFF.initialise_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L584>`__

.. code-block:: python

   def initialise_FF_(self)

.. rubric:: Docstring

.. code-block:: text

   run this after (n_mol and n_atoms_mol) defined near the end of __init__ of SingleComponent
   this is at the end of __init__


``GAFF._single_mol_pdb_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L603>`__

.. code-block:: python

   def _single_mol_pdb_file_(self)

.. warning:: Docstring pending.


``GAFF._single_mol_prepi_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L606>`__

.. code-block:: python

   def _single_mol_prepi_file_(self)

.. warning:: Docstring pending.


``GAFF._single_mol_frcmod_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L609>`__

.. code-block:: python

   def _single_mol_frcmod_file_(self)

.. warning:: Docstring pending.


``GAFF._single_mol_mol2_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L612>`__

.. code-block:: python

   def _single_mol_mol2_file_(self)

.. warning:: Docstring pending.


``GAFF.first_time_molecule_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L615>`__

.. code-block:: python

   def first_time_molecule_(self)

.. warning:: Docstring pending.


``GAFF.can_run_tleap`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L676>`__

.. code-block:: python

   def can_run_tleap(self)

.. warning:: Docstring pending.


``GAFF.tleap_file`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L686>`__

.. code-block:: python

   def tleap_file(self)

.. warning:: Docstring pending.


``GAFF.prmtop_file`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L689>`__

.. code-block:: python

   def prmtop_file(self)

.. warning:: Docstring pending.


``GAFF._inpcrd_file_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L692>`__

.. code-block:: python

   def _inpcrd_file_(self)

.. warning:: Docstring pending.


``GAFF.create_prmtop_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L695>`__

.. code-block:: python

   def create_prmtop_(self)

.. warning:: Docstring pending.


``GAFF.top_file_gmx`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L729>`__

.. code-block:: python

   def top_file_gmx(self)

.. warning:: Docstring pending.


``GAFF.top_file_gmx_adjusted_charges`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L733>`__

.. code-block:: python

   def top_file_gmx_adjusted_charges(self)

.. warning:: Docstring pending.


``GAFF.top_file_gmx_crys`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L737>`__

.. code-block:: python

   def top_file_gmx_crys(self)

.. warning:: Docstring pending.


``GAFF.reset_n_mol_top_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L740>`__

.. code-block:: python

   def reset_n_mol_top_(self)

.. warning:: Docstring pending.


``GAFF.set_FF_amber_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L750>`__

.. code-block:: python

   def set_FF_amber_(self)

.. warning:: Docstring pending.


``GAFF.set_FF_gmx_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L754>`__

.. code-block:: python

   def set_FF_gmx_(self)

.. warning:: Docstring pending.


``GAFF.set_FF_`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/sc_system.py#L779>`__

.. code-block:: python

   def set_FF_(self)

.. warning:: Docstring pending.
