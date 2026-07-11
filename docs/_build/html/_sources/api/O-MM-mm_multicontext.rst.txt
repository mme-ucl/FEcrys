.. _api-O-MM-mm_multicontext:

O.MM.mm_multicontext
====================

`View module on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py>`__

.. rubric:: Docstring

.. code-block:: text

   Evaluate batches of configurations in parallel OpenMM contexts.

   Each worker process owns one OpenMM ``Context`` and receives configurations
   through multiprocessing queues. The implementation is derived from bgflow's
   OpenMM bridge. Inputs and returned values are plain NumPy-compatible arrays;
   the units expected at this boundary are documented on :meth:`MultiContext.evaluate`.


Classes and functions
---------------------

``MultiContext`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L17>`__

.. code-block:: python

   class MultiContext(n_workers, system, integrator, platform_name, platform_properties={})

.. rubric:: Docstring

.. code-block:: text

   Manage OpenMM contexts running in independent worker processes.

   Parameters
   ----------
   n_workers : int
       The number of workers which operate one context each.
   system : openmm.System
       The system that contains all forces.
   integrator : openmm.Integrator
       An OpenMM integrator.
   platform_name : str
       The name of an OpenMM platform ('Reference', 'CPU', 'CUDA', or 'OpenCL')
   platform_properties : dict, optional
       A dictionary of platform properties.


``MultiContext.__init__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L34>`__

.. code-block:: python

   def __init__(self, n_workers, system, integrator, platform_name, platform_properties={})

.. rubric:: Docstring

.. code-block:: text

   Store context configuration and create empty task/result queues.

   Workers are started lazily on the first call to :meth:`evaluate`.
   Each worker receives a pickle-cloned integrator because an OpenMM
   integrator can belong to only one context.


``MultiContext._reinitialize`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L61>`__

.. code-block:: python

   def _reinitialize(self)

.. rubric:: Docstring

.. code-block:: text

   Replace queues and start a fresh set of worker processes.

   Existing workers first receive a soft-termination sentinel. The method
   mutates this object in place and returns ``None``.


``MultiContext.evaluate`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L83>`__

.. code-block:: python

   def evaluate(self, positions, box_vectors=None, evaluate_energy=True, evaluate_force=False, evaluate_positions=False, evaluate_path_probability_ratio=False, err_handling='warning', n_simulation_steps=0)

.. rubric:: Docstring

.. code-block:: text

   Delegate energy and force computations to the workers.

   Parameters
   ----------
   positions : numpy.ndarray
       The particle positions in nanometer; its shape is (batch_size, num_particles, 3).
   box_vectors : numpy.ndarray, optional
       The periodic box vectors in nanometer; its shape is (batch_size, 3, 3).
       If not specified, don't change the box vectors.
   evaluate_energy : bool, optional
       Whether to compute energies.
   evaluate_force : bool, optional
       Whether to compute forces.
   evaluate_positions : bool, optional
       Whether to return positions.
   evaluate_path_probability_ratio : bool, optional
       Whether to compute the log path probability ratio. Makes only sense for PathProbabilityIntegrator instances.
   err_handling : {"warning", "ignore", "exception"}, default="warning"
       How each worker handles an exception while evaluating a state.
   n_simulation_steps : int, optional
       If > 0, perform a number of simulation steps and compute energy and forces for the resulting state.

   Returns
   -------
   energies : np.ndarray or None
       The energies in units of kilojoule/mole; its shape  is (len(positions), )
   forces : np.ndarray or None
       The forces in units of kilojoule/mole/nm; its shape is (len(positions), num_particles, 3)
   new_positions : np.ndarray or None
       The positions in units of nm; its shape is (len(positions), num_particles, 3)
   log_path_probability_ratio : np.ndarray or None
       The logarithmic path probability ratios; its shape  is (len(positions), )

   Notes
   -----
   Results are reordered to match the input batch after parallel
   evaluation. Although ``n_simulation_steps`` and path-probability
   evaluation are part of the API, the current worker does not advance
   the integrator and returns a path-probability ratio of zero.


``MultiContext.is_alive`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L158>`__

.. code-block:: python

   def is_alive(self)

.. rubric:: Docstring

.. code-block:: text

   Return true when at least one worker exists and all are alive.


``MultiContext.terminate`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L162>`__

.. code-block:: python

   def terminate(self)

.. rubric:: Docstring

.. code-block:: text

   Request soft termination of every worker.

   One ``None`` sentinel is placed on the shared task queue per worker.
   The method does not join the processes or wait for their termination.


``MultiContext.__del__`` (method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L175>`__

.. code-block:: python

   def __del__(self)

.. rubric:: Docstring

.. code-block:: text

   Request worker termination when this manager is garbage-collected.


``MultiContext.Worker`` (class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L179>`__

.. code-block:: python

   class Worker(task_queue, result_queue, system, integrator, platform_name, platform_properties)

.. rubric:: Docstring

.. code-block:: text

   Process that evaluates queued states in one OpenMM context.

   Parameters
   ----------
   task_queue : multiprocessing.Queue
       The queue that the MultiContext pushes tasks to.
   result_queue : multiprocessing.Queue
       The queue that the MultiContext receives results from.
   system : openmm.System
       The system that contains all forces.
   integrator : openmm.Integrator
       An OpenMM integrator.
   platform_name : str
       The name of an OpenMM platform ('Reference', 'CPU', 'CUDA', or 'OpenCL')
   platform_properties : dict
       A dictionary of platform properties.


``MultiContext.Worker.__init__`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L198>`__

.. code-block:: python

   def __init__(self, task_queue, result_queue, system, integrator, platform_name, platform_properties)

.. rubric:: Docstring

.. code-block:: text

   Store queues and serializable OpenMM context configuration.

   The actual OpenMM ``Context`` is deliberately created by
   :meth:`run` after the process starts. Creating CPU contexts in the
   parent process can deadlock on some platforms.


``MultiContext.Worker.run`` (nested helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View source on GitHub <https://github.com/mme-ucl/FEcrys/blob/main/O/MM/mm_multicontext.py#L214>`__

.. code-block:: python

   def run(self)

.. rubric:: Docstring

.. code-block:: text

   Consume tasks and return requested OpenMM state quantities.

   Positions and box vectors are received in nanometres. Energies,
   forces, and positions are returned in kJ/mol, kJ/(mol nm), and nm,
   respectively. A ``None`` task stops the loop. Evaluation errors are
   warned, ignored, or re-raised according to each task's
   ``err_handling`` value.
