"""Evaluate batches of configurations in parallel OpenMM contexts.

Each worker process owns one OpenMM ``Context`` and receives configurations
through multiprocessing queues. The implementation is derived from bgflow's
OpenMM bridge. Inputs and returned values are plain NumPy-compatible arrays;
the units expected at this boundary are documented on :meth:`MultiContext.evaluate`.
"""

import warnings
import multiprocessing as mp

import numpy as np
import pickle

##

class MultiContext:
    """Manage OpenMM contexts running in independent worker processes.

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
    """

    def __init__(self, n_workers, system, integrator, platform_name, platform_properties={}):
        """Store context configuration and create empty task/result queues.

        Workers are started lazily on the first call to :meth:`evaluate`.
        Each worker receives a pickle-cloned integrator because an OpenMM
        integrator can belong to only one context.
        """
        self._n_workers = n_workers
        self._system = system
        self._integrator = integrator
        self._platform_name = platform_name
        self._platform_properties = platform_properties
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._workers = []  # workers are initialized in first evaluate call
        # using multiple workers
        try:
            get_ipython
            warnings.warn(
                "It looks like you are using an OpenMMBridge with multiple workers in an ipython environment. "
                "This can behave a bit silly upon KeyboardInterrupt (e.g., kill the stdout stream). "
                "If you experience any issues, consider initializing the bridge with n_workers=1 in ipython/jupyter.",
                UserWarning
            )
        except NameError:
            pass

    def _reinitialize(self):
        """Replace queues and start a fresh set of worker processes.

        Existing workers first receive a soft-termination sentinel. The method
        mutates this object in place and returns ``None``.
        """
        self.terminate()
        # recreate objects
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._workers = []
        for i in range(self._n_workers):
            worker = MultiContext.Worker(
                self._task_queue,
                self._result_queue,
                self._system, self._integrator,
                self._platform_name,
                self._platform_properties,
            )
            self._workers.append(worker)
            worker.start()

    def evaluate(
            self,
            positions,
            box_vectors=None,
            evaluate_energy=True,
            evaluate_force=False,
            evaluate_positions=False,
            evaluate_path_probability_ratio=False,
            err_handling="warning",
            n_simulation_steps=0
    ):
        """Delegate energy and force computations to the workers.

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
        """
        assert box_vectors is None or len(box_vectors) == len(positions), \
            "box_vectors and positions have to be the same length"
        if not self.is_alive():
            self._reinitialize()

        box_vectors = [None for _ in positions] if box_vectors is None else box_vectors
        try:
            for i, (p, bv) in enumerate(zip(positions, box_vectors)):
                self._task_queue.put([
                    i, p, bv, evaluate_energy, evaluate_force, evaluate_positions,
                    evaluate_path_probability_ratio, err_handling, n_simulation_steps
                ])
            results = [self._result_queue.get() for _ in positions]
        except Exception as e:
            self.terminate()
            raise e
        results = sorted(results, key=lambda x: x[0])
        return (
            np.array([res[1] for res in results]) if evaluate_energy else None,
            np.array([res[2] for res in results]) if evaluate_force else None,
            np.array([res[3] for res in results]) if evaluate_positions else None,
            np.array([res[4] for res in results]) if evaluate_path_probability_ratio else None
        )

    def is_alive(self):
        """Return true when at least one worker exists and all are alive."""
        return all(worker.is_alive() for worker in self._workers) and len(self._workers) > 0

    def terminate(self):
        """Request soft termination of every worker.

        One ``None`` sentinel is placed on the shared task queue per worker.
        The method does not join the processes or wait for their termination.
        """
        # soft termination
        for _ in self._workers:
            self._task_queue.put(None)
        # hard termination
        #for worker in self._workers:
        #    worker.terminate()

    def __del__(self):
        """Request worker termination when this manager is garbage-collected."""
        self.terminate()

    class Worker(mp.Process):
        """Process that evaluates queued states in one OpenMM context.

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
        """

        def __init__(self, task_queue, result_queue, system, integrator, platform_name, platform_properties):
            """Store queues and serializable OpenMM context configuration.

            The actual OpenMM ``Context`` is deliberately created by
            :meth:`run` after the process starts. Creating CPU contexts in the
            parent process can deadlock on some platforms.
            """
            super(MultiContext.Worker, self).__init__()
            self._task_queue = task_queue
            self._result_queue = result_queue
            self._openmm_system = system
            self._openmm_integrator = pickle.loads( pickle.dumps(integrator))
            self._openmm_platform_name = platform_name
            self._openmm_platform_properties = platform_properties
            self._openmm_context = None

        def run(self):
            """Consume tasks and return requested OpenMM state quantities.

            Positions and box vectors are received in nanometres. Energies,
            forces, and positions are returned in kJ/mol, kJ/(mol nm), and nm,
            respectively. A ``None`` task stops the loop. Evaluation errors are
            warned, ignored, or re-raised according to each task's
            ``err_handling`` value.
            """
            try:
                from openmm import unit
                from openmm import Platform, Context
            except ImportError: # fall back to older version < 7.6
                from simtk import unit
                from simtk.openmm import Platform, Context

            # create the context
            # it is crucial to do that in the run function and not in the constructor
            # for some reason, the CPU platform hangs if the context is created in the constructor
            # see also https://github.com/openmm/openmm/issues/2602
            openmm_platform = Platform.getPlatformByName(self._openmm_platform_name)
            self._openmm_context = Context(
                self._openmm_system,
                self._openmm_integrator,
                openmm_platform,
                self._openmm_platform_properties
            )
            self._openmm_context.reinitialize(preserveState=True)

            # get tasks from the task queue
            for task in iter(self._task_queue.get, None):
                (index, positions, box_vectors, evaluate_energy, evaluate_force,
                 evaluate_positions, evaluate_path_probability_ratio, err_handling, n_simulation_steps) = task
                try:
                    # initialize state
                    self._openmm_context.setPositions(positions)
                    if box_vectors is not None:
                        self._openmm_context.setPeriodicBoxVectors(*box_vectors)
                    #log_path_probability_ratio = self._openmm_integrator.step(n_simulation_steps)
                    log_path_probability_ratio = 0.0

                    # compute energy and forces
                    state = self._openmm_context.getState(
                        getEnergy=evaluate_energy,
                        getForces=evaluate_force,
                        getPositions=evaluate_positions
                    )
                    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) if evaluate_energy else None
                    forces = (
                        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                        if evaluate_force else None
                    )
                    new_positions = state.getPositions().value_in_unit(unit.nanometers) if evaluate_positions else None
                except Exception as e:
                    if err_handling == "warning":
                        warnings.warn("Suppressed exception: {}".format(e))
                    elif err_handling == "exception":
                        raise e

                # push energies and forces to the results queue
                self._result_queue.put(
                    [index, energy, forces, new_positions, log_path_probability_ratio]
                )
