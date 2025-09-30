from typing import override, Optional
from dataclasses import dataclass, field

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve

from luna_quantum.algorithms import Kerberos
from luna_quantum.solve.parameters.algorithms.base_params import (
    Decomposer,
    QuantumAnnealingParams,
    SimulatedAnnealingBaseParams,
    TabuKerberosParams
)
from luna_quantum.backends import DWaveQpu

from .utils import get_best_solution, get_model, scale_sleep,   get_runtime, get_luna_api_key, get_dwave_token


@dataclass
class LUNAKerberos(Core):
    """
    A module for solving a QUBO problem using the Kerberos hybrid quantum-classical optimization solver.
    
    Kerberos divides the problem into subproblems and solves them using Tabu Search, Simulated Annealing 
    and QPU Subproblem Sampling. These algorithms are executed in parallel and afterwards the best solutions 
    are combined. This procedure is applied iteratively until the best solution is found or a termination 
    criterion is met. This approach leverages both classical and quantum resources efficiently, making it 
    effective for large and complex optimization problems beyond the capacity of pure quantum approaches.
    
    For additional information refer to: https://docs.aqarios.com/algorithms/kerberos/

    :param num_reads: Number of output solutions to generate. Higher values provide better statistical coverage of the solution space but increase computational resources. Default is 100.
    :param num_retries: Number of attempts to retry embedding the problem onto the quantum hardware if initial attempts fail. Default is 0.
    :param max_iter: Maximum number of iterations for the solver. Each iteration involves running the three solvers in parallel, combining results, and refining the solution. Default is 100.
    :param max_time: Maximum time in seconds for the solver to run. Provides a hard time limit regardless of convergence or iteration status. Default is 5.
    :param convergence: Number of consecutive iterations without improvement before declaring convergence. Default is 3.
    :param target: Target objective value that triggers termination if reached. Default is None.
    :param rtol: Relative tolerance for convergence detection. Default is 1e-05.
    :param atol: Absolute tolerance for convergence detection. Default is 1e-08.
    :param simulated_annealing_params: Nested configuration for simulated annealing parameters used by the SA component.
    :param quantum_annealing_params: Nested configuration for quantum annealing parameters used by the QPU component.
    :param tabu_kerberos_params: Nested configuration for tabu search parameters used by the Tabu component.
    :param decomposer: Breaks down problems into subproblems of manageable size.
    """

    num_reads: int = 100
    num_retries: int = 0
    max_iter: Optional[int] = 100
    max_time: int = 5
    convergence: int = 3
    target: Optional[float] = None
    rtol: float = 1e-05
    atol: float = 1e-08
    backend = None
    simulated_annealing_params: SimulatedAnnealingBaseParams = field(
        default_factory=lambda: SimulatedAnnealingBaseParams(
            num_reads=None,
            num_sweeps=1000,
            beta_range=None,
            beta_schedule_type='geometric',
            initial_states_generator='random'
        )
    )
    quantum_annealing_params: QuantumAnnealingParams = field(
        default_factory=lambda: QuantumAnnealingParams(
            anneal_offsets=None,
            anneal_schedule=None,
            annealing_time=None,
            auto_scale=None,
            fast_anneal=False,
            flux_biases=None,
            flux_drift_compensation=True,
            h_gain_schedule=None,
            initial_state=None,
            max_answers=None,
            num_reads=1,
            programming_thermalization=None,
            readout_thermalization=None,
            reduce_intersample_correlation=False,
            reinitialize_state=None
        )
    )
    tabu_kerberos_params: TabuKerberosParams = field(
        default_factory=lambda: TabuKerberosParams(
            num_reads=None,
            tenure=None,
            timeout=100,
            initial_states_generator='random',
            max_time=None
        )
    )
    decomposer: Decomposer = field(
        default_factory=lambda: Decomposer(
            size=10,
            min_gain=None,
            rolling=True,
            rolling_history=1.0,
            silent_rewind=True,
            traversal='energy'
        )
    )

    @override
    def preprocess(self, data: Qubo) -> Result:

        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()
        
        model = get_model(data)

        backend = DWaveQpu(
            embedding_parameters=None,
            qpu_backend='default',
            token=get_dwave_token()
        )
        
        algorithm = Kerberos(
            backend=backend,
            num_reads=self.num_reads,
            num_retries=self.num_retries,
            max_iter=self.max_iter,
            max_time=self.max_time,
            convergence=self.convergence,
            target=self.target,
            rtol=self.rtol,
            atol=self.atol,
            simulated_annealing_params=self.simulated_annealing_params,
            quantum_annealing_params=self.quantum_annealing_params,
            tabu_kerberos_params=self.tabu_kerberos_params,
            decomposer=self.decomposer
        )

        job = algorithm.run(model)      

        solution = job.result(**scale_sleep(model))

        self.runtime = get_runtime(solution)
        self._result = get_best_solution(solution)

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result