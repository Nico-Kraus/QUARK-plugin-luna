from typing import override, Optional, Literal, Tuple
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.algorithms import QBSolvLikeSimulatedAnnealing
from luna_quantum.solve.parameters.algorithms.base_params import SimulatedAnnealingBaseParams

from .utils import get_best_solution, get_model,  scale_sleep, get_runtime, get_luna_api_key


@dataclass
class LUNAQBSA(Core):
    """
    LUNAQBSA is a solver class implementing the QBSolv-like Simulated Annealing (QBSA) algorithm
    provided by Luna Quantum. This approach is designed for decomposing large Quadratic Unconstrained
    Binary Optimization (QUBO) problems into smaller subproblems that can be efficiently solved using
    a simulated annealing method. The solver supports rolling decomposition, convergence criteria, and
    flexible simulated annealing parameters.


    Parameters
    ----------
    decomposer_size : int, default=50
    The size of the subproblem into which the QUBO will be decomposed.
    rolling : bool, default=True
    Whether to use a rolling decomposition strategy.
    rolling_history : float, default=0.15
    Fraction of historical solutions to retain during rolling decomposition.
    max_iter : Optional[int], default=100
    Maximum number of iterations for the algorithm.
    max_time : int, default=5
    Maximum runtime (in seconds) for the solver.
    convergence : int, default=3
    Number of iterations without improvement before stopping early.
    target : Optional[float], default=None
    Target energy value to reach. The solver stops once the target is met.
    rtol : float, default=1e-5
    Relative tolerance used in convergence checks.
    atol : float, default=1e-8
    Absolute tolerance used in convergence checks.
    sa_num_reads : Optional[int], default=None
    Number of readouts (independent annealing runs).
    sa_num_sweeps : int, default=1000
    Number of sweeps (steps) per annealing run.
    sa_beta_range : Optional[Tuple[float, float]], default=None
    Range of inverse temperatures (beta) for simulated annealing.
    sa_beta_schedule_type : {"linear", "geometric"}, default="geometric"
    Type of beta schedule to use during annealing.
    sa_initial_states_generator : {"none", "tile", "random"}, default="random"
    Strategy for generating initial states in annealing runs.
    backend : optional
    Backend configuration for running the algorithm.


    Notes
    -----
    - The QBSolv-like simulated annealing algorithm decomposes large QUBOs into smaller subproblems
    and iteratively improves the global solution.
    - For details on the algorithm, see:
    https://docs.aqarios.com/algorithms/qbsolvlikesimulatedannealing/
"""


    decomposer_size: int = 50
    rolling: bool = True
    rolling_history: float = 0.15
    max_iter: Optional[int] = 100
    max_time: int = 5
    convergence: int = 3
    target: Optional[float] = None
    rtol: float = 1e-5
    atol: float = 1e-8
    sa_num_reads: Optional[int] = None
    sa_num_sweeps: int = 1000
    sa_beta_range: Optional[Tuple[float, float]] = None
    sa_beta_schedule_type: Literal["linear", "geometric"] = "geometric"
    sa_initial_states_generator: Literal["none", "tile", "random"] = "random"
    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()
        
        model = get_model(data)

        sa_params = SimulatedAnnealingBaseParams(
            num_reads=self.sa_num_reads,
            num_sweeps=self.sa_num_sweeps,
            beta_range=self.sa_beta_range,
            beta_schedule_type=self.sa_beta_schedule_type,
            initial_states_generator=self.sa_initial_states_generator,
        )

        algorithm = QBSolvLikeSimulatedAnnealing(
            backend=self.backend,
            decomposer_size=self.decomposer_size,
            rolling=self.rolling,
            rolling_history=self.rolling_history,
            max_iter=self.max_iter,
            max_time=self.max_time,
            convergence=self.convergence,
            target=self.target,
            rtol=self.rtol,
            atol=self.atol,
            simulated_annealing=sa_params,
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
        return Data(Other(self._result))
