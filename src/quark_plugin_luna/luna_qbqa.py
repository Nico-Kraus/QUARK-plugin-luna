from typing import override, Optional
from dataclasses import dataclass, field

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.algorithms import QBSolvLikeQpu
from luna_quantum.solve.parameters.algorithms.base_params import Decomposer, QuantumAnnealingParams
from luna_quantum.backends import DWaveQpu

from .utils import get_best_solution, get_model,  scale_sleep,   get_runtime, get_luna_api_key, get_dwave_token


@dataclass
class LUNAQBSolvLikeQpu(Core):
    """
    QPU-backed qbsolv-like decomposition for QUBO problems.

    This algorithm partitions a large QUBO into subproblems using a decomposer and
    solves each subproblem on a quantum annealer. Results are iteratively recombined
    with convergence checks and optional target energy.

    Docs: https://docs.aqarios.com/algorithms/qbsolvlikeqpu/

    Upstream interface: quark.interface_types.qubo
    Downstream interface: None (wrapped as quark.interface_types.Other in postprocess)

    Parameters
    ----------
    decomposer_size : int
        Target size for subproblems created by the decomposer.
    rolling : bool
        Enables rolling window updates in the decomposer.
    rolling_history : float
        Fraction of history used for rolling updates.
    max_iter : Optional[int]
        Maximum number of outer iterations; None means unlimited within time budget.
    max_time : int
        Maximum wall-clock time in seconds for the full routine.
    convergence : int
        Number of stagnant iterations before early stop.
    target : Optional[float]
        Energy target; stop when reached or surpassed.
    rtol : float
        Relative tolerance for energy improvement.
    atol : float
        Absolute tolerance for energy improvement.
    num_reads : int
        Number of reads per QPU submission.
    num_retries : int
        Number of retries for subproblems.
    quantum_annealing_params : QuantumAnnealingParams
        QPU-specific configuration (anneal schedule, readout, etc.).
    decomposer : Decomposer
        Decomposer configuration object used to split and merge subproblems.
    backend : Any
        Luna backend to execute on; None lets Luna select a default.
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
    num_reads: int = 100
    num_retries: int = 0
    quantum_annealing_params: QuantumAnnealingParams = field(default_factory=QuantumAnnealingParams)
    decomposer: Decomposer = field(default_factory=Decomposer)

    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        Convert QUBO, configure qbsolv-like QPU routine, run, and cache results.
        """
        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()

        
        model = get_model(data)
        backend = DWaveQpu(
            embedding_parameters=None,
            qpu_backend='default',
            token=get_dwave_token()
        )

        algorithm = QBSolvLikeQpu(
            backend=backend,
            decomposer_size=self.decomposer_size,
            rolling=self.rolling,
            rolling_history=self.rolling_history,
            max_iter=self.max_iter,
            max_time=self.max_time,
            convergence=self.convergence,
            target=self.target,
            rtol=self.rtol,
            atol=self.atol,
            num_reads=self.num_reads,
            num_retries=self.num_retries,
            quantum_annealing_params=self.quantum_annealing_params,
            decomposer=self.decomposer,
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
