from typing import override, Optional, Literal
from dataclasses import dataclass, field

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import QAGA
from luna_quantum.solve.parameters.algorithms.base_params import QuantumAnnealingParams

from .utils import scale_sleep, converter_solution, converter_model, get_runtime, get_luna_api_key


@dataclass
class LUNAQAGA(Core):
    """
    Genetic Algorithm enhanced with quantum annealing (QAGA+) for QUBO problems.

    Maintains a population of candidate solutions, evolving them through
    mutation and recombination—leveraging quantum annealing in the mutation
    phase to better navigate solution space and escape local optima.

    Docs: https://docs.aqarios.com/algorithms/qaga/

    Upstream interface: quark.interface_types.qubo
    Downstream interface: None (wrapped as quark.interface_types.Other in postprocess)

    Parameters
    ----------
    p_size : int
        Initial population size. Default is 20.
    p_inc_num : int
        Number of new individuals added each generation. Default is 5.
    p_max : Optional[int]
        Maximum population size. Default is 160.
    pct_random_states : float
        Fraction of random individuals added each generation (e.g., 0.25 for 25%). Default is 0.25.
    mut_rate : float
        Mutation rate (0.0–1.0). Default is 0.5.
    rec_rate : int
        Recombination rate (# of mates per individual). Default is 1.
    rec_method : Literal["cluster_moves", "one_point_crossover", "random_crossover"]
        Recombination strategy. Default is "random_crossover".
    select_method : Literal["simple", "shared_energy"]
        Selection strategy. Default is "simple".
    target : Optional[float]
        Target energy level for early stopping. Default is None.
    atol : float
        Absolute tolerance for target energy check. Default is 0.0.
    rtol : float
        Relative tolerance for target energy check. Default is 0.0.
    timeout : float
        Total runtime limit in seconds. Default is 60.0 seconds.
    max_iter : Optional[int]
        Maximum number of generations. Default is 100.
    quantum_annealing_params : QuantumAnnealingParams
        Device-level annealing parameters to apply during quantum-assisted mutation.
    backend : Any
        Luna backend override; None lets Luna choose automatically (defaults to DWaveQpu).
    """

    p_size: int = 20
    p_inc_num: int = 5
    p_max: Optional[int] = 160
    pct_random_states: float = 0.25
    mut_rate: float = 0.5
    rec_rate: int = 1
    rec_method: Literal["cluster_moves", "one_point_crossover", "random_crossover"] = "random_crossover"
    select_method: Literal["simple", "shared_energy"] = "simple"
    target: Optional[float] = None
    atol: float = 0.0
    rtol: float = 0.0
    timeout: float = 60.0
    max_iter: Optional[int] = 100
    quantum_annealing_params: QuantumAnnealingParams = field(default_factory=QuantumAnnealingParams)

    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        Prepare the QUBO, configure QAGA with quantum-assisted mutation, run, and capture results.
        """
        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()

        bqm = converter_model(data._q)
        model = BqmTranslator.to_aq(bqm, name="bqm")

        algorithm = QAGA(
            backend=self.backend,
            p_size=self.p_size,
            p_inc_num=self.p_inc_num,
            p_max=self.p_max,
            pct_random_states=self.pct_random_states,
            mut_rate=self.mut_rate,
            rec_rate=self.rec_rate,
            rec_method=self.rec_method,
            select_method=self.select_method,
            target=self.target,
            atol=self.atol,
            rtol=self.rtol,
            timeout=self.timeout,
            max_iter=self.max_iter,
            quantum_annealing_params=self.quantum_annealing_params,
        )

        job = algorithm.run(model)
        solution = job.result(**scale_sleep(model))

        self.runtime = get_runtime(solution)
        self._result = converter_solution(solution)

        return Data(None)

    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        return Data(Other(self._result))
