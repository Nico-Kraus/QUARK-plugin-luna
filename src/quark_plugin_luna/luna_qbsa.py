from typing import override, Optional, Literal, Tuple
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import QBSolvLikeSimulatedAnnealing
from luna_quantum.solve.parameters.algorithms.base_params import SimulatedAnnealingBaseParams

from .utils import converter_solution, converter_model, get_runtime, get_luna_api_key


@dataclass
class LUNAQBSA(Core):
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
        bqm = converter_model(data._q)
        model = BqmTranslator.to_aq(bqm, name="bqm")

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
        solution = job.result()

        self.runtime = get_runtime(solution)
        self._result = converter_solution(solution)

        return Data(None)

    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        return Data(Other(self._result))
