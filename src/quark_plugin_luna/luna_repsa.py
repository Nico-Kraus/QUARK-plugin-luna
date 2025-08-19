from typing import override, Optional, Literal, List
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import RepeatedReverseSimulatedAnnealing
from luna_quantum.solve.parameters.algorithms.base_params import SimulatedAnnealingBaseParams

from .utils import converter_solution, converter_model, get_runtime, get_luna_api_key


@dataclass
class LUNARRSA(Core):
    num_reads: Optional[int] = None
    num_sweeps: int = 1000
    beta_range: Optional[tuple[float, float]] = None
    beta_schedule_type: Literal["linear", "geometric"] = "geometric"
    initial_states_generator: Literal["none", "tile", "random"] = "random"
    num_sweeps_per_beta: int = 1
    seed: Optional[int] = None
    randomize_order: bool = False
    proposal_acceptance_criteria: Literal["Gibbs", "Metropolis"] = "Metropolis"

    num_reads_per_iter: Optional[List[int]] = None
    initial_states: Optional[List[dict]] = None  # list of dicts mapping var names to 0/1
    timeout: float = 5.0
    max_iter: int = 10
    target: Optional[float] = None

    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        LunaSolve.authenticate(get_luna_api_key()) 
        _ = LunaSolve()
        bqm = converter_model(data._q)
        model = BqmTranslator.to_aq(bqm, name="bqm")

        sa_params = SimulatedAnnealingBaseParams(
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            beta_range=self.beta_range,
            beta_schedule_type=self.beta_schedule_type,
            initial_states_generator=self.initial_states_generator,
            num_sweeps_per_beta=self.num_sweeps_per_beta,
            seed=self.seed,
            randomize_order=self.randomize_order,
            proposal_acceptance_criteria=self.proposal_acceptance_criteria
        )

        algorithm = RepeatedReverseSimulatedAnnealing(
            backend=self.backend,
            num_reads_per_iter=self.num_reads_per_iter,
            initial_states=self.initial_states,
            timeout=self.timeout,
            max_iter=self.max_iter,
            target=self.target,
            simulated_annealing=sa_params
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
