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
    """
    A module for solving a QUBO problem using Repeated Reverse Simulated Annealing.

    Repeated Reverse Simulated Annealing finds the solution to a problem using an annealing process. 
    Initially, random states are chosen in the solution landscape. Afterwards, as the temperature 
    decreases, states are chosen that are more energetically favorable. This algorithm applies 
    principles similar to quantum reverse annealing but in a classical context. It starts from 
    specified states, partially "reverses" the annealing by increasing temperature to explore 
    nearby states, then re-anneals to find improved solutions. This process repeats for multiple 
    iterations, refining solutions progressively.

    For additional information refer to: https://docs.aqarios.com/algorithms/repeatedreversesimulatedannealing/

    :param num_reads_per_iter: Number of reads to perform in each iteration. Uses dynamic control of sampling intensity across iterations. Default is None.
    :param initial_states: Starting states for the first iteration. Each state defines values for all problem variables. Default is None.
    :param timeout: Maximum runtime in seconds before termination. Default is 5.0 seconds.
    :param max_iter: Maximum number of reverse annealing iterations to perform. Default is 10.
    :param target: Target energy value that triggers early termination if reached. Default is None.
    :param num_reads: Number of independent runs of the algorithm, each producing one solution sample. Default is None.
    :param num_sweeps: Number of iterations/sweeps per run, where each sweep updates all variables once. Default is 1000.
    :param beta_range: The inverse temperature schedule endpoints, specified as [start, end]. Default is None.
    :param beta_schedule_type: How beta values change between endpoints - "linear" or "geometric". Default is "geometric".
    :param initial_states_generator: How to handle cases with fewer initial states than num_reads - "none", "tile", or "random". Default is "random".
    :param num_sweeps_per_beta: Number of sweeps to perform at each temperature before cooling. Default is 1.
    :param seed: Random seed for reproducible results. Default is None.
    :param beta_schedule: Explicit sequence of beta values to use. Default is None.
    :param randomize_order: When True, variables are updated in random order during each sweep. Default is False.
    :param proposal_acceptance_criteria: Method for accepting or rejecting proposed moves - "Gibbs" or "Metropolis". Default is "Metropolis".
    """
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
