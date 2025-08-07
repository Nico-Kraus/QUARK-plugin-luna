import numpy as np

from typing import override
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.solve.parameters.algorithms import SimulatedAnnealing

from .utils import converter_solution, converter_model


@dataclass
class LUNASA(Core):
    """
    A module for solving a qubo problem using simulated annealing.

    :param num_reads: Number of independent runs; increases chance to find global optimum.
    :param backend: Backend to use for computation.
    :param num_sweeps: Number of variable sweeps per run (default: 1000).
    :param beta_range: [start, end] range for inverse temperature β (default: auto).
    :param beta_schedule_type: "linear" or "geometric" schedule for β (default: "geometric").
    :param initial_states_generator: How to fill missing initial states ("none", "tile", "random").
    :param num_sweeps_per_beta: Sweeps per temperature step (default: 1).
    :param seed: Random seed for reproducibility (default: None).
    :param beta_schedule: Explicit sequence of β values (overrides beta_range/type).
    :param initial_states: One or more predefined starting states (default: None).
    :param randomize_order: Whether to randomize variable update order (default: False).
    :param proposal_acceptance_criteria: "Gibbs" or "Metropolis" rule for accepting moves (default: "Metropolis").
    """

    num_reads: int = 100
    backend=None
    num_sweeps=1000
    beta_range=None
    beta_schedule_type='geometric'
    initial_states_generator='random'
    num_sweeps_per_beta=1
    seed=None
    beta_schedule=None
    initial_states=None
    randomize_order=False
    proposal_acceptance_criteria='Metropolis'
    

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        This method preprocesses the input data (QUBO) for the LUNA simulated annealing module.
        """

        LunaSolve.authenticate("")
        ls = LunaSolve()

        bqm = converter_model(data._q)

        model = BqmTranslator.to_aq(bqm, name="bqm")
        algorithm = SimulatedAnnealing(
            backend=self.backend,
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            beta_range=self.beta_range,
            beta_schedule_type=self.beta_schedule_type,
            initial_states_generator=self.initial_states_generator,
            num_sweeps_per_beta=self.num_sweeps_per_beta,
            seed=self.seed,
            beta_schedule=self.beta_schedule,
            initial_states=self.initial_states,
            randomize_order=self.randomize_order,
            proposal_acceptance_criteria=self.proposal_acceptance_criteria
        )

        job = algorithm.run(model)      

        solution = job.result()
        best_solution = converter_solution(solution)

        self.runtime = (solution.runtime.end - solution.runtime.start).total_seconds()
        self._result = best_solution

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result
