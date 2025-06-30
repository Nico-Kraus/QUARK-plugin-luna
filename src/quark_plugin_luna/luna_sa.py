import dimod

import numpy as np

from typing import override
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.solve.parameters.algorithms import SimulatedAnnealing

def converter_model(data):
    """
    Converts a QUBO problem represented as a dictionary into a Binary Quadratic Model (BQM).
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')

    for (index1, index2), value in data.items():
        var1 = f"v{index1[0]}_{index1[1]}"
        var2 = f"v{index2[0]}_{index2[1]}"
        
        if var1 == var2:
            bqm.add_variable(var1, value)
        else:
            bqm.add_interaction(var1, var2, value)

    return bqm

def converter_solution(best_sample, var_names):
    """
    Converts the best sample from the simulated annealing solution into a dictionary format.
    """
    solution_dict = {}

    for var, val in zip(var_names, best_sample):
        # Extract index: 'v2_3' -> (2, 3)
        i, j = map(int, var[1:].split('_'))
        solution_dict[(i, j)] = np.int8(val)

    return solution_dict


@dataclass
class LUNASA(Core):
    """
    A module for solving a qubo problem using simulated annealing.

    :param num_reads: The number of reads to perform
    """

    num_reads: int = 100

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        This method preprocesses the input data (QUBO) for the simulated annealing module.
        """

        LunaSolve.authenticate("")
        ls = LunaSolve()

        bqm = converter_model(data._q)

        model = BqmTranslator.to_aq(bqm, name="bqm")

        algorithm = SimulatedAnnealing(
            backend=None,
            num_reads=self.num_reads,
            num_sweeps=1000,
            beta_range=None,
            beta_schedule_type='geometric',
            initial_states_generator='random',
            num_sweeps_per_beta=1,
            seed=None,
            beta_schedule=None,
            initial_states=None,
            randomize_order=False,
            proposal_acceptance_criteria='Metropolis'
        )

        job = algorithm.run(model)

        solution = job.result()
        self.runtime = solution.runtime

        objective_values = solution.obj_values
        best_index = np.argmin(objective_values)

        best_sample = solution.samples.tolist()[best_index]
        var_names = solution.variable_names

        best_solution = converter_solution(best_sample, var_names)

        self._result = best_solution

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data) -> Result:
        result = Data(Other(self._result))
        return result
