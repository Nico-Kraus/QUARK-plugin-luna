import logging
from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Failed, Result
from quark.interface_types import InterfaceType, Other

from luna_quantum.solve.use_cases import SetPacking
from luna_quantum import Model, LunaSolve, Solution
from luna_quantum.translator import LpTranslator

import random

from .utils import get_luna_api_key

@dataclass
class LunaSetPacking(Core):
    """A module for creating a Set Packing instance from LUNA.
    Upstream: Set Picking instance as LP string
    Downstream: Objective value of the solution

    :param set_size: The number of subsets to generate (default: 5)
    :param universe_size: The size of the universe (number of elements) (default: 7)
    :param density: The probability of each element being included in a subset (default: 0.4)
    :param weights: Weight assignment strategy - "equal" for uniform weights or "random" for random weights (default: "equal")
    :param seed: The seed for the random number generator for reproducible results (default: 123)
    """    

    set_size : int = 5
    universe_size : int = 7
    density : float = 0.4
    weights : str = "equal"
    seed : int = 123

    def generate_set_picking(self, set_size, universe_size, density, weights, seed):

        if seed is not None:
            random.seed(seed)
        subset_matrix = []
        for _ in range(set_size):
            row = [1 if random.random() < density else 0 for _ in range(universe_size)]
            # ensure at least one element per subset
            if sum(row) == 0:
                row[random.randint(0, universe_size-1)] = 1
            subset_matrix.append(row)

        if weights == "equal":
            subset_weights = [1] * set_size
        elif weights == "random":
            subset_weights = [random.randint(1, 10) for _ in range(set_size)]
        else:
            raise ValueError(f"weights must be 'equal' or 'random', and not {weights}")

        self.subset_matrix = subset_matrix
        self.subset_weights = subset_weights
        logging.info(f"Generated subset matrix: {self.subset_matrix}")

    @override
    def preprocess(self, data: InterfaceType = None) -> Result:
        self.generate_set_picking(self.set_size, self.universe_size, self.density, self.weights, self.seed)
        LunaSolve.authenticate(get_luna_api_key())
        ls = LunaSolve()

        set_packing = SetPacking(subset_matrix=self.subset_matrix, weights=self.subset_weights)
        meta_model = ls.model.create_from_use_case(name="Set Packing", use_case=set_packing)
        self.model = Model.load_luna(model_id=meta_model.id)

        lp_model = LpTranslator.from_aq(self.model)
        return Data(Other[str](lp_model))

    @override
    def postprocess(self, data: InterfaceType) -> Result:        
        lp_solution = data.data
        if lp_solution is None:
            return Failed("No solution found")
        solution = Solution.from_dict(model=self.model, data=data.data)
        logging.info(f"Best solution found: {solution.best()}")
        if not solution.best().feasible:
            return Failed("Invalid solution.")
        obj_value = solution.best().obj_value
        return Data(Other(obj_value))
