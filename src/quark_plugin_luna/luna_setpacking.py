from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Failed, Result
from quark.interface_types import InterfaceType, Other

from luna_quantum.solve.use_cases import SetPacking
from luna_quantum import Model, LunaSolve
from luna_quantum.translator import LpTranslator

@dataclass
class LunaSetPacking(Core):
    

    subset_matrix = [[1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1]]

    subset_weights = [1, 1, 1, 1, 1, 1, 1]

    @override
    def preprocess(self, data: InterfaceType) -> Result:
        ls = LunaSolve()

        set_packing = SetPacking(subset_matrix=self.subset_matrix, weights=self.subset_weights)
        meta_model = ls.model.create_from_use_case(name="Set Packing", use_case=set_packing)
        model = Model.load_luna(model_id=meta_model.id)
        lp_model = LpTranslator.from_aq(model)
        return Data(Other[str](lp_model))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
            
        lp_solution = data.data.data
        if lp_solution is None:
            return Failed("No solution found")
        
        solution = []
        for i in range(len(self.subset_weights)):
            val = lp_solution.get(f"x_{i}", 0.0)
            solution.append(1 if val >= 0.5 else 0) 

        for row in self.subset_matrix:
            total = sum(x * val for x, val in zip(solution, row))
            if total > 1:
                return Failed("Invalid solution.")
            
        obj_value = sum(w * x for w, x in zip(self.subset_weights, solution))
        return Data(Other(obj_value))

    

