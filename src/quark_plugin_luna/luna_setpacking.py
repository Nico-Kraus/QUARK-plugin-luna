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
        solution = data.data
        if solution is None:
            return Failed("No solution found")
        
        for row in self.subset_matrix:
            total = sum(sol * val for sol, val in zip(solution, row))
        if total > 1:
            return Failed("Invalid Solution")

    
        obj_val = sum(w * x for w, x in zip(self.subset_weights, solution))
        return Data(Other(obj_val))

    

