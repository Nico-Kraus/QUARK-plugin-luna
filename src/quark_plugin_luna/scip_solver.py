from pyscipopt import Model
import io
from typing import override
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other



@dataclass
class ScipSolver(Core):
    
    @override
    def preprocess(self, data: Other) -> Result:
        
        lp_string = data.data.data
        model = Model()
        model.readProblem(io.StringIO(lp_string), "lp")

        model.optimize()

        self.status = model.getStatus()
        self.solution = model.getBestSol()
        self.runtime = model.getSolvingTime()

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime, "status": self.status}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self.solution))
        return result
