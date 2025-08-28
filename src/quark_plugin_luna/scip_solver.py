from pyscipopt import Model
import os
from typing import override
from dataclasses import dataclass
import tempfile

from quark.core import Core, Data, Result
from quark.interface_types import Other



@dataclass
class ScipSolver(Core):
    
    @override
    def preprocess(self, data):
        lp_string = data.data

        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as tmpfile:
            tmpfile.write(lp_string)
            tmpfile.flush()
            tmpfile_name = tmpfile.name

        try:
            model = Model()
            model.setParam('display/verblevel', 0)
            model.readProblem(tmpfile_name)
            model.optimize()

            self.status = model.getStatus()
            self.solution = {var.name: model.getSolVal(model.getBestSol(), var) for var in model.getVars()}
            self.runtime = model.getSolvingTime()
        finally:
            os.remove(tmpfile_name)

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime, "status": self.status}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self.solution))
        return result
