from typing import override
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other

from luna_quantum import LunaSolve
from luna_quantum.translator import LpTranslator
from luna_quantum.algorithms import SCIP

from .utils import scale_sleep, get_luna_api_key, get_runtime


@dataclass
class LUNASCIP(Core):
    """
    A module for solving constrained optimization problems using the SCIP (Solve Constraint Integer Programming) solver.
    
    SCIP is a freely available solver for mixed integer programming (MIP) and mixed integer nonlinear programming 
    (MINLP) developed by the Zuse Institut Berlin. Currently it is amongst the fastest non-commercial solvers 
    available. SCIP can handle a wide range of optimization problems including linear programming, mixed integer 
    programming, and constraint satisfaction problems.
    
    Note: SCIP may introduce auxiliary variables during the solving process, e.g., the variable quadobjvar. 
    These variables will be stored in the solution metadata under the key auxiliary_variables.
    
    For additional information refer to: https://docs.aqarios.com/algorithms/scip/
    """

    backend = None

    @override
    def preprocess(self, data: Other) -> Result:
        """
        Preprocesses constrained optimization data for the LUNA SCIP module.
        """

        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()

        model = LpTranslator.to_aq(data.data)

        algorithm = SCIP(
            backend=self.backend
        )

        job = algorithm.run(model)      

        solution = job.result(**scale_sleep(model))
        best_solution = solution.best()
        
        self._result = dict(zip(solution.variable_names, best_solution.sample))
        self.runtime = get_runtime(solution)
        
        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result