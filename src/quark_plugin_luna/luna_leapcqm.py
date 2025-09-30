from typing import override, Optional, List, Union
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other

from luna_quantum import LunaSolve
from luna_quantum.translator import LpTranslator
from luna_quantum.algorithms import LeapHybridCqm
from luna_quantum.backends import DWaveQpu

from .utils import scale_sleep, get_luna_api_key, get_dwave_token


@dataclass
class LUNALeapHybridCqm(Core):
    """
    A module for solving constrained optimization problems using D-Wave's Leap Hybrid Constrained Quadratic Model (CQM) solver.
    
    Leap's quantum-classical hybrid solvers are intended to solve arbitrary application problems formulated 
    as quadratic models. This solver accepts arbitrarily structured problems formulated as CQMs, with any 
    constraints represented natively. The Leap Hybrid CQM solver extends hybrid quantum-classical optimization 
    to handle constrained problems, allowing both linear and quadratic constraints alongside the quadratic 
    objective function. This enables solving many practical optimization problems in their natural formulation 
    without manual penalty conversion.
    
    The solver is suitable for mixed binary, integer, and continuous problems with thousands of variables and constraints.
    
    For additional information refer to: https://docs.aqarios.com/algorithms/leaphybridcqm/

    :param time_limit: Maximum running time in seconds. Longer limits generally yield better solutions but increase resource usage. Default is None, which uses the service's default time limit.
    :param spin_variables: Variables to represent as spins (-1/+1) rather than binary (0/1) values. Useful for problems naturally formulated in spin space. Default is None, which uses binary representation for all discrete variables.
    """

    time_limit: Optional[Union[float, int]] = None
    spin_variables: Optional[List[str]] = None
    backend = None

    @override
    def preprocess(self, data: Other) -> Result:

        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()

        model = LpTranslator.to_aq(data.data)
        backend = DWaveQpu(
            embedding_parameters=None,
            qpu_backend='default',
            token=get_dwave_token()
        )

        algorithm = LeapHybridCqm(
            backend=backend,
            time_limit=self.time_limit,
            spin_variables=self.spin_variables
        )

        job = algorithm.run(model)      

        solution = job.result(**scale_sleep(model))

        self.runtime = solution.runtime 
        self._result = dict(zip(solution.variable_names, solution.best().sample))

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result