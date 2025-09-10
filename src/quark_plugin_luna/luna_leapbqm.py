from typing import override, Optional
from dataclasses import dataclass, field

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.algorithms import LeapHybridBqm
from luna_quantum.solve.parameters.algorithms.base_params import QuantumAnnealingParams
from luna_quantum.backends import DWaveQpu

from .utils import get_best_solution, get_model, scale_sleep, get_runtime, get_luna_api_key, get_dwave_token


@dataclass
class LUNALeapHybridBqm(Core):
    """
    Hybrid classical–quantum solver using D-Wave’s Leap Hybrid BQM service.

    This algorithm submits the full Binary Quadratic Model (BQM) directly
    to D-Wave's hybrid solver, which automatically handles problem decomposition,
    quantum annealing, and solution reconstruction in the cloud.
    
    For constrained problems, refer to the luna_leapcqm module.

    Docs: https://docs.aqarios.com/algorithms/leaphybridbqm/

    Upstream interface: quark.interface_types.qubo
    Downstream interface: None (wrapped as quark.interface_types.Other in postprocess)

    Parameters
    ----------
    time_limit : Optional[float]
        Maximum allowed runtime in seconds. `None` uses the backend’s default limit.
    quantum_annealing_params : QuantumAnnealingParams
        Optional device-level configuration (e.g., schedule, readout) for the QPU component.
    backend : Any
        Luna backend override; `None` lets Luna automatically select a default compatible backend.
    """

    time_limit: Optional[float] = None
    quantum_annealing_params: QuantumAnnealingParams = field(default_factory=QuantumAnnealingParams)

    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:

        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()

        
        model = get_model(data)
        backend = DWaveQpu(
            embedding_parameters=None,
            qpu_backend='default',
            token=get_dwave_token()
        )

        algorithm = LeapHybridBqm(
            backend=backend,
            time_limit=self.time_limit,
            quantum_annealing_params=self.quantum_annealing_params,
        )

        job = algorithm.run(model)
        solution = job.result(**scale_sleep(model))

        self.runtime = get_runtime(solution)
        self._result = get_best_solution(solution)

        return Data(None)

    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        return Data(Other(self._result))
