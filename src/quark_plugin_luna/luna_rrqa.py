from typing import override, Optional, Any, List, Dict
from dataclasses import dataclass, field

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import RepeatedReverseQuantumAnnealing

from .utils import scale_sleep, converter_solution, converter_model, get_runtime, get_luna_api_key


@dataclass
class LUNARepeatedReverseQuantumAnnealing(Core):
    """
    A module for solving a QUBO problem using Repeated Reverse Quantum Annealing.
    
    Repeated Reverse Quantum Annealing begins the annealing process from a previously initialized state 
    and increases the temperature from there. Afterwards, the temperature is decreased again until the 
    solution is found. This procedure is repeated several times with this particular solver. This approach 
    combines reverse annealing (starting from a classical state) with repetition to refine solutions iteratively.
    
    For additional information refer to: https://docs.aqarios.com/algorithms/repeatedreversequantumannealing/

    :param anneal_offsets: Per-qubit time offsets for the annealing path, allowing qubits to anneal at different rates. Default is None.
    :param annealing_time: Duration of the annealing process in microseconds. Longer times can improve solution quality. Default is None.
    :param auto_scale: Whether to automatically normalize the problem energy range to match hardware capabilities. Default is None.
    :param flux_biases: Custom flux bias offsets for each qubit to compensate for manufacturing variations. Default is None.
    :param flux_drift_compensation: Whether to compensate for drift in qubit flux over time. Default is True.
    :param h_gain_schedule: Schedule for h-gain (linear coefficient strength) during annealing. Default is None.
    :param max_answers: Maximum number of unique answer states to return from the quantum hardware. Default is None.
    :param programming_thermalization: Wait time after programming the QPU before starting annealing. Default is None.
    :param readout_thermalization: Wait time after each anneal before reading results. Default is None.
    :param reduce_intersample_correlation: Whether to add delay between samples to reduce temporal correlations. Default is False.
    :param initial_states: Initial classical states to start the reverse annealing from. Default is None.
    :param n_initial_states: Number of initial states to create when initial_states is None. Default is 1.
    :param samples_per_state: How many samples to create per state in each iteration after the first. Default is 1.
    :param beta_schedule: Beta schedule controlling the quantum fluctuation strength during reverse annealing. Default is [0.5, 3].
    :param timeout: Maximum runtime in seconds before the solver stops. Default is 300 seconds.
    :param max_iter: Maximum number of iterations (reverse annealing cycles) to perform. Default is 10.
    :param target: Target energy value that triggers early termination if reached. Default is None.
    :param check_trivial: Whether to check for and handle trivial variables before sending to QPU. Default is True.
    """

    anneal_offsets: Optional[Any] = None
    annealing_time: Optional[Any] = None
    auto_scale: Optional[Any] = None
    flux_biases: Optional[Any] = None
    flux_drift_compensation: bool = True
    h_gain_schedule: Optional[Any] = None
    max_answers: Optional[int] = None
    programming_thermalization: Optional[float] = None
    readout_thermalization: Optional[float] = None
    reduce_intersample_correlation: bool = False
    initial_states: Optional[List[Dict[str, int]]] = None
    n_initial_states: int = 1
    samples_per_state: int = 1
    beta_schedule: List[float] = field(default_factory=lambda: [0.5, 3.0])
    timeout: float = 300.0
    max_iter: int = 10
    target: Optional[Any] = None
    check_trivial: bool = True
    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        This method preprocesses the input data (QUBO) for the LUNA Repeated Reverse Quantum Annealing module.
        """

        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()
        # bqm = converter_model(data._q)
        bqm = converter_model(data.as_dict())
        model = BqmTranslator.to_aq(bqm, name="bqm")
        
        algorithm = RepeatedReverseQuantumAnnealing(
            backend=self.backend,
            anneal_offsets=self.anneal_offsets,
            annealing_time=self.annealing_time,
            auto_scale=self.auto_scale,
            flux_biases=self.flux_biases,
            flux_drift_compensation=self.flux_drift_compensation,
            h_gain_schedule=self.h_gain_schedule,
            max_answers=self.max_answers,
            programming_thermalization=self.programming_thermalization,
            readout_thermalization=self.readout_thermalization,
            reduce_intersample_correlation=self.reduce_intersample_correlation,
            initial_states=self.initial_states,
            n_initial_states=self.n_initial_states,
            samples_per_state=self.samples_per_state,
            beta_schedule=self.beta_schedule,
            timeout=self.timeout,
            max_iter=self.max_iter,
            target=self.target,
            check_trivial=self.check_trivial
        )

        job = algorithm.run(model)      

        solution = job.result(**scale_sleep(model))

        # self.runtime = get_runtime(solution)
        self.runtime = solution.runtime.total_seconds
        self._result = converter_solution(solution)

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result