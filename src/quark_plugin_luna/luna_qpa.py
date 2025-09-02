from typing import override, Optional
from dataclasses import dataclass, field

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import PopulationAnnealingQpu
from luna_quantum.solve.parameters.algorithms.base_params import Decomposer, QuantumAnnealingParams

from .utils import scale_sleep, converter_solution, converter_model, get_runtime, get_luna_api_key


@dataclass
class LUNAPopulationAnnealingQpu(Core):
    """
    QPU-backed Population Annealing for QUBO problems.

    This algorithm performs population annealing while delegating subproblems to a quantum annealer
    (e.g., D-Wave QPU). A decomposer controls how the problem is split and stitched, and
    QuantumAnnealingParams exposes device-level knobs for the QPU run.

    Docs: https://docs.aqarios.com/algorithms/populationannealingqpu/

    Upstream interface: quark.interface_types.qubo
    Downstream interface: None (wrapped as quark.interface_types.Other in postprocess)

    Parameters
    ----------
    num_reads : int
        Number of reads per QPU submission.
    num_retries : int
        Number of retries for failed or unsatisfactory subproblem solves.
    max_iter : int
        Maximum number of annealing iterations.
    max_time : int
        Maximum wall-clock time in seconds for the algorithm.
    fixed_temp_sampler_num_sweeps : int
        Number of classical sweeps per fixed-temperature sampling stage.
    fixed_temp_sampler_num_reads : Optional[int]
        Number of classical reads per fixed-temperature sampling stage.
    decomposer_size : int
        Target size of subproblems produced by the decomposer.
    decomposer_min_gain : Optional[float]
        Minimum expected gain threshold for accepting a decomposition step.
    decomposer_rolling : bool
        Enables rolling window decomposition strategy.
    decomposer_rolling_history : float
        Fraction of history used for rolling decomposition.
    decomposer_silent_rewind : bool
        Enables silent rewind behavior in the decomposer.
    decomposer_traversal : str
        Traversal heuristic name used by the decomposer.
    quantum_annealing_params : QuantumAnnealingParams
        QPU-specific configuration such as anneal schedule and readout options.
    backend : Any
        Luna backend to execute on; None lets Luna select a default.
    """

    num_reads: int = 100
    num_retries: int = 0
    max_iter: int = 20
    max_time: int = 2
    fixed_temp_sampler_num_sweeps: int = 10000
    fixed_temp_sampler_num_reads: Optional[int] = None

    decomposer_size: int = 10
    decomposer_min_gain: Optional[float] = None
    decomposer_rolling: bool = True
    decomposer_rolling_history: float = 1.0
    decomposer_silent_rewind: bool = True
    decomposer_traversal: str = "energy"

    quantum_annealing_params: QuantumAnnealingParams = field(default_factory=QuantumAnnealingParams)

    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        Convert QUBO, configure QPU population annealing, run, and cache results.
        """
        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()

        # bqm = converter_model(data._q)
        bqm = converter_model(data.as_dict())
        model = BqmTranslator.to_aq(bqm, name="bqm")

        decomposer = Decomposer(
            size=self.decomposer_size,
            min_gain=self.decomposer_min_gain,
            rolling=self.decomposer_rolling,
            rolling_history=self.decomposer_rolling_history,
            silent_rewind=self.decomposer_silent_rewind,
            traversal=self.decomposer_traversal,
        )

        algorithm = PopulationAnnealingQpu(
            backend=self.backend,
            num_reads=self.num_reads,
            num_retries=self.num_retries,
            max_iter=self.max_iter,
            max_time=self.max_time,
            fixed_temp_sampler_num_sweeps=self.fixed_temp_sampler_num_sweeps,
            fixed_temp_sampler_num_reads=self.fixed_temp_sampler_num_reads,
            decomposer=decomposer,
            quantum_annealing_params=self.quantum_annealing_params,
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
        return Data(Other(self._result))
