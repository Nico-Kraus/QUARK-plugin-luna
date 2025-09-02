import dimod

import numpy as np

from typing import override
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import QuantumAnnealing

from .utils import scale_sleep, converter_solution, converter_model, get_runtime, get_luna_api_key

@dataclass
class LUNAQA(Core):
    """
    A module for solving a QUBO problem using quantum annealing.

    :param anneal_offsets: Per-qubit time offsets in normalized units.
    :param anneal_schedule: Custom annealing schedule as (time, s) pairs.
    :param annealing_time: Duration of annealing process in microseconds.
    :param auto_scale: Whether to normalize energy range to hardware limits.
    :param fast_anneal: Use accelerated annealing protocol.
    :param flux_biases: Flux bias offsets per qubit in Φ₀.
    :param flux_drift_compensation: Enable compensation for flux drift.
    :param h_gain_schedule: Schedule for h-gain during annealing.
    :param initial_state: Starting state as list of {-1, +1}.
    :param max_answers: Max number of unique answer states to return.
    :param num_reads: Number of annealing cycles to perform.
    :param programming_thermalization: Wait time after programming QPU (μs).
    :param readout_thermalization: Wait time after anneal before reading (μs).
    :param reduce_intersample_correlation: Add delay between samples.
    :param reinitialize_state: Reset to new initial state between reads.
    """

    anneal_offsets = None
    anneal_schedule = None
    annealing_time = None
    auto_scale = None
    fast_anneal = False
    flux_biases = None
    flux_drift_compensation = True
    h_gain_schedule = None
    initial_state = None
    max_answers = None
    num_reads = 1
    programming_thermalization = None
    readout_thermalization = None
    reduce_intersample_correlation = False
    reinitialize_state = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        Preprocesses QUBO data for the LUNA quantum annealing module.
        """

        LunaSolve.authenticate(get_luna_api_key())
        ls = LunaSolve()

        # bqm = converter_model(data._q)
        bqm = converter_model(data.as_dict())
        model = BqmTranslator.to_aq(bqm, name="bqm")

        algorithm = QuantumAnnealing(
            backend=None,
            anneal_offsets=self.anneal_offsets,
            anneal_schedule=self.anneal_schedule,
            annealing_time=self.annealing_time,
            auto_scale=self.auto_scale,
            fast_anneal=self.fast_anneal,
            flux_biases=self.flux_biases,
            flux_drift_compensation=self.flux_drift_compensation,
            h_gain_schedule=self.h_gain_schedule,
            initial_state=self.initial_state,
            max_answers=self.max_answers,
            num_reads=self.num_reads,
            programming_thermalization=self.programming_thermalization,
            readout_thermalization=self.readout_thermalization,
            reduce_intersample_correlation=self.reduce_intersample_correlation,
            reinitialize_state=self.reinitialize_state
        )

        job = algorithm.run(model)      

        solution = job.result(**scale_sleep(model))
        best_solution = converter_solution(solution)

        # self.runtime = get_runtime(solution)
        self.runtime = solution.runtime.total_seconds
        self._result = best_solution

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result
