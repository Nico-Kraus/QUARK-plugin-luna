import dimod

import numpy as np

from typing import override
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import ParallelTempering


from .utils import converter_solution, converter_model, get_runtime

@dataclass
class LUNAPT(Core):
    """
    A module for solving a QUBO problem using parallel tempering, using multiple optimization procedures per temperature. 
    During the cooling process, an exchange of replicas can take place between the parallel procedures, 
    thus enabling higher energy mountains to be overcome.

    :param n_replicas: Number of system replicas at different temperatures. More replicas improve temperature coverage but increase cost. Default is 2.
    :param random_swaps_factor: Factor controlling frequency of swap attempts between replicas. Higher values increase mixing but add overhead. Default is 1.
    :param max_iter: Maximum number of iterations to perform. Each iteration samples all temperatures and attempts exchanges. Default is 100.
    :param max_time: Maximum runtime in seconds. Provides hard time limit regardless of other stopping criteria. Default is 5.
    :param convergence: Number of consecutive iterations without improvement before stopping. Default is 3.
    :param target: Target objective value for early termination. Default is None.
    :param rtol: Relative tolerance for convergence detection when comparing objective values. Default is DEFAULT_RTOL.
    :param atol: Absolute tolerance for convergence detection when comparing objective values. Default is DEFAULT_ATOL.
    :param fixed_temp_sampler_num_sweeps: Number of Monte Carlo sweeps per temperature level. More sweeps improve sampling quality. Default is 10,000.
    :param fixed_temp_sampler_num_reads: Number of independent sampling runs per temperature level. Default is None.
    """

    n_replicas: int = 2
    random_swaps_factor: int = 1
    max_iter: int = 100
    max_time: int = 5
    convergence: int = 3
    target: float = None
    rtol: float = 1e-05
    atol: float = 1e-08
    fixed_temp_sampler_num_sweeps: int = 10000
    fixed_temp_sampler_num_reads: int = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        Preprocesses QUBO data for the LUNA quantum annealing module.
        """

        LunaSolve.authenticate("")
        ls = LunaSolve()

        bqm = converter_model(data._q)
        model = BqmTranslator.to_aq(bqm, name="bqm")

        algorithm = ParallelTempering(
            backend=None,
            n_replicas=self.n_replicas,
            random_swaps_factor=self.random_swaps_factor,
            max_iter=self.max_iter,
            max_time=self.max_time,
            convergence=self.convergence,
            target=self.target,
            rtol=self.rtol,
            atol=self.atol,
            fixed_temp_sampler_num_sweeps=self.fixed_temp_sampler_num_sweeps,
            fixed_temp_sampler_num_reads=self.fixed_temp_sampler_num_reads
        )

        job = algorithm.run(model)      

        solution = job.result()
        best_solution = converter_solution(solution)

        self.runtime = get_runtime(solution)
        self._result = best_solution

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result
