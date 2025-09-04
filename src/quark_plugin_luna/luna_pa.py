from typing import override, Optional
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import PopulationAnnealing

from .utils import scale_sleep, converter_solution, converter_model, get_runtime, get_luna_api_key


@dataclass
class LUNAPopulationAnnealing(Core):
    """
    A module for solving a QUBO problem using Population Annealing.
    
    Population Annealing uses a sequential Monte Carlo method to minimize the energy of a population. 
    The population consists of walkers that can explore their neighborhood during the cooling process. 
    Afterwards, walkers are removed and duplicated using bias to lower energy. Eventually, a population 
    collapse occurs where all walkers are in the lowest energy state.
    
    For additional information refer to: https://docs.aqarios.com/algorithms/populationannealing/

    :param max_iter: Maximum number of annealing iterations (temperature steps) to perform. Each iteration involves lowering the temperature, allowing walkers to explore locally, and then resampling the population. Default is 20.
    :param max_time: Maximum time in seconds that the algorithm is allowed to run. Provides a hard time limit regardless of convergence or iteration status. Default is 2.
    :param fixed_temp_sampler_num_sweeps: Number of Monte Carlo sweeps to perform at each temperature level, where one sweep attempts to update all variables once. More sweeps allow better exploration of local configuration space. Default is 10000.
    :param fixed_temp_sampler_num_reads: Number of independent sampling runs to perform at each temperature level. Each run effectively initializes a separate walker in the population. Default is None.
    """

    max_iter: int = 20
    max_time: int = 2
    fixed_temp_sampler_num_sweeps: int = 10000
    fixed_temp_sampler_num_reads: Optional[int] = None
    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        This method preprocesses the input data (QUBO) for the LUNA Population Annealing module.
        """

        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()
        bqm = converter_model(data._q)
        model = BqmTranslator.to_aq(bqm, name="bqm")
        
        algorithm = PopulationAnnealing(
            backend=self.backend,
            max_iter=self.max_iter,
            max_time=self.max_time,
            fixed_temp_sampler_num_sweeps=self.fixed_temp_sampler_num_sweeps,
            fixed_temp_sampler_num_reads=self.fixed_temp_sampler_num_reads
        )

        job = algorithm.run(model)      

        solution = job.result(**scale_sleep(model))

        self.runtime = get_runtime(solution)
        self._result = converter_solution(solution)

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result