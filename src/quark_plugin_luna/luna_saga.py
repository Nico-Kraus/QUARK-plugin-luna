from typing import override, Literal, Optional, Tuple
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import SAGA

from .utils import scale_sleep, converter_solution, converter_model, get_runtime, get_luna_api_key


@dataclass
class LUNASAGA(Core):
    """
    A module for solving a QUBO problem using the Simulated Annealing Genetic Algorithm (SAGA).
    
    The SAGA algorithm uses the paradigm of Genetic Algorithms: We keep track of a population of 
    possible solutions to an optimization problem in the QUBO formulation, and iteratively create 
    new solutions from these using mutations and recombinations. A selection ensures we only keep 
    track of the most promising solutions in the population for the next iteration. This process 
    is run until a predefined stopping criterion is reached. For SAGA, Simulated Annealing is 
    used during the mutation phase.
    
    For additional information refer to: https://docs.aqarios.com/algorithms/saga/

    :param p_size: Initial population size (number of candidate solutions). Default is 20.
    :param p_inc_num: Number of new individuals added to the population after each generation. Default is 5.
    :param p_max: Maximum population size. Once reached, no more growth occurs. Default is 160.
    :param pct_random_states: Percentage of random states added to the population after each iteration. Default is 0.25 (25%).
    :param mut_rate: Mutation rate - probability to mutate an individual after each iteration. Default is 0.5.
    :param rec_rate: Recombination rate - number of mates each individual is recombined with per generation. Default is 1.
    :param rec_method: Method used for recombining individuals. Default is "random_crossover".
    :param select_method: Selection strategy for the next generation. Default is "simple".
    :param target: Target energy level to stop the algorithm. Default is None.
    :param atol: Absolute tolerance when comparing energies to target. Default is 1e-08.
    :param rtol: Relative tolerance when comparing energies to target. Default is 1e-05.
    :param timeout: Maximum runtime in seconds. Default is 60.0 seconds.
    :param max_iter: Maximum number of generations before stopping. Default is 100.
    :param num_sweeps: Initial number of sweeps for simulated annealing in the first iteration. Default is 10.
    :param num_sweeps_inc_factor: Factor by which to increase num_sweeps after each iteration. Default is 1.2.
    :param num_sweeps_inc_max: Maximum number of sweeps that may be reached when increasing the num_sweeps. Default is 7000.
    :param beta_range_type: Method used to compute the temperature range (beta range) for annealing. Default is "default".
    :param beta_range: Explicit beta range (inverse temperature) used with beta_range_type "fixed" or "percent". Default is None.
    """

    p_size: int = 20
    p_inc_num: int = 5
    p_max: int = 160
    pct_random_states: float = 0.25
    mut_rate: float = 0.5
    rec_rate: int = 1
    rec_method: Literal['cluster_moves', 'one_point_crossover', 'random_crossover'] = 'random_crossover'
    select_method: Literal['simple', 'shared_energy'] = 'simple'
    target: Optional[float] = None
    atol: float = 1e-08
    rtol: float = 1e-05
    timeout: float = 60.0
    max_iter: int = 100
    num_sweeps: int = 10
    num_sweeps_inc_factor: float = 1.2
    num_sweeps_inc_max: int = 7000
    beta_range_type: Literal['default', 'percent', 'fixed', 'inc'] = 'default'
    beta_range: Optional[Tuple[float, float]] = None
    backend = None

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        This method preprocesses the input data (QUBO) for the LUNA SAGA module.
        """

        LunaSolve.authenticate(get_luna_api_key())
        ls = LunaSolve()
        bqm = converter_model(data._q)
        model = BqmTranslator.to_aq(bqm, name="bqm")
        
        algorithm = SAGA(
            backend=self.backend,
            p_size=self.p_size,
            p_inc_num=self.p_inc_num,
            p_max=self.p_max,
            pct_random_states=self.pct_random_states,
            mut_rate=self.mut_rate,
            rec_rate=self.rec_rate,
            rec_method=self.rec_method,
            select_method=self.select_method,
            target=self.target,
            atol=self.atol,
            rtol=self.rtol,
            timeout=self.timeout,
            max_iter=self.max_iter,
            num_sweeps=self.num_sweeps,
            num_sweeps_inc_factor=self.num_sweeps_inc_factor,
            num_sweeps_inc_max=self.num_sweeps_inc_max,
            beta_range_type=self.beta_range_type,
            beta_range=self.beta_range
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