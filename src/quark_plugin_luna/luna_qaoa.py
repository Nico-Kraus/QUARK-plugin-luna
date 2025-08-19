from typing import override
from dataclasses import dataclass

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import BqmTranslator
from luna_quantum.algorithms import QAOA
from luna_quantum.solve.parameters.algorithms.base_params import (
    LinearQAOAParams,
    ScipyOptimizerParams
)

from .utils import converter_solution, converter_model, get_runtime, get_luna_api_key


@dataclass
class LUNAQA0A(Core):
    """
    A module for solving a QUBO problem using the Quantum Approximate Optimization Algorithm (QAOA).
    
    The Quantum Approximate Optimization Algorithm (QAOA) is one of the most promising combinatorial 
    optimization algorithms for gate-based quantum computers. The input problem needs to be unconstrained 
    and binary, typically even quadratic in interactions (QUBO). Such a problem can be represented as an 
    Ising cost Hamiltonian, which can directly be implemented on gate-based quantum hardware. Starting 
    with an equal superposition, the QAOA works through the alternating application of the cost Hamiltonian 
    and the mixer operator, essentially boosting the probability of measuring the sought-after solution 
    at the end of the algorithm.
    
    For additional information refer to: https://docs.aqarios.com/algorithms/qaoa/

    :param reps: Number of QAOA layers (p). Each layer consists of applying both the cost and mixing Hamiltonians with different variational parameters. Higher values generally lead to better solutions but increase circuit depth. Default is 1.
    :param shots: Number of measurement samples to collect per circuit execution. Higher values reduce statistical noise but increase runtime. Default is 1024.
    :param optimizer: Configuration for the classical optimization routine that updates the variational parameters. Default is ScipyOptimizer with default settings.
    :param initial_params: Custom QAOA variational circuit parameters. By default linear increasing/decreasing parameters for the selected reps are generated.
    """

    reps: int = 1
    shots: int = 1024
    backend = None
    optimizer: ScipyOptimizerParams = ScipyOptimizerParams(
        method='cobyla',
        tol=None,
        bounds=None,
        jac=None,
        hess=None,
        maxiter=100,
        options={}
    )
    initial_params: LinearQAOAParams = LinearQAOAParams(
        delta_beta=0.5,
        delta_gamma=0.5
    )

    @override
    def preprocess(self, data: Qubo) -> Result:
        """
        This method preprocesses the input data (QUBO) for the LUNA QAOA module.
        """

        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()
        bqm = converter_model(data._q)
        model = BqmTranslator.to_aq(bqm, name="bqm")
        
        algorithm = QAOA(
            backend=self.backend,
            reps=self.reps,
            shots=self.shots,
            optimizer=self.optimizer,
            initial_params=self.initial_params
        )

        job = algorithm.run(model)      

        solution = job.result()

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