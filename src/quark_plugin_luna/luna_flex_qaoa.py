from typing import override
from dataclasses import dataclass, field

from quark.core import Core, Data, Result
from quark.interface_types import Other

from luna_quantum import LunaSolve
from luna_quantum.translator import LpTranslator
from luna_quantum.algorithms import FlexQAOA
from luna_quantum.solve.parameters.algorithms.base_params import (
    LinearQAOAParams,
    ScipyOptimizerParams
)
from luna_quantum.solve.parameters.algorithms.quantum_gate.flex_qaoa import (
    AdvancedConfig,
    XYMixer,
    IndicatorFunctionParams,
    OneHotParams,
    PipelineParams,
    QuadraticPenaltyParams
)

from .utils import scale_sleep, get_luna_api_key

@dataclass
class LUNAFlexQaoa(Core):
    """
    The FlexQAOA is a variant of QAOA developed by Aqarios, specifically focused on problems coming from the real world.
    The core issue with plain QAOA is that it requires a Quadratic Unconstrained Binary Optimization (QUBO) input format. 
    However, real-world problems typically contain constraints, that can not be violated for a feasible solution. 
    While constraints can be reformulated into penalty terms to form a QUBO. These penalty terms distort the energy spectrum
    of the optimization heavily, leading to impaired optimization performance. Additionally, in inequality-constrained cases,
    slack variables have to be introduced, effectively enlarging the search space.
    
    For additional information refer to: https://docs.aqarios.com/algorithms/flexqaoa/

    :param shots: Number of sampled shots.
    :param reps: Number of QAOA layer repetitions.
    :param pipeline: Pipeline defining selected features for QAOA circuit generation. By default, all supported features are enabled (one-hot constraints, inequality constraints, quadratic penalties).
    :param optimizer: Classical optimizer for parameter tuning. Default is ScipyOptimizer. Setting to None disables optimization and evaluates initial parameters only.
    :param qaoa_config: Additional options for QAOA circuit and evaluation.
    :param initial_params: Custom QAOA variational circuit parameters. By default, linear increasing/decreasing parameters for the selected reps are generated.
    """

    shots: int = 1024
    reps: int = 1
    pipeline: PipelineParams = field(
        default_factory=lambda: PipelineParams(
            indicator_function=IndicatorFunctionParams(
                penalty=None,
                penalty_scaling=2
            ),
            one_hot=OneHotParams(),
            quadratic_penalty=QuadraticPenaltyParams(
                penalty=None
            )
        )
    )
    optimizer: ScipyOptimizerParams = field(
        default_factory=lambda: ScipyOptimizerParams(
            method='cobyla',
            tol=None,
            bounds=None,
            jac=None,
            hess=None,
            maxiter=100,
            options={}
        )
    )
    qaoa_config: AdvancedConfig = field(
        default_factory=lambda: AdvancedConfig(
            mixer=XYMixer(types=['even', 'odd', 'last']),
            parallel_indicators=True,
            discard_slack=False,
            infeas_penalty=None
        )
    )
    initial_params: LinearQAOAParams = field(
        default_factory=lambda: LinearQAOAParams(
            delta_beta=0.5,
            delta_gamma=0.5
        )
    )

    @override
    def preprocess(self, data: Other) -> Result:
        """
        Preprocesses QUBO data for the LUNA quantum annealing module.
        """

        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()

        model = LpTranslator.to_aq(data.data)

        algorithm = FlexQAOA(
            backend=None,
            shots=self.shots,
            reps=self.reps,
            pipeline=self.pipeline,
            optimizer=self.optimizer,
            qaoa_config=self.qaoa_config,
            initial_params=self.initial_params
        )

        job = algorithm.run(model)      

        solution = job.result(**scale_sleep(model))
        best_solution = solution.best()
        best_value = best_solution.obj_value

        self.runtime = solution.runtime 
        self._result = best_value

        return Data(None)
    
    @override
    def get_metrics(self):
        return {"runtime": self.runtime}

    @override
    def postprocess(self, data: Data = None) -> Result:
        result = Data(Other(self._result))
        return result