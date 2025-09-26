

from typing import override, Literal, Union
from dataclasses import dataclass

from quark.core import Core, Data, Result, Failed
from quark.interface_types import Other, Qubo

from luna_quantum import LunaSolve
from luna_quantum.translator import LpTranslator
from luna_quantum.algorithms import FujitsuDAv3c, FujitsuDAv4

from .utils import get_best_solution, scale_sleep, get_runtime, get_luna_api_key

@dataclass
class LUNADA(Core):
    """
    Interface to the Fujitsu Digital Annealer (versions v3c and v4) via the
    Luna Quantum API. The Digital Annealer is a quantum-inspired, CMOS-based
    architecture for large-scale combinatorial optimization. It mimics annealing
    dynamics through massively parallel updates and escape strategies, enabling
    the solution of fully connected QUBO problems with up to tens of thousands
    of variables.

    Parameters for the Fujitsu Digital Annealer (v3c/v4).

    Attributes
    ----------
    version: Literal["v3c", "v4"]
        Solver version. Default: v3c
    time_limit_sec: int | None
        Maximum running time of DA in seconds. Specifies the upper limit of running
        time of DA. Time_limit_sec should be selected according to problem hardness
        and size (number of bits). Min: 1, Max: 3600
    target_energy: int | None
        Threshold energy for fast exit. This may not work correctly if the specified
        value is larger than its max value or lower than its min value.
        Min: -99_999_999_999, Max: 99_999_999_999
    num_group: int
        Number of independent optimization processes. Increasing the number of
        independent optimization processes leads to better coverage of the search
        space. Note: Increasing this number requires to also increase time_limit_sec
        such that the search time for each process is sufficient.
        Default: 1, Min: 1, Max: 16
    num_solution: int
        Number of solutions maintained and updated by each optimization process.
        Default: 16, Min: 1, Max: 1024
    num_output_solution: int
        Maximal number of the best solutions returned by each optimization.
        Total number of results is ``num_solution`` * ``num_group``.
        Default: 5, Min: 1, Max: 1024
    gs_num_iteration_factor: int
        Maximal number of iterations in one epoch of the global search in each
        optimization is ``gs_num_iteration_factor`` * *number of bits*.
        Default: 5, Min: 0, Max: 100
    gs_num_iteration_cl: int
        Maximal number of iterations without improvement in one epoch of the global
        search in each optimization before terminating and continuing with the next
        epoch. For problems with very deep local minima having a very low value is
        helpful. Default: 800, Min: 0, Max: 1000000
    gs_ohs_xw1h_num_iteration_factor: int
        Maximal number of iterations in one epoch of the global search in each
        optimization is ``gs_ohs_xw1h_num_iteration_factor`` * *number of bits*.
        Only used when 1Hot search is defined. Default: 3, Min: 0, Max: 100
    gs_ohs_xw1h_num_iteration_cl: int
        Maximal number of iterations without improvement in one epoch of the global
        search in each optimization before terminating and continuing with the next
        epoch. For problems with very deep local minima having a very low value is
        helpful. Only used when 1Hot search is defined.
        Default: 100, Min: 0, Max: 1000000
    ohs_xw1h_internal_penalty: int | str
        Mode of 1hot penalty constraint generation.
        - 0: internal penalty generation off: 1hot constraint as part of penalty
             polynomial required
        - 1: internal penalty generation on: 1hot constraint not as part of penalty
             polynomial required
        If 1way 1hot constraint or a 2way 1hot constraint is specified,
        ``ohs_xw1h_internal_penalty`` = 1 is recommended.
        Default: 0, Min: 0, Max: 1
    gs_penalty_auto_mode: int
        Parameter to choose whether to automatically incrementally adapt
        ``gs_penalty_coef`` to the optimal value.
        - 0: Use ``gs_penalty_coef`` as the fixed factor to weight the penalty
             polynomial during optimization.
        - 1: Start with ``gs_penalty_coef`` as weight factor for penalty polynomial
             and automatically and incrementally increase this factor during
             optimization by multiplying ``gs_penalty_inc_rate`` / 100 repeatedly
             until ``gs_max_penalty_coef`` is reached or the penalty energy iszero.
        Default: 1, Min: 0, Max: 1
    gs_penalty_coef: int
        Factor to weight the penalty polynomial. If ``gs_penalty_auto_mode`` is 0,
        this value does not change. If ``gs_penalty_auto_mode`` is 1, this initial
        weight factor is repeatedly increased by ``gs_penalty_inc_rate`` until
        ``gs_max_penalty_coef`` is reached or the penalty energy is zero.
        Default: 1, Min: 1, Max: 9_223_372_036_854_775_807
    gs_penalty_inc_rate: int
        Only used if ``gs_penalty_auto_mode`` is 1. In this case, the initial weight
        factor ``gs_penalty_coef`` for the penalty polynomial is repeatedly
        increased by multiplying ``gs_penalty_inc_rate`` / 100 until
        ``gs_max_penalty_coef`` is reached or the penalty energy is zero.
        Default: 150, Min: 100, Max: 200
    gs_max_penalty_coef: int
        Maximal value for the penalty coefficient. If ``gs_penalty_auto_mode`` is 0,
        this is the maximal value for ``gs_penalty_coef``.
        If ``gs_penalty_auto_mode`` is 1, this is the maximal value to which
        ``gs_penalty_coef`` can be increased during the automatic adjustment.
        If ``gs_max_penalty_coef`` is set to 0, then the maximal penalty coefficient
        is 2^63 - 1.
        Default: 0, Min: 0, Max: 9_223_372_036_854_775_807
    scaling_action: Literal["NOTHING", "SCALING", "AUTO_SCALING"]
        Method for scaling ``qubo`` and determining temperatures:
        - "NOTHING": No action (use parameters exactly as specified)
        - "SCALING": ``scaling_factor`` is multiplied to ``qubo``,
          ``temperature_start``, ``temperature_end`` and ``offset_increase_rate``.
        - "AUTO_SCALING": A maximum scaling factor w.r.t. ``scaling_bit_precision``
          is multiplied to ``qubo``, ``temperature_start``, ``temperature_end`` and
          ``offset_increase_rate``.
    scaling_factor: int | float
        Multiplicative factor applied to model coefficients, temperatures, and other
        parameters: the ``scaling_factor`` for ``qubo``, ``temperature_start``,
        ``temperature_end`` and ``offset_increase_rate``.
        Higher values can improve numerical precision but may lead to overflow.
        Default is 1.0 (no scaling).
    scaling_bit_precision: int
        Maximum bit precision to use when scaling. Determines the maximum allowable
        coefficient magnitude. Default is 64, using full double precision.
    random_seed: Union[int, None]
        Seed for random number generation to ensure reproducible results.
        Must be between 0 and 9_999. Default is None (random seed).
    penalty_factor: float
        Penalty factor used to scale the equality constraint penalty function,
        default 1.0.
    inequality_factor: int
        Penalty factor used to scale the inequality constraints, default 1.
    remove_ohg_from_penalty: bool
        If equality constraints, identified to be One-Hot constraints are only
        considered within one-hot groups (`remove_ohg_from_penalty=True`),
        i.e., identified one-hot constraints are not added to the penalty function,
        default True.
    """

    # Fujitsu Digital Annealer parameters
    version: Literal["v3c", "v4"] = "v3c"
    time_limit_sec: int | None = None
    target_energy: int | None = None
    num_group: int = 1
    num_solution: int = 16
    num_output_solution: int = 5
    gs_num_iteration_factor: int = 5
    gs_num_iteration_cl: int = 800
    gs_ohs_xw1h_num_iteration_factor: int = 3
    gs_ohs_xw1h_num_iteration_cl: int = 100
    ohs_xw1h_internal_penalty: int | str = 0
    gs_penalty_auto_mode: int = 1
    gs_penalty_coef: int = 1
    gs_penalty_inc_rate: int = 150
    gs_max_penalty_coef: int = 0
    scaling_action: Literal["NOTHING", "SCALING", "AUTO_SCALING"] = "NOTHING"
    scaling_factor: int | float = 1.0
    scaling_bit_precision: int = 64
    random_seed: Union[int, None] = None
    penalty_factor: float = 1.0
    inequality_factor: int = 1
    remove_ohg_from_penalty: bool = True

    @override
    def preprocess(self, data: Qubo) -> Result:
        LunaSolve.authenticate(get_luna_api_key())
        _ = LunaSolve()

        
        model = LpTranslator.to_aq(data.data)

        params = {
            "backend": None,
            "time_limit_sec": self.time_limit_sec,
            "target_energy": self.target_energy,
            "num_group": self.num_group,
            "num_solution": self.num_solution,
            "num_output_solution": self.num_output_solution,
            "gs_num_iteration_factor": self.gs_num_iteration_factor,
            "gs_num_iteration_cl": self.gs_num_iteration_cl,
            "gs_ohs_xw1h_num_iteration_factor": self.gs_ohs_xw1h_num_iteration_factor,
            "gs_ohs_xw1h_num_iteration_cl": self.gs_ohs_xw1h_num_iteration_cl,
            "ohs_xw1h_internal_penalty": self.ohs_xw1h_internal_penalty,
            "gs_penalty_auto_mode": self.gs_penalty_auto_mode,
            "gs_penalty_coef": self.gs_penalty_coef,
            "gs_penalty_inc_rate": self.gs_penalty_inc_rate,
            "gs_max_penalty_coef": self.gs_max_penalty_coef,
            "scaling_action": self.scaling_action,
            "scaling_factor": self.scaling_factor,
            "scaling_bit_precision": self.scaling_bit_precision,
            "random_seed": self.random_seed,
            "penalty_factor": self.penalty_factor,
            "inequality_factor": self.inequality_factor,
            "remove_ohg_from_penalty": self.remove_ohg_from_penalty
        }
        if self.version == "v3c":
            algorithm = FujitsuDAv3c(**params)
        elif self.version == "v4":
            algorithm = FujitsuDAv4(**params)
        else:
            return Failed("Wrong version parameter.")

        job = algorithm.run(model)      

        solution = job.result(**scale_sleep(model))
        best_solution = get_best_solution(solution)
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
