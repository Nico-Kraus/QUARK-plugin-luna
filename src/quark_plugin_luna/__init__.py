from quark.plugin_manager import factory

from .luna_sa import LUNASA
from .luna_setpacking import LunaSetPacking
from .scip_solver import ScipSolver
from .lp_qubo_mapping import LunaLpQuboMapping
from .luna_flex_qaoa import LUNAFlexQaoa
from .luna_qa import LUNAQA
from .luna_parallel_tempering import LUNAPT
from .luna_qaoa import LUNAQAOA
from .luna_saga import LUNASAGA
from .luna_qbsolv import LUNAQBSolv
from .luna_rrsa import LUNARRSA
from .luna_pa import LUNAPopulationAnnealing


def register() -> None:

    # solvers
    factory.register("luna_sa", LUNASA)
    factory.register("luna_flex_qaoa", LUNAFlexQaoa)
    factory.register("luna_qa", LUNAQA)
    factory.register("luna_pt", LUNAPT)
    factory.register("luna_qaoa", LUNAQAOA)
    factory.register("luna_saga", LUNASAGA)
    factory.register("scip_solver", ScipSolver)
    factory.register("luna_qbsolv_sa", LUNAQBSolv)
    factory.register("luna_rrsa", LUNARRSA)
    factory.register("luna_pa", LUNAPopulationAnnealing)

    # mappings
    factory.register("luna_lp_qubo_mapping", LunaLpQuboMapping)

    # usecases
    factory.register("luna_setpacking", LunaSetPacking)
