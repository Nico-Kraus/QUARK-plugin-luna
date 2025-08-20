from quark.plugin_manager import factory

from .luna_sa import LUNASA
from .luna_setpacking import LunaSetPacking
from .scip_solver import ScipSolver
from .lp_qubo_mapping import LunaLpQuboMapping
from .luna_flex_qaoa import LUNAFlexQaoa
from .luna_qa import LUNAQA
from .luna_pt import LUNAPT
from .luna_qaoa import LUNAQAOA
from .luna_saga import LUNASAGA
from .luna_qbsa import LUNAQBSA
from .luna_rrsa import LUNARRSA
from .luna_pa import LUNAPopulationAnnealing
from .luna_kerb import LUNAKerberos
from .luna_leapcqm import LUNALeapHybridCqm
from .luna_leapbqm import LUNALeapHybridBqm
from .luna_qaga import LUNAQAGA
from .luna_qbqa import LUNAQBSolvLikeQpu
from .luna_qpa import LUNAPopulationAnnealingQpu
from .luna_rrqa import LUNARepeatedReverseQuantumAnnealing


def register() -> None:

    # solvers
    factory.register("luna_sa", LUNASA)
    factory.register("luna_flex_qaoa", LUNAFlexQaoa)
    factory.register("luna_qa", LUNAQA)
    factory.register("luna_pt", LUNAPT)
    factory.register("luna_qaoa", LUNAQAOA)
    factory.register("luna_saga", LUNASAGA)
    factory.register("scip_solver", ScipSolver)
    factory.register("luna_qbsa", LUNAQBSA)
    factory.register("luna_rrsa", LUNARRSA)
    factory.register("luna_pa", LUNAPopulationAnnealing)
    factory.register("luna_kerb", LUNAKerberos)
    factory.register("luna_leapcqm", LUNALeapHybridCqm)
    factory.register("luna_leapbqm", LUNALeapHybridBqm)
    factory.register("luna_qaga", LUNAQAGA)
    factory.register("luna_qbqa", LUNAQBSolvLikeQpu)
    factory.register("luna_qpa", LUNAPopulationAnnealingQpu)
    factory.register("luna_rrqa", LUNARepeatedReverseQuantumAnnealing)

    # mappings
    factory.register("luna_lp_qubo_mapping", LunaLpQuboMapping)

    # usecases
    factory.register("luna_setpacking", LunaSetPacking)
