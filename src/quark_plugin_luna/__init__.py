from quark.plugin_manager import factory

from .luna_sa import LUNASA
from .luna_setpacking import LunaSetPacking
from .scip_solver import ScipSolver

def register() -> None:

    factory.register("luna_sa", LUNASA)
    factory.register("luna_setpacking", LunaSetPacking)
    factory.register("scip_solver", ScipSolver)
