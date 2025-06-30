from quark.plugin_manager import factory

from .luna_sa import LUNASA

def register() -> None:

    factory.register("luna_sa", LUNASA)
