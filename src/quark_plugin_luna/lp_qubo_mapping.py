# import logging #TODO: add logging
from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum.translator import LpTranslator
from luna_quantum.translator import CqmTranslator

import dimod


@dataclass
class LunaLpQuboMapping(Core):
    """A module for mapping an LP from Luna Usecases to a QUBO."""


    @override
    def preprocess(self, data: Other) -> Result:
        luna_model = LpTranslator.to_aq(data.data)
        self.cqm = CqmTranslator.from_aq(luna_model)
        self.bqm, self.inverter = dimod.cqm_to_bqm(self.cqm)
        q, _ = self.bqm.to_qubo() # _ is offset with a value of 0, so it can be omitted
        return Data(Qubo.from_dict(q))

    @override
    def postprocess(self, data: Other) -> Result:
        qubo_solution = data.data
        bqm_sample = dict(qubo_solution)
        lp_solution = dict(self.inverter(bqm_sample))

        return Data(Other(lp_solution))



