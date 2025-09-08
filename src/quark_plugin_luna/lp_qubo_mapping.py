# import logging #TODO: add logging
from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum.translator import LpTranslator
from luna_quantum.translator import CqmTranslator

import dimod

from .utils import convert_qubo

@dataclass
class LunaLpQuboMapping(Core):
    """A module for mapping an LP from Luna Usecases to a QUBO."""


    @override
    def preprocess(self, data: Other) -> Result:
        luna_model = LpTranslator.to_aq(data.data)
        self.cqm = CqmTranslator.from_aq(luna_model)
        self.bqm, self.inverter = dimod.cqm_to_bqm(self.cqm)
        qubo_dict = convert_qubo(self.bqm)
        
        return Data(Qubo.from_dict(qubo_dict))

    @override
    def postprocess(self, data: Other) -> Result:
       
        lp_solution = dict(self.inverter(data.data))

        return Data(Other(lp_solution))



