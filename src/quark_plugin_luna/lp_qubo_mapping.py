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
        cqm = CqmTranslator.from_aq(luna_model)
        bqm, _ = dimod.cqm_to_bqm(cqm) # _ is an inverter that is not needed here
        q, _ = bqm.to_qubo() # _ is offset with a value of 0, so it can be omitted
        formatted_qubo_dict = {}
        for (var1, var2), coeff in q.items():
            if var1 == var2:
                key = "q" + str(var1[2:])
            else:
                key = "q" + str(var1[2:]) + ",q" + str(var2[2:])
            formatted_qubo_dict[key] = coeff
        return Data(Qubo.from_dict(formatted_qubo_dict))

    @override
    def postprocess(self, data: Other) -> Result:
        qubo_solution = data.data
        reformatted_solution = {}
        for key, value in qubo_solution.items():
            reformatted_solution["x_" + key[1:]] = qubo_solution[key]
        return Data(Other(reformatted_solution))



