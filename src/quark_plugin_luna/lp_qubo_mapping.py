# import logging #TODO: add logging
from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Result
from quark.interface_types import Other, Qubo

from luna_quantum.translator import LpTranslator
from luna_quantum.translator import CqmTranslator

import dimod

from .utils import build_varmap

@dataclass
class LunaLpQuboMapping(Core):
    """A module for mapping an LP from Luna Usecases to a QUBO."""


    @override
    def preprocess(self, data: Other) -> Result:
        luna_model = LpTranslator.to_aq(data.data)
        self.cqm = CqmTranslator.from_aq(luna_model)
        self.bqm, self.inverter = dimod.cqm_to_bqm(self.cqm)
        q, _ = self.bqm.to_qubo()

        self.varmap, self.inv_varmap = build_varmap(q)
        qubo_dict = {}
        for (i, j), value in q.items():
            i_new, j_new = self.varmap[i], self.varmap[j]
            if i_new == j_new:
                qubo_dict[i_new] = float(value)
            else:
                qubo_dict[f"{i_new},{j_new}"] = float(value)

        return Data(Qubo.from_dict(qubo_dict))

    @override
    def postprocess(self, data: Other) -> Result:
        qubo_solution = data.data
        variables = list(self.bqm.variables)
        qubo_solution_dict = dict(zip(variables, qubo_solution))
        lp_solution = dict(self.inverter(qubo_solution_dict))

        return Data(Other(lp_solution))



