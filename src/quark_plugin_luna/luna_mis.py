from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Failed, Result
from quark.interface_types import InterfaceType, Other

from luna_quantum.solve.use_cases import MaxIndependentSet
from luna_quantum import Model, LunaSolve, Solution
from luna_quantum.translator import LpTranslator

import random
import networkx as nx

from .utils import get_luna_api_key


@dataclass
class LunaMaxIndependentSet(Core):
    """A module for generating a Maximum Independent Set (MIS) instance via LUNA.
    Upstream: MIS instance as LP string
    Downstream: Objective value of the solution

    :param num_nodes: Number of nodes in the graph (default: 6)
    :param edge_prob: Probability of an edge between two nodes (default: 0.5)
    :param seed: Random seed for reproducibility (default: 123)
    """

    num_nodes: int = 6
    edge_prob: float = 0.5
    seed: int = 123

    def generate_graph(self, num_nodes, edge_prob, seed):
        if seed is not None:
            random.seed(seed)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_prob:
                    G.add_edge(i, j)
        if G.number_of_edges() == 0 and num_nodes >= 2:
            G.add_edge(0, 1)
        self.graph = G

    @override
    def preprocess(self, data: InterfaceType = None) -> Result:
        self.generate_graph(self.num_nodes, self.edge_prob, self.seed)
        LunaSolve.authenticate(get_luna_api_key())
        ls = LunaSolve()
        mis = MaxIndependentSet(graph=nx.to_dict_of_dicts(self.graph))
        meta_model = ls.model.create_from_use_case(name="MaxIndependentSet", use_case=mis)
        self.model = Model.load_luna(model_id=meta_model.id)
        lp_model = LpTranslator.from_aq(self.model)
        return Data(Other[str](lp_model))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        lp_solution = data.data
        if lp_solution is None:
            return Failed("No solution found")
        
        solution = Solution.from_dict(model=self.model, data=data.data)
        if not solution.best().feasible:
            return Failed("Invalid solution.")
        obj_value = abs(solution.best().obj_value)
        return Data(Other(obj_value))

        # indep_set = {i: 1 if lp_solution.get(f"x_{i}", 0.0) >= 0.5 else 0 for i in self.graph.nodes()}

        # # validate: no two adjacent nodes in set
        # for (u, v) in self.graph.edges():
        #     if indep_set[u] + indep_set[v] > 1:
        #         return Failed("Invalid solution: adjacent nodes in independent set")

        # obj_value = sum(indep_set.values())
        # return Data(Other(obj_value))
