from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Failed, Result
from quark.interface_types import InterfaceType, Other

from luna_quantum.solve.use_cases import MinimumVertexCover as MinVertexCover
from luna_quantum import Model, LunaSolve
from luna_quantum.translator import LpTranslator

import random
import networkx as nx

from .utils import scale_sleep, get_luna_api_key


@dataclass
class LunaMinVertexCover(Core):
    """A module for creating a Minimum Vertex Cover instance via LUNA.
    Upstream: Minimum Vertex Cover instance as LP string
    Downstream: Objective value of the solution

    :param num_nodes: Number of nodes in the graph (default: 6)
    :param edge_prob: Probability of an edge between any two nodes (default: 0.5)
    :param seed: Random seed for reproducibility (default: 123)
    """

    num_nodes: int = 6
    edge_prob: float = 0.5
    seed: int = 123
    penalty: int = 8

    def generate_graph(self, num_nodes, edge_prob, seed):
        if seed is not None:
            random.seed(seed)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_prob:
                    self.graph.add_edge(i, j)
        if self.graph.number_of_edges() == 0:
            self.graph.add_edge(0, 1)

    def evaluate_solution(self, solution: dict) -> tuple[int, bool]:
        cover = {i: 1 if solution.get(f"x_{i}", 0.0) >= 0.5 else 0 for i in self.graph.nodes()}
        valid = True
        for (u, v) in self.graph.edges():
            if cover[u] + cover[v] < 1:
                valid = False
                break
        obj_value = sum(cover.values())
        return obj_value, valid

    @override
    def preprocess(self, data: InterfaceType = None) -> Result:
        self.generate_graph(self.num_nodes, self.edge_prob, self.seed)
        LunaSolve.authenticate(get_luna_api_key())
        ls = LunaSolve()
        str_graph = nx.relabel_nodes(self.graph, lambda x: str(x))
        graph_dict = nx.to_dict_of_dicts(str_graph)
        mvc = MinVertexCover(graph=graph_dict, P=self.penalty)
        meta_model = ls.model.create_from_use_case(name="MinVertexCover", use_case=mvc)
        model = Model.load_luna(model_id=meta_model.id)
        lp_model = LpTranslator.from_aq(model)

        return Data(Other[str](lp_model))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        lp_solution = data.data
        if lp_solution is None:
                return Failed("No solution found")

        obj_value, valid = self.evaluate_solution(lp_solution)
        if not valid:
            return Failed("Invalid solution: edge not covered")

        return Data(Other(obj_value))