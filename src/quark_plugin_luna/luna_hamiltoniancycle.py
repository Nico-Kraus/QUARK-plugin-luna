from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Failed, Result
from quark.interface_types import InterfaceType, Other

from luna_quantum.solve.use_cases import HamiltonianCycle
from luna_quantum import Model, LunaSolve
from luna_quantum.translator import LpTranslator

import networkx as nx
import random

from .utils import get_luna_api_key


@dataclass
class LunaHamiltonianCycle(Core):
    """A module for creating a Hamiltonian Cycle instance via LUNA.
    Upstream: Hamiltonian Cycle instance as LP string
    Downstream: Objective value of the solution if a valid cycle exists

    :param num_nodes: Number of nodes in the graph (default: 5)
    :param edge_prob: Probability of an edge between nodes (default: 0.5)
    :param seed: Random seed for reproducibility (default: 123)
    """

    num_nodes: int = 5
    edge_prob: float = 0.5
    seed: int = 123

    def generate_graph(self, num_nodes: int, edge_prob: float, seed: int):
        if seed is not None:
            random.seed(seed)

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_prob:
                    G.add_edge(i, j)

        # Guarantee connectivity or at least one cycle
        if G.number_of_edges() < num_nodes:
            for i in range(num_nodes - 1):
                G.add_edge(i, i + 1)
            G.add_edge(0, num_nodes - 1)

        self.graph = G

    @override
    def preprocess(self, data: InterfaceType = None) -> Result:
        self.generate_graph(self.num_nodes, self.edge_prob, self.seed)
        LunaSolve.authenticate(get_luna_api_key())
        ls = LunaSolve()

        hc = HamiltonianCycle(graph=nx.to_dict_of_dicts(self.graph))
        meta_model = ls.model.create_from_use_case(name="HamiltonianCycle", use_case=hc)
        model = Model.load_luna(model_id=meta_model.id)
        lp_model = LpTranslator.from_aq(model)
        return Data(Other[str](lp_model))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        lp_solution = data.data
        if lp_solution is None:
            return Failed("No solution found")

        chosen_edges = []
        for (u, v) in self.graph.edges():
            var_name = f"x_{u}_{v}"
            val = lp_solution.get(var_name, 0.0)
            if val >= 0.5:
                chosen_edges.append((u, v))

        H = nx.Graph()
        H.add_nodes_from(self.graph.nodes())
        H.add_edges_from(chosen_edges)

        if not nx.is_connected(H):
            return Failed("Invalid solution: subgraph not connected")
        if any(deg != 2 for _, deg in H.degree()):
            return Failed("Invalid solution: not all nodes have degree 2")
        if len(H.edges()) != self.num_nodes:
            return Failed("Invalid solution: wrong number of edges")

        try:
            cycle = list(nx.find_cycle(H))
        except nx.NetworkXNoCycle:
            return Failed("Invalid solution: no Hamiltonian cycle found")

        obj_value = len(cycle)
        return Data(Other(obj_value))

