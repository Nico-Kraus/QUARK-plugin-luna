from dataclasses import dataclass
from typing import override

from quark.core import Core, Data, Failed, Result
from quark.interface_types import InterfaceType, Other

from luna_quantum.solve.use_cases import MaxCut
from luna_quantum import Model, LunaSolve, Solution
from luna_quantum.translator import LpTranslator

import random
import networkx as nx

from .utils import get_luna_api_key


@dataclass
class LunaMaxCut(Core):
    """A module for creating a Max-Cut instance from LUNA.
    Upstream: Max-Cut instance as LP string
    Downstream: Objective value of the solution

    :param num_nodes: Number of nodes in the graph (default: 6)
    :param edge_prob: Probability of an edge between two nodes (default: 0.5)
    :param weight_range: Tuple (min_weight, max_weight) for edge weights (default: (1, 10))
    :param seed: Random seed for reproducibility (default: 123)
    """

    num_nodes: int = 6
    edge_prob: float = 0.5
    weight_range: tuple = (1, 10)
    seed: int = 123

    def generate_graph(self, num_nodes, edge_prob, weight_range, seed=None):
        '''Created a spanning tree first and adds edges afterwards, to avoid isolated nodes.'''
        if seed is not None:
            random.seed(seed)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Step 1: build a random spanning tree (connects all nodes)
        nodes = list(G.nodes)
        random.shuffle(nodes)
        for i in range(1, num_nodes):
            u = nodes[i]
            v = random.choice(nodes[:i])
            w = random.randint(*weight_range)
            G.add_edge(u, v, weight=w)
        
        # Step 2: add random extra edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if not G.has_edge(i, j) and random.random() < edge_prob:
                    w = random.randint(*weight_range)
                    G.add_edge(i, j, weight=w)
        
        self.graph = G

    @override
    def preprocess(self, data: InterfaceType = None) -> Result:
        self.generate_graph(self.num_nodes, self.edge_prob, self.weight_range, self.seed)
        LunaSolve.authenticate(get_luna_api_key())
        ls = LunaSolve()
        str_graph = nx.relabel_nodes(self.graph, lambda x: str(x))
        
        graph_dict = nx.to_dict_of_dicts(str_graph)
        maxcut = MaxCut(graph=graph_dict)
        meta_model = ls.model.create_from_use_case(name="MaxCut", use_case=maxcut)
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
        
        # partition = {i: 1 if lp_solution.get(f"x_{i}", 0.0) >= 0.5 else 0 for i in self.graph.nodes()}
        # cut_value = 0
        # for u, v, w in self.graph.edges(data="weight", default=1):
        #     if partition[u] != partition[v]:
        #         cut_value += w

        # return Data(Other(cut_value))