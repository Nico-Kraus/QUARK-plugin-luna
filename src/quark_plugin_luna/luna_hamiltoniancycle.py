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

    def generate_hamiltonian_graph(self, num_nodes: int, edge_prob: float, seed: int):
        '''Generates a graph with a guaranteed hamiltonian cycle'''
        
        random.seed(seed)

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # First, create a Hamiltonian cycle to guarantee one exists
        for i in range(num_nodes):
            G.add_edge(i, (i + 1) % num_nodes)
        
        # Then add random edges with the given probability
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Skip if edge already exists (from the Hamiltonian cycle)
                if not G.has_edge(i, j):
                    if random.random() < edge_prob:
                        G.add_edge(i, j)

        self.graph = G
        # try:
        #     nx.find_cycle(G)
        #     print("Graph has HC.")
        # except nx.NetworkXNoCycle:
        #     print("WARNING: Problem Graph has no HC.")

    

    @override
    def preprocess(self, data: InterfaceType = None) -> Result:
        self.generate_hamiltonian_graph(self.num_nodes, self.edge_prob, self.seed)
        LunaSolve.authenticate(get_luna_api_key())
        ls = LunaSolve()
        str_graph = nx.relabel_nodes(self.graph, lambda x: str(x))
        hc = HamiltonianCycle(graph=nx.to_dict_of_dicts(str_graph))
        meta_model = ls.model.create_from_use_case(name="HamiltonianCycle", use_case=hc)
        model = Model.load_luna(model_id=meta_model.id)
        lp_model = LpTranslator.from_aq(model)
        return Data(Other[str](lp_model))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        lp_solution = data.data
        if lp_solution is None:
            return Failed("No solution found")
        else:
            return self.evaluate_qubo_hamiltonian_solution(lp_solution)

    def evaluate_qubo_hamiltonian_solution(self, lp_solution):
        """
        Evaluate QUBO-based Hamiltonian cycle solution.
        Variables x_i represent position-based encoding where:
        - x_{i*n + j} = 1 means node i is at position j in the cycle
        """
        if lp_solution is None:
            return Failed("No solution found")
        
        n = self.num_nodes
        
        # Extract the position matrix
        position_matrix = {}  # position_matrix[node][position] = value
        for i in range(n):
            position_matrix[i] = {}
            for j in range(n):
                var_name = f"x_{i*n + j}"  # Flattened index
                value = lp_solution.get(var_name, 0.0)
                position_matrix[i][j] = value
        
        # Construct the cycle from position assignments
        cycle_positions = {}  # position -> node
        node_positions = {}   # node -> position
        
        for node in range(n):
            for position in range(n):
                if position_matrix[node][position] >= 0.5:
                    if position in cycle_positions:
                        return Failed(f"Invalid solution: Multiple nodes assigned to position {position}")
                    if node in node_positions:
                        return Failed(f"Invalid solution: Node {node} assigned to multiple positions")
                    
                    cycle_positions[position] = node
                    node_positions[node] = position
                
        # Validate that we have a complete assignment
        if len(cycle_positions) != n:
            return Failed(f"Invalid solution: Only {len(cycle_positions)} positions assigned, need {n}")
        
        if len(node_positions) != n:
            return Failed(f"Invalid solution: Only {len(node_positions)} nodes assigned, need {n}")
        
        # Construct the cycle as a sequence of nodes
        cycle_sequence = []
        for position in range(n):
            if position not in cycle_positions:
                return Failed(f"Invalid solution: Position {position} not assigned")
            cycle_sequence.append(cycle_positions[position])
                
        # Validate that consecutive nodes in the cycle are connected by edges
        cycle_edges = []
        for i in range(n):
            current_node = cycle_sequence[i]
            next_node = cycle_sequence[(i + 1) % n]  # Wrap around for last edge
            
            if not self.graph.has_edge(current_node, next_node):
                return Failed(f"Invalid solution: No edge between {current_node} and {next_node}")
            
            cycle_edges.append((current_node, next_node))
        
        # Create the resulting graph to double-check
        H = nx.Graph()
        H.add_nodes_from(range(n))
        H.add_edges_from(cycle_edges)
        
        if not nx.is_connected(H):
            return Failed("Invalid solution: Resulting cycle is not connected")
        
        if len(H.edges()) != n:
            return Failed(f"Invalid solution: Expected {n} edges, got {len(H.edges())}")
        
        if not all(deg == 2 for _, deg in H.degree()):
            degrees = dict(H.degree())
            return Failed(f"Invalid solution: Not all nodes have degree 2: {degrees}")
                
        # Return the cycle length (should always be n for Hamiltonian cycle)
        obj_value = len(cycle_sequence)
        return Data(Other(obj_value))

