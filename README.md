# QUARK-plugin-luna

### Problem Classes:

**Set Packing (`luna_setpacking`)**
- Creates a set packing optimization problem where the goal is to select non-overlapping subsets from a collection to maximize total weight
- Parameters: `set_size` (number of subsets), `universe_size` (total elements), `density` (inclusion probability), `weights` (equal/random), `seed`

**Hamiltonian Cycle (`luna_hamiltoniancycle`)**  
- Generates a Hamiltonian cycle problem on a random graph, seeking a cycle that visits each node exactly once
- Parameters: `num_nodes` (graph size), `edge_prob` (edge probability), `seed`
- Guarantees connectivity by adding a ring structure if needed

**Maximum Cut (`luna_maxcut`)**
- Creates a max-cut problem where the objective is to partition graph nodes to maximize the total weight of edges crossing the partition
- Parameters: `num_nodes`, `edge_prob`, `weight_range` (min/max edge weights), `seed`
- Ensures at least one edge exists for meaningful optimization

**Minimum Vertex Cover (`luna_minvertexcover`)**
- Generates a minimum vertex cover problem: find the smallest set of vertices such that every edge has at least one endpoint in the set
- Parameters: `num_nodes`, `edge_prob`, `seed` 
- Guarantees at least one edge to ensure non-trivial problem instances

**Maximum Independent Set (`luna_mis`)**
- Creates a maximum independent set problem: find the largest set of vertices with no edges between them
- Parameters: `num_nodes`, `edge_prob`, `seed`
- Ensures at least one edge exists to create meaningful constraints

### Solving Algorithms and Mapping:

Luna Usecases provide an LP string, solveable directly by the scip solver or 
transformable to qubo by the `lp_qubo_maping` module.
This transformation will encode constraints as slack variables.
Constraint conserving algorithms are `scip_solver`, `luna_leapcqm`, as well as `luna_flex` in a sense where
constraints are encoded in a limited search space for QAOA.

All module parameters are documented in detail within their respective `.py` files.

<br>

**Solving Algortihms**
| Module               | Upstream Interface          | Downstream Interface       | Type
|----------------------|----------------------------|----------------------------|----------------------------|
| luna_qa              | quark.interface_types.qubo | None                       | Quantum
| luna_qaoa            | quark.interface_types.qubo | None                       | Quantum
| luna_qaga            | quark.interface_types.qubo | None                       | Quantum
| luna_qbqa            | quark.interface_types.qubo | None                       | Quantum
| luna_qpa             | quark.interface_types.qubo | None                       | Quantum
| luna_rrqa            | quark.interface_types.qubo | None                       | Quantum
| luna_leapbqm         | quark.interface_types.qubo | None                       | Hybrid
| luna_kerb            | quark.interface_types.qubo | None                       | Hybrid
| luna_sa              | quark.interface_types.qubo | None                       | Classical
| luna_saga            | quark.interface_types.qubo | None                       | Classical
| luna_qbsa            | quark.interface_types.qubo | None                       | Classical
| luna_rrsa            | quark.interface_types.qubo | None                       | Classical
| luna_pa              | quark.interface_types.qubo | None                       | Classical
| luna_flex_qaoa       | quark.interface_types.other (LP) | None                  | Quantum
| luna_leapcqm         | quark.interface_types.other (LP) | None                  | Hybrid
| scip_solver          | quark.interface_types.other (LP) | None                  | Classical

<br>

**Mapping**
| Module               | Upstream Interface          | Downstream Interface       |
|----------------------| --------------------------- |----------------------------|
| lp_qubo_mapping      | quark.interface_types.other (LP) | quark.interface_types.qubo |

<br>

**Use Cases**
| Module               | Upstream Interface          | Downstream Interface       |
|----------------------| --------------------------- |----------------------------|
| luna_setpacking      | None                        | quark.interface_types.other (LP)|
| luna_hamiltoniancycle| None                        | quark.interface_types.other (LP) |
| luna_minvertexcover  | None                        | quark.interface_types.other (LP) |
| luna_mis             | None                        | quark.interface_types.other (LP) |
| luna_maxcut          | None                        | quark.interface_types.other (LP) |