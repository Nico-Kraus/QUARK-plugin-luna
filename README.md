# QUARK-plugin-luna

### Provided Modules:

Luna Usecases provide an LP string, solveable directly by the scip solver or 
transformable to qubo by the `lp_qubo_maping` module.
This transformation will encode constraints as slack variables.
Constraint conserving algorithms are `scip_solver` and `luna_flex` 

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
| luna_sa              | quark.interface_types.qubo | None                       | Classical
| luna_saga            | quark.interface_types.qubo | None                       | Classical
| luna_qbsa            | quark.interface_types.qubo | None                       | Classical
| luna_rrsa            | quark.interface_types.qubo | None                       | Classical
| luna_pa              | quark.interface_types.qubo | None                       | Classical
| luna_kerb            | quark.interface_types.qubo | None                       | Classical
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