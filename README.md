# QUARK-plugin-luna

### Provided Modules:

Luna Usecases provide an LP string, solveable directly by the scip solver or 
transformable to qubo by the `lp_qubo_maping` module.
This transformation will however encode constraints as slack variables.
Constraint conserving algorithms are `scip_solver` and `luna_flex` 

 <br>

**Solving Algortihms**
| Module               | Upstream Interface          | Downstream Interface       |
|----------------------| --------------------------- |----------------------------|
| luna_sa              | quark.interface_types.qubo  | None                       |
| luna_saga              | quark.interface_types.qubo  | None                       |
| luna_qa              | quark.interface_types.qubo  | None                       |
| scip_solver          | quark.interface_types.other (LP) | None                  |
| luna_pt              | quark.interface_types.qubo  | None                       |
| luna_qaoa            | quark.interface_types.qubo  | None                       |
| luna_flex            | quark.interface_types.other (LP) | None                  |

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
