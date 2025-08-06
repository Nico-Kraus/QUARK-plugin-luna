import dimod
import numpy as np

def converter_model(data):
    """
    Converts a QUBO problem represented as a dictionary into a Binary Quadratic Model (BQM).
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')

    for (index1, index2), value in data.items():
        var1 = f"v{index1[0]}_{index1[1]}"
        var2 = f"v{index2[0]}_{index2[1]}"
        
        if var1 == var2:
            bqm.add_variable(var1, value)
        else:
            bqm.add_interaction(var1, var2, value)

    return bqm

def converter_solution(solution):
    """
    Converts the best sample from the simulated annealing solution into a dictionary format.
    """
   
    objective_values = solution.obj_values
    best_index = np.argmin(objective_values)

    best_sample = solution.samples.tolist()[best_index]
    var_names = solution.variable_names
    solution_dict = {}

    for var, val in zip(var_names, best_sample):
        # Extract index: 'v2_3' -> (2, 3)
        i, j = map(int, var[1:].split('_'))
        solution_dict[(i, j)] = np.int8(val)

    return solution_dict