import dimod
import numpy as np
import os
import yaml
from pathlib import Path

def converter_model(data):
    sample_key = next(iter(data))
    
    if isinstance(sample_key, tuple) and isinstance(sample_key[0], tuple):
        bqm = dimod.BinaryQuadraticModel('BINARY')
        for (index1, index2), value in data.items():
            var1 = f"v{index1[0]}_{index1[1]}"
            var2 = f"v{index2[0]}_{index2[1]}"
            
            if var1 == var2:
                bqm.add_variable(var1, value)
            else:
                bqm.add_interaction(var1, var2, value)
        return bqm
    else:
        bqm = dimod.BinaryQuadraticModel.from_qubo(data)
        return bqm


def converter_solution(solution):
    objective_values = solution.obj_values
    best_index = np.argmin(objective_values)

    best_sample = solution.samples.tolist()[best_index]
    var_names = solution.variable_names

    sample_dict = {}
    
    for var, val in zip(var_names, best_sample):
        if var.startswith('v') and '_' in var:  # 2-tuple style
            i, j = map(int, var[1:].split('_'))
            sample_dict[(i, j)] = np.int8(val)
        else:  # string keys
            sample_dict[str(var)] = int(val)
    
    return sample_dict


def get_runtime(solution):
    return (solution.runtime.end - solution.runtime.start).total_seconds()

def get_luna_api_key() -> str:
    key = os.getenv("LUNA_API_KEY")
    if key:
        return key

    cred_file = Path("credentials.yaml")
    if cred_file.exists():
        with open(cred_file, "r") as f:
            creds = yaml.safe_load(f)
        if "LUNA_API_KEY" in creds and creds["LUNA_API_KEY"]:
            return creds["LUNA_API_KEY"]

    raise RuntimeError(
        "LUNA_API_KEY not found. Please set it as an environment variable "
        "or provide it in credentials.yaml"
    )

# def matrix_to_qubo_dict(matrix: np.ndarray) -> dict:
#     qubo = {}
#     rows, cols = matrix.shape
#     for i in range(rows):
#         for j in range(cols):
#             if matrix[i, j] != 0:
#                 qubo[(i, j)] = float(matrix[i, j])
#     return qubo

# def convert_solution_qubo_to_lp(solution_dict):
#     named_solution = {}
#     for key, val in solution_dict.items():
#         i = key[0]
#         named_solution[f'x_{i}'] = float(val)
#     return named_solution
