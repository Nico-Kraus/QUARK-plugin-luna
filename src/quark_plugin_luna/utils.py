import dimod
import numpy as np
import math
import os
import yaml
from pathlib import Path

from luna_quantum import Model

sleep_params = {"sleep_time_increment": 0.1, "sleep_time_initial":  0.1, "sleep_time_max": 5,}

def scale_sleep(model: Model):
    '''Scales the query intervalls to luna according to the number of variables.'''
    
    problem_size = len(model.variables())
    if problem_size <= 0:
        raise ValueError("problem_size must be > 0")
    factor = math.ceil(problem_size / 100)
    sleep_params_copy = sleep_params.copy()
    for k in sleep_params_copy:
        sleep_params_copy[k] *= factor
    return sleep_params_copy

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
    elif isinstance(sample_key, str):
        bqm = dimod.BinaryQuadraticModel("BINARY")
        for key, value in data.items():
            if ',' not in key:
                bqm.add_variable(key, value)
            else:
                bqm.add_interaction(key.split(',')[0], key.split(',')[1], value)
        # bqm = dimod.BinaryQuadraticModel.from_qubo({(str(k[1:]), str(k[1:])): v for k, v in data.items() if ',' not in k} |
        #                                           {(str(k.split(',')[0][1:]), str(k.split(',')[1][1:])): v for k, v in data.items() if ',' in k})
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
    if solution is not None:
        return (solution.runtime.end - solution.runtime.start).total_seconds()
    else:
        return None

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

def count_logical_qubits(qubo_dict):
    vars_in_qubo = set()
    for i, j in qubo_dict.keys():
        vars_in_qubo.add(i)
        vars_in_qubo.add(j)
    return len(vars_in_qubo)
