import math
import os
import yaml
from pathlib import Path

from quark.interface_types import Qubo

from luna_quantum import Model, Solution
from luna_quantum.translator import QuboTranslator

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

def get_model(data:Qubo) -> Model:
    print("matrix:\n", data.as_matrix())
    return QuboTranslator.to_aq(data.as_matrix())

def get_best_solution(solution:Solution)->list:
    if solution is not None:
        return solution.best().sample.to_dict()
    else:
        return None


def convert_qubo(bqm):
    qubo, _ = bqm.to_qubo()
    varmap = {}
    for (i, j) in qubo.keys():
        for v in (i, j):
            if v not in varmap:
                idx = v.split("_")[1]
                varmap[v] = f"q{idx}"

    qubo_dict = {}
    for (i, j), value in qubo.items():
        i_new, j_new = varmap[i], varmap[j]
        if i_new == j_new:
            qubo_dict[i_new] = float(value)
        else:
            qubo_dict[f"{i_new},{j_new}"] = float(value)
    return qubo_dict

def get_runtime(solution):
    if solution is not None and solution.runtime is not None:
        return solution.runtime.total_seconds
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
