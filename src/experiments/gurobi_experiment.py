from run_experiment import run_experiment
from solver_gurobi import run_gurobi

### Standard configuration ###
param_grid = {
    "n": [100],
    "K": [20],
    "spectral_radius": [5],
    "lambda_min": [1],
    "density": [1.0],
    "kernel_dim": [0],
    "seed": [42],
}

experiment_settings = [
    ("afw", "random", None),
    ("afw", "internal", None),
    ("afw", "external", "sum_simplex"),
    ("afw", "external", "non_negative"),
    ("afw", "external", "all"),
]

# AFW execution
run_experiment(param_grid, experiment_settings, subfolder="gurobi")

# Gurobi execution
run_gurobi(param_grid, experiment_settings)


### Ill-conditioned configuration ###
param_grid = {
    "n": [1000],
    "K": [20, 200],
    "spectral_radius": [500],
    "lambda_min": [1],
    "density": [1.0],
    "kernel_dim": [20],
    "seed": [42],
}

experiment_settings = [
    ("afw", "random", None),
    ("afw", "internal", None),
    ("afw", "external", "sum_simplex"),
    ("afw", "external", "non_negative"),
    ("afw", "external", "all"),
]

# AFW execution
run_experiment(param_grid, experiment_settings, subfolder="gurobi")

# Gurobi execution
run_gurobi(param_grid, experiment_settings)
