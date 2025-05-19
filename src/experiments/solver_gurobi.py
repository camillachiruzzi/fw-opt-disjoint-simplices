import os
import time
import numpy as np
from itertools import product
import gurobipy as gp
from gurobipy import GRB
from data_gen.generate_data import generate_data
from utils.logging_utils import log_to_dataframe, append_final_result, make_output_dir

def gurobi_logging_callback(log, start_time):
    """
    Constructs a callback function for Gurobi that logs solver progress
    at each iteration.

    Parameters:
    - log (list): A reference to a list where iteration data will be appended.
                  Each entry will follow the format:
                  [iteration, time, f(x), gap, step_type, gamma, grad_norm, dgrad].

    Returns:
    - callback (function): A function that records the iteration count, 
                           objective value, and timestamp at each iteration.
    """
    def callback(model, where):
        if where == gp.GRB.Callback.BARRIER:
            it = model.cbGet(gp.GRB.Callback.BARRIER_ITRCNT)
            obj = model.cbGet(gp.GRB.Callback.BARRIER_PRIMOBJ)
            timestamp = time.time() - start_time
            log.append([it, timestamp, obj, 0.0, None, None, None, None])
    return callback

def solve_with_gurobi(Q, q, partitions):
    """
    Solves the quadratic program using Gurobi.

    Parameters:
    - Q (ndarray): Positive semi-definite symmetric matrix of shape (n, n).
    - q (ndarray): Coefficient vector of shape (n,).
    - partitions (list of ndarrays): List of index arrays defining the simplices.

    Returns:
    - result (dict): Dictionary containing:
        - 'x' (ndarray): Optimal solution vector.
        - 'objective' (float): Optimal objective value.
        - 'time' (float): Solver runtime in seconds.
        - 'status' (int): Gurobi status code (2 = optimal).
        - 'log' (list): List containing the all the information at each iteration.
    """
    
    n = len(q)

    # Model constructor
    model = gp.Model("QP_Gurobi")
    
    log = []

    start_time = time.time()

    callback = gurobi_logging_callback(log, start_time)

    # Add multiple decision variables to the model
    x = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

    # Add a constraint for each partition 
    for k, part in enumerate(partitions):
        model.addConstr(gp.quicksum(x[i] for i in part) == 1, name=f"simp{k}")

    # Construct the quadratic form
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            if Q[i, j] != 0.0:
                obj.add(Q[i, j] * x[i] * x[j])
    
    obj.add(gp.LinExpr((q[i], x[i]) for i in range(n)))
    
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize(callback)
    
    elapsed = model.Runtime

    x_sol = np.array([x[i].X for i in range(n)])

    return {
        "x": x_sol,
        "objective": model.ObjVal,
        "time": elapsed,
        "status": model.Status,
        "log": log if log else [[0, elapsed, model.ObjVal, 0.0, None, None, None, None]]
    }

def run_gurobi(param_grid, experiment_settings, subfolder="gurobi"):
    """
    Runs Gurobi on multiple problem instances defined by a parameter grid.

    For each combination of parameters, it generates a quadratic 
    optimization problem, solves it with Gurobi, and saves results as CSV.

    Parameters:
    - param_grid (dict): Dictionary specifying values for each parameter.
    - experiment_settings (list of tuples): Each tuple specifies (_, q_generation_mode, violated_constraint).
    - subfolder (str): Directory under which to create the desired subfolder (default: gurobi).
    """

    # Iterate over all combinations of parameters in the grid
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        config = dict(zip(keys, values))

        # Create a dedicated output directory for this config
        output_dir = make_output_dir(config, subfolder=subfolder)

        # Loop over all experiment settings (method + data generation strategy)
        for _, q_mode, viol in experiment_settings:
            label = f"gurobi_{q_mode}" + (f"_{viol}" if viol else "")
            print(f"\nGurobi solving for: {label}")

            # Instance generation
            Q, q, partitions = generate_data(
                n=config["n"],
                K=config["K"],
                spectral_radius=config["spectral_radius"],
                lambda_min=config["lambda_min"],
                density=config["density"],
                kernel_dim=config["kernel_dim"],
                q_generation=q_mode,
                violated_constraint=viol,
                seed=config["seed"]
            )

            # Solve using gurobi
            result = solve_with_gurobi(Q, q, partitions)

            # Logging the results
            df = log_to_dataframe(result["log"], method="gurobi")

            log_file = f"log_gurobi_{q_mode}" + (f"_{viol}" if viol else "") + ".csv"
            df.to_csv(os.path.join(output_dir, log_file), index=False)
            append_final_result(df, output_dir, log_file)
