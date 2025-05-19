import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import itertools
from collections import defaultdict 
from data_gen.generate_data import generate_data, start_point
from frank_wolfe.frank_wolfe import frank_wolfe
from utils.logging_utils import log_to_dataframe, make_output_dir, append_final_result
from plot_and_results.plot import save_comparative_plots
from solver_gurobi import solve_with_gurobi


def run_experiment(param_grid, experiment_settings, subfolder=""):
    """
    Runs the main experiment loop across a grid of parameters and algorithm settings.

    Parameters:
    - param_grid (dict): Dictionary specifying values for each experimental parameter.
    - experiment_settings (list of tuples): Each tuple specifies (method, q_generation_mode, violated_constraint).
    - subfolder (str): Directory under which to create the desired subfolder (default: no subfolder).

    Returns:
    - results_by_config (defaultdict): Nested dictionary containing convergence logs by configuration and method.
    - output_dir (str): Path to the directory where results and plots are saved.
    """

    results_by_config = defaultdict(dict)
    output_dir = None

    # Iterate over all combinations of parameters in the grid
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        config = dict(zip(keys, values))

        # Create a dedicated output directory for this config
        output_dir = make_output_dir(config, subfolder=subfolder)
        print(f"\nCONFIG: {config}")

        # Loop over all experiment settings (method + data generation strategy)
        for method, q_mode, viol in experiment_settings:
            label = f"{method}_{q_mode}" + (f"_{viol}" if viol else "")
            plot_key = f"{q_mode}" if not viol else f"{q_mode}_{viol}"
            print(f"\nRun experiment: {label}")

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

            # Starting point
            x0 = start_point(n=config["n"], partitions=partitions)

            # FW algorithm
            start = time.time()
            _, log, check_d = frank_wolfe(
                Q=Q,
                q=q,
                x0=x0,
                partitions=partitions,
                method=method,
                epsilon=1e-6,
                max_iter=50000,
                verbose=False
            )
            end = time.time()
            print(f"{label} ended in {end - start:.2f} s")
            print("x feasible" if check_d else "x infeasible")

            # Use the solver gurobi to get the true optimum ex-post (used to find abs. and rel. gap)
            result = solve_with_gurobi(Q, q, partitions)
            fx_star = result["objective"]
            
            # Logging the results
            df = log_to_dataframe(log, method=method)
            log_filename = f"log_{label}.csv"
            df.to_csv(os.path.join(output_dir, log_filename), index=False)
            append_final_result(df, output_dir, log_filename, fx_star=fx_star)

            # Store results
            results_by_config[plot_key][method] = df

        save_comparative_plots(results_by_config, output_dir)

    return results_by_config, output_dir
