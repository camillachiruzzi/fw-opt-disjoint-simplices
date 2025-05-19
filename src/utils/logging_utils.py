import os
import pandas as pd

def log_to_dataframe(log, method='FW'):
    """
    Convert a log list into a pandas DataFrame.

    Parameters:
        log (list of tuples): List of tuples or lists containing log entries for each iteration.
              Each entry should include:
                [iteration, time, f(x), gap, step_type, gamma, grad_norm, dgrad]
        method (str): Name of the algorithm used to generate the log (default: 'FW').

    Returns:
        df (dataFrame): pandas DataFrame with named columns and an additional 'method' column
             to identify the optimization method used.
    """

    # Create the DataFrame 
    df = pd.DataFrame(log, columns=[
        'iteration', 
        'time', 
        'f(x)', 
        'gap', 
        'step_type', 
        'gamma', 
        'grad_norm', 
        'dgrad'
    ])

    # Add a column indicating the method
    df['method'] = method.upper()

    return df

def make_output_dir(config, subfolder=""):
    """
    Generate an output directory path based on experiment configuration.

    Parameters:
        config (dictionary): Dictionary containing the experiment parameters. Required keys:
                 'n', 'K', 'spectral_radius', 'lambda_min', 'density', 'kernel_dim'.
        subfolder (str):  Directory under which to create the desired subfolder (default: no subfolder).

    Returns:
        path (str): Full path to the created directory identified by the configuration.
    """

    # Construct a tag summarizing the configuration
    tag = "_".join([f"{k}={v}" for k, v in config.items()])

    base_dir = "src/plot_and_results/results"
    if subfolder:
        base_dir = os.path.join(base_dir, subfolder)
    
    # Combine base directory and tag to form the full path
    path = os.path.join(base_dir, tag)

    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    return path

def append_final_result(df, output_dir, log_filename, fx_star=None):
    """
    Extract the last row from the log DataFrame and save it to results.csv,
    adding the log filename and optionally the optimal value and gaps.

    Parameters:
        df (DataFrame): Log DataFrame containing iteration details.
        output_dir (str): Path to the output folder where the summary file will be saved.
        log_filename (str): Name of the log file associated with this result (e.g., 'log_fw_random.csv').
        fx_star (float): Optional, optimal objective value to compute absolute and relative gaps.
    """

    desired_columns = [
        'iteration', 'time', 'f(x)', 'gap',
        'step_type', 'gamma', 'grad_norm', 'dgrad', 'method'
    ]

    summary_row = df.iloc[[-1]][desired_columns]
    summary_row['log_file'] = log_filename

    if fx_star is not None:
        fx = summary_row['f(x)'].values[0]
        summary_row['f*'] = fx_star
        summary_row['abs_gap'] = fx - fx_star
        summary_row['rel_gap'] = (fx - fx_star) / abs(fx_star) if fx_star != 0 else float('inf')

    results_path = os.path.join(output_dir, "results.csv")
    if os.path.exists(results_path):
        summary_row.to_csv(results_path, mode='a', header=False, index=False)
    else:
        summary_row.to_csv(results_path, index=False)
