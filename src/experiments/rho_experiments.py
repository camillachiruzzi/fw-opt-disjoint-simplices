from run_experiment import run_experiment

param_grid = {
        "n": [100], 
        "K": [20], 
        "spectral_radius": [2, 10, 50, 100, 200],
        "lambda_min": [1],
        "density": [1.0],
        "kernel_dim": [0],
        "seed": [42]
    }

experiment_settings = [
        ("fw", "random", None),
        ("afw", "random", None),
        ("fw", "internal", None),
        ("afw", "internal", None),
        ("fw", "external", "sum_simplex"),
        ("afw", "external", "sum_simplex"),
        ("fw", "external", "non_negative"),
        ("afw", "external", "non_negative"),
        ("fw", "external", "all"),
        ("afw", "external", "all"),
    ]

# Run the experiments with the specified parameter setting
results_by_config, output_dir = run_experiment(param_grid, experiment_settings, "rho-experiment")
