# fw-opt-disjoint-simplices
Implementation and benchmarking of Frank-Wolfe algorithms for quadratic optimization over disjoint simplices.

# Quadratic Optimization on Disjoint Simplices

The problem to be solved is a convex quadratic program on a set of disjoint simplices using an algorithm of the class of **conditional gradient methods** (a.k.a. **Frank-Wolfe type**). Specifically, the objective is to minimize the quadratic function $f : \mathcal{D} \subseteq \mathbb{R}^n \to \mathbb{R}$ of the form:

$$
f(x) = x^\top Q x + q^\top x,
$$

where $Q \in \mathbb{R}^{n \times n}$ is a **positive semidefinite symmetric matrix**, and $x, q \in \mathbb{R}^n$.

The constrained domain $\mathcal{D}$ is defined as follows:

$$
\mathcal{D} = \left\lbrace x \in \mathbb{R}^n \mid \sum_{i \in I^k} x_i = 1, \ \forall k \in K, \ x \geq 0 \right\rbrace
$$

where the index sets $I^k$ form a partition of $$\\\{1, \dots, n\\\}$$, and each **simplex** corresponds to one of these sets. So,

$$
\bigcup_{k \in K} I^k = \\\{1, \dots, n\\\}, \quad \text{and} \quad I^h \cap I^k = \emptyset
$$

Thus, the problem **(P)** can be written in a more compact way as:

$$
\min \left\lbrace x^\top Q x + q^\top x \ : \ \sum_{i \in I^k} x_i = 1,\ \forall k \in K,\ x \geq 0 \right\rbrace
$$

The goal is to find a solution $x^*$ that minimizes the objective function subject to the specified constraints.

Algorithms
----------

The project implements and compares the following algorithms:
- Standard Frank-Wolfe (FW)
- Away-Step Frank-Wolfe (AFW)
- Gurobi optimizer (used as benchmark)

Project Structure
-----------------
```
opt/
└── src/
    ├── data_gen/
    │   ├── generate_data.py         # Generate Q, q and simplex structure
    │   └── start_point.py           # Generate feasible initial point
    ├── experiments/
    │   ├── gurobi_experiment.py     # Compare AFW and Gurobi across settings
    │   ├── kernel_experiments.py    # Experiments on varying kernel dimension
    │   ├── n_experiments.py         # Experiments on varying dimension n
    │   ├── q_experiments.py         # Experiments on q generation modes
    │   ├── rho_experiments.py       # Experiments on varying spectral radius
    │   ├── run_experiment.py        # Core function to run and log experiments
    │   └── solver_gurobi.py         # Solver Gurobi
    ├── frank_wolfe/
    │   ├── afw_direction.py         # AFW direction
    │   ├── check_domain.py          # Check feasibility of a point
    │   ├── frank_wolfe.py           # Main FW/AFW function
    │   ├── fw_direction.py          # FW direction
    │   └── line_search.py           # Exact line search
    ├── plot_and_results/
    │   ├── plot.py                  # Plotting convergence and results
    │   └── results/                 # Automatically saved logs and plots
    └── utils/
        └── logging_utils.py         # Logging, formatting, and saving tools
```

Usage
-----

To run an experiment, execute one of the scripts in `src/experiments/`, each named after the parameter it varies (e.g., `n_experiments.py`, `q_experiments.py`, etc.).

Example:

   python src/experiments/n_experiments.py

   You can customize:
   - n, K: problem size and number of simplices
   - spectral_radius, lambda_min, density, kernel_dim: matrix structure
   - q_generation: strategy for generating q ('random', 'internal', 'external')
   - method: 'fw' or 'afw'

Results
-------

- Logs are saved in results/ subfolders by configuration
- Logs of the results include iteration, time, f(x), gap, step_type, gamma, grad_norm, dgrad, method, log_file, f*, abs_gap, rel_gap 
- Plots and CSV exports available
- Logarithmic convergence plots

Documentation
-------------

Based on the report:
"Quadratic Optimization on Simplices: An Implementation of the Frank-Wolfe Algorithm"
by Camilla Chiruzzi and Niccolò Seghieri, University of Pisa

