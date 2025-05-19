import numpy as np
import time
from .afw_direction import afw_direction
from .fw_direction import fw_direction
from .line_search import line_search
from .check_domain import check_domain

def frank_wolfe(Q, q, x0, partitions, method='fw', epsilon=1e-6, max_iter=1000, verbose=True):
    """
    Implements the Frank-Wolfe (FW) and Away-Step Frank-Wolfe (AFW) algorithms.

    Parameters:
    - Q (ndarray): Positive semi-definite symmetric matrix of shape (n, n).
    - q (ndarray): Coefficient vector of shape (n,).
    - x0 (ndarray): Initial point.
    - partitions (list of ndarrays): List of index arrays defining disjoint simplices.
    - method (str): 'fw' for standard Frank-Wolfe, 'afw' for Away-Step Frank-Wolfe.
    - epsilon (float): Convergence tolerance based on duality gap.
    - max_iter (int): Maximum number of iterations.
    - verbose (bool): If True, prints progress at each iteration.

    Returns:
    - x (ndarray): Optimized solution.
    - log (list): List of tuples tracking:
        (iteration, elapsed time, objective value, dual gap, step type, step size, grad norm, directional derivative)
    - check_d (bool): Check if the solution found is feasible or not.
    """

    # Define the objective function and gradient
    f = lambda x: x @ Q @ x + q @ x
    grad = lambda x: 2 * Q @ x + q

    x = x0.copy()
    log = []
    start_time = time.time()

    for i in range(max_iter):
        g = grad(x)

        # Compute FW direction
        d_fw, gap_fw = fw_direction(g, x, partitions)
        gamma_max = 1
        step_type = 'FW'
        
        if method == 'afw':
            d_aw, gamma_max_aw, gap_aw  = afw_direction(g, x, partitions)
            if gap_aw > gap_fw:
                d_fw, gamma_max = d_aw, gamma_max_aw
                step_type = 'AW'

        gap = -g @ d_fw
        fx = f(x)
        grad_norm = np.linalg.norm(g)
        directional_derivative = g @ d_fw
        
        elapsed_time = time.time() - start_time
        
        log.append((i, elapsed_time, fx, gap, step_type, None, grad_norm, directional_derivative))

        if verbose:
            print(f"Iter {i:3d} | f(x) = {fx:.6f} | gap = {gap:.2e} | step = {step_type}")

        if gap < epsilon:
            break

        # Compute step size gamma
        gamma = line_search(Q, q, x, d_fw, gamma_max)
        x += gamma * d_fw

        log[-1] = (i, elapsed_time, fx, gap, step_type, gamma, grad_norm, directional_derivative)

    # Check if the solution found is feasible or not
    check_d = check_domain(x, partitions)
    
    return x, log, check_d
