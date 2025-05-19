import numpy as np

def line_search(Q, q, x, d, gamma_max):
    """
    Computes the optimal step size (gamma).

    Parameters:
    - Q (ndarray): Positive semi-definite symmetric matrix of shape (n, n).
    - d (ndarray): Descent direction vector.
    - x (ndarray): Current iterate.
    - q (ndarray): Coefficient vector of shape (n,).
    - gamma_max (float): Upper bound for the step size.

    Returns:
    - gamma (float): Computed step size for the iteration.
    """
    
    Qd = Q @ d

    numerator = -d @ (2 * Q @ x + q)
    denominator = 2 * (d @ Qd)
    
    # Compute step size
    gamma = gamma_max if denominator == 0 else np.clip(numerator / denominator, 0.0, gamma_max)

    return gamma
