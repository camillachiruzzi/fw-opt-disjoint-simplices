import numpy as np

def afw_direction(grad, x, partitions):
    """
    Computes the Away Frank-Wolfe (AFW) direction.

    Parameters:
    - x (ndarray): Current iterate.
    - grad (ndarray): Gradient of the objective function at x.
    - partitions (list of ndarrays): List of index arrays defining the simplices.

    Returns:
    - d_aw (ndarray): AFW direction.
    - gamma_max (float): Maximum feasible step size for AFW step.
    - gap_aw (float): Gap for AFW step.
    """
    # Initialize away vertex representation and maximum step size
    v = np.zeros_like(x)
    gamma_max = float('inf')
    
    # Find the worst vertex in each partition
    for part in partitions:
        # Consider only active variables (those strictly greater than zero)
        active = [i for i in part if x[i] > 0]
        if not active:
            continue
        
        # Select the index corresponding to the worst gradient (maximization step)
        i = max(active, key=lambda i: grad[i])
        v[i] = 1
        
        # Compute maximum step size for moving away from the active vertex
        gamma_i = x[i] / (1 - x[i]) if x[i] < 1 else 1.0
        gamma_max = min(gamma_max, gamma_i)
    
    # Compute the AFW direction
    d_aw = x - v
    
    # Compute the gap for AFW
    gap_aw = -grad @ d_aw
    
    return d_aw, gamma_max, gap_aw
