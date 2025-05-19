import numpy as np

def fw_direction(grad, x, partitions):
    """
    Computes the Frank-Wolfe (FW) search direction.

    Parameters:
    - grad (ndarray): Gradient vector of the objective function.
    - partitions (list of ndarrays): List of index arrays defining the simplices.
    - x (ndarray): Current iterate.

    Returns:
    - d_fw (ndarray): Frank-Wolfe direction.
    - gap_fw (float): Gap for the FW step.
    """
    
    s = np.zeros_like(x)  # Initialize the FW minimizer
    
    # Compute FW minimizer by selecting the minimum gradient index per simplex
    for part in partitions:
        i = part[np.argmin(grad[part])]  # Index with the smallest gradient
        s[i] = 1  # Assign 1 to the selected index
    
    d_fw = s - x  # Compute FW direction
    gap_fw = -grad @ d_fw  # Compute gap for FW step
    
    return d_fw, gap_fw
