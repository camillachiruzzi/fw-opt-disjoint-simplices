import numpy as np

def check_domain(x, partitions):
    """
    Determine if x is inside the domain

    Parameters:
        - x (ndarray): array of variables to be check
        - partitions (list of ndarrays): List of index arrays defining the simplices.

    Returns:
        True if x is within the feasible domain, False otherwise.
    """

    # Check non-negativity constraint
    if np.any(x < 0):
        return False

    # Check that each simplex sums to 1 (within numerical tolerance)
    if not all(np.isclose(np.sum(x[idx]), 1, atol=1e-12) for idx in partitions):
        return False

    return True
