import numpy as np

def start_point(n, partitions):
    """
    Generates a valid initial feasible point.

    Parameters:
    - n (int): Number of decision variables.
    - partitions (list of ndarrays): List of index arrays defining the simplices.

    Returns:
    - x0 (ndarray): Feasible initial point satisfying simplex constraints.
    """

    # Input validation
    if not isinstance(n, int) or n <= 0:
      raise ValueError("n must be a positive integer.")

    x0 = np.zeros(n)

    # Equal distribution within each simplex
    for part in partitions:
        val = 1.0 / len(part)
        x0[part] = val

    return x0
