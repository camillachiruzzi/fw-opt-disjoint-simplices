from .start_point import start_point
import numpy as np

def generate_data(n, K, spectral_radius=10, lambda_min=1, density=1.0, kernel_dim=0,
                  q_generation='random', seed=42, violated_constraint='all'):
    """
    Generates a structured quadratic optimization problem on disjoint simplices.

    Parameters:
    - n (int): Number of decision variables.
    - K (int): Number of disjoint simplices.
    - spectral_radius (float): Maximum eigenvalue of Q.
    - lambda_min (float): Minimum strictly positive eigenvalue of Q.
    - density (float): Density of the matrix Q (between 0 and 1, where 1 means a full matrix).
    - kernel_dim (int): the dimension of the null space of Q.
    - q_generation (str): The strategy adopted for generating q ('random', 'internal', 'external').
    - seed (int): Random seed for reproducibility.
    - violated_constraint (str): Determines which constraints are violated ('sum_simplex', 'non_negative', 'all').

    Returns:
    - Q (ndarray): Positive semi-definite symmetric matrix of shape (n, n).
    - q (ndarray): Coefficient vector of shape (n,).
    - partitions (list of ndarrays): List of index arrays defining the simplices.
    """

     # Input validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0 < lambda_min <= spectral_radius):
        raise ValueError("lambda_min must be > 0 and ≤ spectral_radius.")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be a float between 0 and 1.")
    if not (0 <= kernel_dim <= n):
        raise ValueError("kernel_dim must be an integer between 0 and n.")
    if not isinstance(K, int) or K <= 0 or K > n:
        raise ValueError("K must be a positive integer and ≤ n.")
  
    np.random.seed(seed)

    # Generate a positive semi-definite matrix Q with controlled eigenvalues
    eigenvalues = np.zeros(n)
    eigenvalues[:n - kernel_dim] = np.linspace(spectral_radius, lambda_min, n - kernel_dim)
    U, _ = np.linalg.qr(np.random.randn(n, n)) # Generate random orthonormal basis
    Q = U @ np.diag(eigenvalues) @ U.T # Construct Q with desired eigenvalues

    # Apply sparsity if density < 1
    if density < 1.0:
        mask = np.random.rand(n, n) > density
        Q[mask] = 0 # Make sparse
        Q = (Q + Q.T) / 2 # Ensure symmetry

    # Create random partitions
    indices = np.random.permutation(n)  # Shuffle indices randomly
    partitions = np.array_split(indices, K) # Split into K groups

    # Generate an initial feasible point
    x = start_point(n, partitions)

    # Generate vector q based on the selected strategy
    if q_generation == 'random':
        q = np.random.randn(n) # Generate random vector
        q /= np.linalg.norm(q)

        return Q, q, partitions

    else:

        if q_generation == 'internal':
            x = np.zeros(n)
            for part in partitions:
                simplex_vals = np.random.rand(len(part))
                simplex_vals /= np.sum(simplex_vals)  # Normalize to sum to 1 within each simplex
                x[part] = simplex_vals

        elif q_generation == 'external':
            # Modify x so that the sum in each simplex is no longer 1
            if violated_constraint == 'sum_simplex':
                for part in partitions:
                    x[part] *= np.random.uniform(0.5, 1.5)

            elif violated_constraint == 'non_negative':
                # Introduce negative values in x while ensuring sum remains 1
                for part in partitions:
                    k = max(1, len(part) // 10) # Select fraction of elements to modify (10%)
                    neg_idx = np.random.choice(part, size=k, replace=False) # Random indices to make negative
                    delta = np.random.uniform(0.1, 1.0, size=k)
                    x[neg_idx] = -delta
                   
                    # Correct the remaining values to ensure sum remains 1
                    total = np.sum(x[part])
                    correction = (1.0 - total)
                    pos_mask = x[part] >= 0
                    if np.any(pos_mask):
                        x[part[pos_mask]] +=  correction / np.sum(pos_mask)

            elif violated_constraint == 'all':
                # Apply both non-negative and sum violations
                for part in partitions:
                    k = max(1, len(part) // 10)
                    neg_idx = np.random.choice(part, size=k, replace=False)
                    delta = np.random.uniform(0.1, 1.0, size=k)
                    x[neg_idx] = -delta
                    x[part] *= np.random.uniform(0.5, 1.5)

        else:
            raise ValueError("Invalid q_generation")

        # Compute q
        q = -2 * Q @ x

        return Q, q, partitions
