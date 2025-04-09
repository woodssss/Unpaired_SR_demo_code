import numpy as np
import ot  # POT library
from scipy.stats import gaussian_kde


def compute_w2(a, b):
    """
    Computes the 2-Wasserstein distance between two empirical distributions
    a and b. Each of a and b is assumed to be an array of flattened samples.

    Parameters:
    -----------
    a : np.ndarray, shape [n, d]
        The first set of samples (n samples of dimension d).
    b : np.ndarray, shape [m, d]
        The second set of samples (m samples of dimension d).

    Returns:
    --------
    float
        The 2-Wasserstein distance between the empirical distributions.
    """
    # Number of samples
    n = a.shape[0]
    m = b.shape[0]

    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)

    # Define uniform weights for each distribution
    mu = np.ones(n) / n  # weights for a
    nu = np.ones(m) / m  # weights for b

    # Compute pairwise distance matrix (squared Euclidean distance)
    cost_matrix = ot.dist(a, b, metric='euclidean') ** 2

    # Solve the linear program for the optimal transport
    # (we can use ot.emd since we have exact uniform marginals)
    gamma = ot.emd(mu, nu, cost_matrix)

    # The 2-Wasserstein distance is the square root of the
    # sum of the cost_matrix * optimal coupling.
    w2_dist_sq = np.mean(gamma * cost_matrix)
    w2_dist = np.sqrt(w2_dist_sq)

    return w2_dist

def compute_KL(a, b):
    # Flatten spatial dimensions (keep samples as rows)
    a_flat = a.reshape(a.shape[0], -1)  # Shape [100, 1024]
    b_flat = b.reshape(b.shape[0], -1)  # Shape [400, 1024]

    kl_divergences = []

    for i in range(a_flat.shape[1]):
        # Extract pixel values for the i-th feature
        a_pixel = a_flat[:, i]
        b_pixel = b_flat[:, i]

        # Estimate densities using Kernel Density Estimation (KDE)
        kde_a = gaussian_kde(a_pixel)
        kde_b = gaussian_kde(b_pixel)

        # Compute KL divergence: mean(log(p_a / p_b)) over samples from `a`
        log_p_a = kde_a.logpdf(a_pixel)
        log_p_b = kde_b.logpdf(a_pixel)
        kl = np.mean(log_p_a - log_p_b)

        kl_divergences.append(kl)

    total_kl = np.mean(kl_divergences)
    return total_kl