import numpy as np
from scipy.stats import binom

def binomial_pvalue(failures: int, n: int, alpha: float) -> float:
    """Probability of at most 'failures' failures in n queries if the true risk is alpha."""
    return binom.cdf(failures, n, alpha)

def fixed_sequence_test(pvalues: np.ndarray, delta: float) -> int:
    """Number of leading p-values at most delta, stopping at the first p-value greater than delta."""
    count = 0 
    for p in pvalues:
        if p > delta:
            break
        count += 1
    return count

def calibrate(losses: np.ndarray, alpha: float, delta: float) -> int | None:
    """Index of the last certified lambda (we order the columns from safest to riskiest), or None if no lambda is certified."""
    n = losses.shape[0]
    failures = losses.sum(axis=0)
    pvalues = binomial_pvalue(failures, n, alpha)
    k = fixed_sequence_test(pvalues, delta)
    return k-1 if k > 0 else None