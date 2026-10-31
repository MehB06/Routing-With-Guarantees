from scipy.stats import binom

def binomial_pvalue(failures: int, n: int, alpha: float) -> float:
    """"Probability of at most 'failures' failures in n queries if the true risk is alpha."""
    return binom.cdf(failures, n, alpha)