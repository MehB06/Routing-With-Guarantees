import numpy as np
from ltt import binomial_pvalue, fixed_sequence_test

def test_binomial_pvalue():
    assert abs(binomial_pvalue(220, 1000, 0.25) - 0.0147) < 1e-3
    assert abs(binomial_pvalue(240, 1000, 0.25) - 0.2449) < 1e-3

def test_fixed_sequence_test():
    assert fixed_sequence_test(np.array([0.01, 0.02, 0.50, 0.01]), 0.1) == 2
    assert fixed_sequence_test(np.array([0.50, 0.01]), 0.1) == 0
    assert fixed_sequence_test(np.array([0.01, 0.02]), 0.1) == 2