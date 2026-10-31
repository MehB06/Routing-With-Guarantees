from ltt import binomial_pvalue

def test_binomial_pvalue():
    assert abs(binomial_pvalue(220, 1000, 0.25) - 0.0147) < 1e-3
    assert abs(binomial_pvalue(240, 1000, 0.25) - 0.2449) < 1e-3