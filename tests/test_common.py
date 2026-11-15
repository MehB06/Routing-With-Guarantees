import numpy as np

from experiments.common import uncalibrated
from ltt import calibrate

def test_uncalibrated():
    losses = np.zeros((1000, 4), dtype=int)
    losses[:50, 1] = 1
    losses[:98, 2] = 1
    losses[:200, 3] = 1
    assert uncalibrated(losses, 0.1) == 2
    assert calibrate(losses, 0.1, 0.1) == 1