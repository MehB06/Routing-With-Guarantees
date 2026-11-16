import numpy as np
import pandas as pd

from data.data_loader import RoutingData
from experiments.common import reference_share_losses, uncalibrated
from ltt import calibrate
from router import REFERENCE

def test_uncalibrated():
    losses = np.zeros((1000, 4), dtype=int)
    losses[:50, 1] = 1
    losses[:98, 2] = 1
    losses[:200, 3] = 1
    assert uncalibrated(losses, 0.1) == 2
    assert calibrate(losses, 0.1, 0.1) == 1

    losses[:, 0] = 1
    assert uncalibrated(losses, 0.1) is None

def test_reference_share_losses():
    scores = np.array([[0.9, 0.8, 0.5],
                       [0.2, 0.7, 0.5],
                       [0.9, 0.3, 0.5]])
    data = RoutingData(pd.DataFrame({"dataset_id": ["a"] * 3}), ["cheap", "mid", REFERENCE], np.ones((3, 3), dtype=bool), np.ones((3, 3)))
    losses = reference_share_losses(data, scores, [0, 1], np.arange(3))
    assert losses.shape == (3, 101)
    assert not losses[:, 0].any()
    assert losses[:, -1].all()
    assert losses[:, 85].tolist() == [False, True, False]