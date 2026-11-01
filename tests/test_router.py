import numpy as np

from data.data_loader import load
from router import candidate_order, route, split

def test_split():
    datasets = load().queries["dataset_id"].to_numpy()
    train, cal, test = split(datasets, seed=0)
    assert (len(train), len(cal), len(test)) == (7464, 2490, 2492)

    together = np.sort(np.concatenate([train, cal, test]))
    assert np.array_equal(together, np.arange(len(datasets)))

    for part in (train, cal, test):
        assert set(datasets[part]) == set(datasets)

    again, _, _ = split(datasets, seed=0)
    other, _, _ = split(datasets, seed=1)
    assert np.array_equal(train, again)
    assert not np.array_equal(train, other)

def test_candidate_order():
    cost = np.array([[3.0, 1.0, 5.0, 2.0],
                     [3.0, np.nan, 5.0, 2.0],
                     [0.0, 9.0, 5.0, 2.0]])
    assert candidate_order(cost, np.array([0, 1]), reference=2) == [1, 3, 0]

def test_route():
    scores = np.array([[0.9, 0.8, 0.5],
                       [0.2, 0.7, 0.5],
                       [0.9, 0.3, 0.5]])
    available = np.array([[True, True, True],
                          [True, True, True],
                          [False, True, True]])
    order, reference = [0, 1], 2
    assert route(scores, available, order, reference, 1.0).tolist() == [2, 2, 2]
    assert route(scores, available, order, reference, 0.0).tolist() == [0, 0, 1]
    assert route(scores, available, order, reference, 0.6).tolist() == [0, 1, 2]