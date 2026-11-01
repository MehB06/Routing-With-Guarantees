import numpy as np

from data.data_loader import load
from router import candidate_order, route, split, loss_matrix, GRID, fit_scorer

SCORES = np.array([[0.9, 0.8, 0.5],
                   [0.2, 0.7, 0.5],
                   [0.9, 0.3, 0.5]])
AVAILABLE = np.array([[True, True, True],
                      [True, True, True],
                      [False, True, True]])

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
    assert route(SCORES, AVAILABLE, [0, 1], 2, 1.0).tolist() == [2, 2, 2]
    assert route(SCORES, AVAILABLE, [0, 1], 2, 0.0).tolist() == [0, 0, 1]
    assert route(SCORES, AVAILABLE, [0, 1], 2, 0.6).tolist() == [0, 1, 2]

def test_loss_matrix():
    correct = np.array([[False, True, True],
                        [True, False, True],
                        [False, False, False]])
    losses = loss_matrix(SCORES, AVAILABLE, [0, 1], 2, correct, correct[:, 2])
    assert losses.shape == (3, 101)
    assert not losses[:, 0].any()
    assert GRID[40] == 0.6
    assert losses[:, 40].tolist() == [True, True, False]
    assert losses[:, -1].tolist() == [True, False, False]

def test_fit_scorer():
    embeddings = np.array([[0.0], [0.1], [1.0], [1.1], [0.05]])
    correct = np.array([[True, True],
                        [True, False],
                        [False, True],
                        [False, True],
                        [False, False]])
    available = np.ones((5, 2), dtype=bool)
    available[1, 1] = False
    scores = fit_scorer(embeddings, correct, available, np.array([0, 1, 2, 3]), k=2)
    assert scores[4].tolist() == [1.0, 1.0]
    assert scores[2].tolist() == [0.0, 1.0]