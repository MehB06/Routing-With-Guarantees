import numpy as np

from data.data_loader import load
from router import split

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