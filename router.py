import numpy as np

def split(datasets: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices for train, calibration and test (60/20/20), shuffled within each dataset"""
    rng = np.random.default_rng(seed)
    train, cal, test = [], [], []
    for name in np.unique(datasets):
        rows = np.flatnonzero(datasets == name)
        rng.shuffle(rows)
        a, b = int(0.6 * len(rows)), int(0.8 * len(rows))
        train.append(rows[:a])
        cal.append(rows[a:b])
        test.append(rows[b:])
    return np.concatenate(train), np.concatenate(cal), np.concatenate(test)

def candidate_order(cost: np.ndarray, train: np.ndarray, reference: int) -> list[int]:
    """Model indices other than the reference, cheapest first by mean cost on the training rows"""
    mean_cost = np.nanmean(cost[train], axis=0)
    return [int(j) for j in np.argsort(mean_cost) if j != reference]