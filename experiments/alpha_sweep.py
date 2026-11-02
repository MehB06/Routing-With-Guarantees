from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import load
from experiments.common import ALPHAS, DELTA, SEED, choose, evaluate, fit, reference_losses
from ltt import calibrate
from router import GRID, load_embeddings

RESULTS_DIR = Path(__file__).parent.parent / "results" / "alpha_sweep"

def sweep() -> pd.DataFrame:
    """Calibrate on the calibration rows and evaluate on the test rows, once per alpha."""
    data = load()
    embeddings = load_embeddings(data.queries["prompt"].tolist())
    train, cal, test, scores, order = fit(data, embeddings, SEED)
    losses = reference_losses(data, scores, order, cal)
    rows = []
    for alpha in ALPHAS:
        k = calibrate(losses, alpha, DELTA)
        choice = choose(data, scores, order, test, GRID[k])
        shares = np.bincount(choice, minlength=len(data.models)) / len(choice)
        rows.append({"alpha": alpha, "lambda_hat": GRID[k], **evaluate(data, test, choice), **dict(zip(data.models, shares))})
    return pd.DataFrame(rows)

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = sweep()
    table.to_csv(RESULTS_DIR / "sweep.csv", index=False)
    columns = ["alpha", "lambda_hat", "risk", "conditional_risk", "accuracy", "mean_cost", "saving", "routed"]
    headline = table[table["alpha"].isin([0.05, 0.10, 0.15, 0.20, 0.25])]
    print(headline[columns].round(3).to_string(index=False))

if __name__ == "__main__":
    main()