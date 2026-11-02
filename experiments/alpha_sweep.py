from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_risk(table: pd.DataFrame, path: Path) -> None:
    """Test risk against the target alpha, with the line risk = alpha."""
    plt.figure(figsize=(6, 5))
    plt.plot([0, 0.31], [0, 0.31], "--", color="#8a8984", linewidth=1, label="risk = α")
    plt.plot(table["alpha"], table["risk"], "o-", color="#2a78d6", linewidth=2, markersize=4, label="Marginal risk (guaranteed ≤ α - all queries)")
    plt.plot(table["alpha"], table["conditional_risk"], "s-", color="#eb6834", linewidth=2, markersize=4, alpha=0.6, label="Conditional risk (not guaranteed - routed queries)")
    plt.xlim(0, 0.31)
    plt.ylim(0, 0.31)
    plt.xlabel("Target α")
    plt.ylabel("Reference regret on the test set")
    plt.title(f"Risk control across α (δ = {DELTA})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = sweep()
    table.to_csv(RESULTS_DIR / "sweep.csv", index=False)
    columns = ["alpha", "lambda_hat", "risk", "conditional_risk", "accuracy", "mean_cost", "saving", "routed"]
    headline = table[table["alpha"].isin([0.05, 0.10, 0.15, 0.20, 0.25])]
    print(headline[columns].round(3).to_string(index=False))
    plot_risk(table, RESULTS_DIR / "risk.png")

if __name__ == "__main__":
    main()