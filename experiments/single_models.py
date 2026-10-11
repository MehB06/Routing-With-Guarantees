from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load

RESULTS_DIR = Path(__file__).parent.parent / "results" / "single_models"

def model_table() -> pd.DataFrame:
    """Accuracy and mean cost per query of each model, cheapest first."""
    data = load()
    table = pd.DataFrame({
        "model": data.models,
        "accuracy": data.correct.mean(axis=0),
        "mean_cost": np.nanmean(data.cost, axis=0),
    })
    return table.sort_values("mean_cost").reset_index(drop=True)

def plot(table: pd.DataFrame, path: Path) -> None:
    """Scatter plot of accuracy vs mean cost per query, one labelled point per model."""
    plt.figure(figsize=(8, 5))
    plt.scatter(table["mean_cost"], table["accuracy"])
    for _, row in table.iterrows():
        plt.annotate(row["model"], (row["mean_cost"], row["accuracy"]), fontsize=8)
    plt.xscale("log")
    plt.xticks([0.001, 0.002, 0.005, 0.01, 0.02, 0.05], ["$0.001", "$0.002", "$0.005", "$0.01", "$0.02", "$0.05"])
    plt.minorticks_off()
    plt.xlabel("Mean cost per query (USD, log scale)")
    plt.ylabel("Accuracy")
    plt.title("Single model performance")
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=200, bbox_inches="tight")

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = model_table()
    print(table)
    table.to_csv(RESULTS_DIR / "table.csv", index=False)
    plot(table, RESULTS_DIR / "single_models.png")

if __name__ == "__main__":
    main()