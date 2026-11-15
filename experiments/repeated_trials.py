from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from data.data_loader import load
from experiments.common import run_split
from router import REFERENCE, load_embeddings
from experiments.single_models import model_table

RESULTS_DIR = Path(__file__).parent.parent / "results" / "repeated_trials"
SEEDS = range(100)
LABELLED = [REFERENCE, "gpt-5", "qwen3-235b-a22b-2507", "deepseek-v3-0324"]

def trials() -> pd.DataFrame:
    """Run every seed and stack the rows: one row per seed, alpha and method."""
    data = load()
    embeddings = load_embeddings(data.queries["prompt"].tolist())
    rows = []
    for seed in SEEDS:
        rows.extend(run_split(data, embeddings, seed))
        print(f"seed {seed} done")
    return pd.DataFrame(rows)

def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Number of splits, out of all seeds, where the chosen router's risk is over the target alpha, per method."""
    table = table[table["alpha"].isin([0.05, 0.10, 0.15, 0.20])]
    table = table.assign(over_target=table["risk"] > table["alpha"])
    return table.pivot_table(index="alpha", columns="method", values="over_target", aggfunc="sum").astype(int).reset_index()

def plot_tradeoff(table: pd.DataFrame, singles: pd.DataFrame, path: Path) -> None:
    """Regret against cost: the LTT router (one point per alpha, mean over the splits) and every single model."""
    router = table[table["method"] == "ltt"].groupby("alpha")[["risk", "saving"]].mean()
    reference_cost = singles.set_index("model").loc[REFERENCE, "mean_cost"]
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(router["risk"], 100 * (1 - router["saving"]), "o-", color="#2a78d6", linewidth=2, markersize=4, label="Router with LTT (one point per target α)")
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        plt.annotate(f"α = {alpha:.2f}", (router["risk"][alpha], 100 * (1 - router["saving"][alpha])), textcoords="offset points", xytext=(8, 6), fontsize=8, color="#2a78d6")
    plt.scatter(singles["reference_regret"], 100 * singles["mean_cost"] / reference_cost, color="#52514e", s=22, zorder=3, label="Single models")
    for _, row in singles[singles["model"].isin(LABELLED)].iterrows():
        plt.annotate(row["model"], (row["reference_regret"], 100 * row["mean_cost"] / reference_cost), textcoords="offset points", xytext=(6, -10), fontsize=7.5, color="#52514e")
    plt.xlabel(f"Regret against {REFERENCE} (share of queries)")
    plt.ylabel(f"Cost (% of always using {REFERENCE})")
    plt.title("Regret against cost")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = trials()
    table.to_csv(RESULTS_DIR / "trials.csv", index=False)
    counts = summary(table)
    counts.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(counts.to_string(index=False))
    plot_tradeoff(table, model_table(), RESULTS_DIR / "tradeoff.png")

if __name__ == "__main__":
    main()