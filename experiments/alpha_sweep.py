from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.data_loader import load
from experiments.common import ALPHAS, DELTA, SEED, choose, evaluate, fit, reference_losses
from ltt import calibrate
from router import GRID, load_embeddings, REFERENCE

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
    singles = pd.DataFrame([{"model": name, **evaluate(data, test, np.full(len(test), j))} for j, name in enumerate(data.models)])
    return pd.DataFrame(rows), singles

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

def panel(ax, table: pd.DataFrame, column: str, reference_value: float, ylabel: str, color: str, marker: str) -> None:
    """One panel: the router's column against alpha, with the reference model's value as a dashed line."""
    ax.axhline(reference_value, linestyle="--", color="#8a8984", linewidth=1, label=f"Always {REFERENCE}")
    ax.plot(table["alpha"], table[column], marker + "-", color=color, linewidth=2, markersize=4, label="Router")
    ax.set_xlabel("Target α")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 0.31)
    ax.set_ylim(0, 1.1 * max(reference_value, table[column].max()))
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

def plot_vs_reference(table: pd.DataFrame, singles: pd.DataFrame, path: Path) -> None:
    """Router accuracy (left) and mean cost (right) against alpha, next to always using the reference."""
    reference = singles.set_index("model").loc[REFERENCE]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    panel(left, table, "accuracy", reference["accuracy"], "Test accuracy", "#2a78d6", "o")
    panel(right, table, "mean_cost", reference["mean_cost"], "Mean cost per query (USD)", "#eb6834", "s")
    fig.suptitle(f"Router vs always using {REFERENCE}, across α")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_plane(table: pd.DataFrame, singles: pd.DataFrame, path: Path) -> None:
    """Accuracy against mean cost: every single model on the test rows, and the router's curve over alpha."""
    plt.figure(figsize=(11, 6))
    plt.scatter(singles["mean_cost"], singles["accuracy"], color="#52514e", s=20, label="Single models (test set)")
    for _, row in singles.iterrows():
        plt.annotate(row["model"], (row["mean_cost"], row["accuracy"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color="#52514e")
    plt.plot(table["mean_cost"], table["accuracy"], "o-", color="#2a78d6", linewidth=2, markersize=4, label="Router (one point per α; α grows from right to left)")
    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    plt.xscale("log")
    plt.xticks(ticks, [f"${t}" for t in ticks])
    plt.minorticks_off()
    plt.xlabel("Mean cost per query (USD, log scale)")
    plt.ylabel("Test accuracy")
    plt.title("Router vs single models: accuracy and cost")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_traffic(table: pd.DataFrame, models: list[str], path: Path) -> None:
    """Share of test queries sent to each model against alpha, stacked; models under 5% grouped as other."""
    main = [m for m in models if m != REFERENCE and table[m].max() >= 0.05]
    other = [m for m in models if m != REFERENCE and m not in main]
    shares = [table[REFERENCE]] + [table[m] for m in main] + [table[other].sum(axis=1)]
    labels = [f"{REFERENCE} (reference)"] + main + ["Other models"]
    colors = ["#8a8984"] + ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"][:len(main)] + ["#c3c2b7"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(table["alpha"], [100 * s for s in shares], labels=labels, colors=colors, edgecolor="white", linewidth=0.5)
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], names[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xlim(table["alpha"].min(), table["alpha"].max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Target α")
    ax.set_ylabel("Share of test queries (%)")
    ax.set_title("Where the router sends test queries, across α")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table, singles = sweep()
    table.to_csv(RESULTS_DIR / "sweep.csv", index=False)
    singles.to_csv(RESULTS_DIR / "singles.csv", index=False)
    columns = ["alpha", "lambda_hat", "risk", "conditional_risk", "accuracy", "mean_cost", "saving", "routed"]
    headline = table[table["alpha"].isin([0.05, 0.10, 0.15, 0.20, 0.25])]
    print(headline[columns].round(3).to_string(index=False))
    plot_risk(table, RESULTS_DIR / "risk.png")
    plot_vs_reference(table, singles, RESULTS_DIR / "vs_reference.png")
    plot_plane(table, singles, RESULTS_DIR / "plane.png")
    plot_traffic(table, singles["model"].tolist(), RESULTS_DIR / "traffic.png")
        
if __name__ == "__main__":
    main()