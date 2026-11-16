from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from data.data_loader import load
from experiments.common import control_cost, control_regret, setup
from router import REFERENCE, load_embeddings
from experiments.single_models import model_table

RESULTS_DIR = Path(__file__).parent.parent / "results" / "repeated_trials"
SEEDS = range(100)
LABELLED = [REFERENCE, "gpt-5", "qwen3-235b-a22b-2507", "deepseek-v3-0324"]

def trials() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every seed on both sides: one row per seed, target and method, for regret control and for cost control."""
    data = load()
    embeddings = load_embeddings(data.queries["prompt"].tolist())
    regret, cost = [], []
    for seed in SEEDS:
        parts = setup(data, embeddings, seed)
        regret.extend(control_regret(data, seed, *parts))
        cost.extend(control_cost(data, seed, *parts))
        print(f"seed {seed} done")
    return pd.DataFrame(regret), pd.DataFrame(cost)

def summary(table: pd.DataFrame, target: str, measured: pd.Series, levels: list[float]) -> pd.DataFrame:
    """Number of splits where the measured value is over its target, per method, at the given target levels."""
    table = table.assign(over_target=measured > table[target])
    table = table[table[target].isin(levels)]
    return table.pivot_table(index=target, columns="method", values="over_target", aggfunc="sum").astype(int).reset_index()

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

def panel_regret(ax, regret: pd.DataFrame) -> None:
    """Control regret: mean cost of the LTT router against the regret target alpha."""
    saving = regret[regret["method"] == "ltt"].groupby("alpha")["saving"].mean()
    ax.plot(saving.index, 100 * (1 - saving), "o-", color="#2a78d6", linewidth=2, markersize=4, label="Router with LTT")
    ax.plot([0, saving.index[0]], [100, 100 * (1 - saving.iloc[0])], "--", color="#2a78d6", linewidth=1.5)
    ax.plot(0, 100, "o", color="#52514e", markersize=5, label=f"Always {REFERENCE}")
    ax.set_xlabel("Regret target α (fixed)")
    ax.set_ylabel(f"Cost (% of always using {REFERENCE})")
    ax.set_title("Control regret, minimise cost")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)

def panel_cost(ax, cost: pd.DataFrame) -> None:
    """Control cost: mean regret of the LTT router against the budget on the share of queries sent to the reference."""
    risk = cost[cost["method"] == "ltt"].groupby("budget")["risk"].mean()
    ax.plot(100 * risk.index, risk, "s-", color="#bb2018", linewidth=2, markersize=4, label="Router with LTT")
    ax.plot([100 * risk.index[-1], 100], [risk.iloc[-1], 0], "--", color="#bb2018", linewidth=1.5)
    ax.plot(100, 0, "o", color="#52514e", markersize=5, label=f"Always {REFERENCE}")
    ax.set_xlabel(f"Budget: share of queries sent to {REFERENCE}, % (fixed)")
    ax.set_ylabel(f"Regret against {REFERENCE}")
    ax.set_title("Control cost, minimise regret")
    ax.set_ylim(0, 0.25)
    ax.legend()
    ax.grid(alpha=0.3)

def plot_control(regret: pd.DataFrame, cost: pd.DataFrame, path: Path) -> None:
    """What the router delivers when one objective is fixed and the other is minimised."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    panel_regret(left, regret)
    panel_cost(right, cost)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    regret, cost = trials()
    regret.to_csv(RESULTS_DIR / "trials.csv", index=False)
    cost.to_csv(RESULTS_DIR / "trials_cost.csv", index=False)
    over_alpha = summary(regret, "alpha", regret["risk"], [0.05, 0.10, 0.15, 0.20])
    over_budget = summary(cost, "budget", 1 - cost["routed"], [0.2, 0.4, 0.6, 0.8])
    over_alpha.to_csv(RESULTS_DIR / "summary.csv", index=False)
    over_budget.to_csv(RESULTS_DIR / "summary_cost.csv", index=False)
    print(over_alpha.to_string(index=False))
    print(over_budget.to_string(index=False))
    plot_tradeoff(regret, model_table(), RESULTS_DIR / "tradeoff.png")
    plot_control(regret, cost, RESULTS_DIR / "control.png")

if __name__ == "__main__":
    main()