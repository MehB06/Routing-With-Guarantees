from pathlib import Path

import pandas as pd

from data.data_loader import load
from experiments.common import run_split
from router import load_embeddings

RESULTS_DIR = Path(__file__).parent.parent / "results" / "repeated_trials"
SEEDS = range(100)

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
    """Per target alpha and method: mean and spread of the test risk, share of splits over the target, share routed and cost saving."""
    table = table[table["alpha"].isin([0.05, 0.10, 0.15, 0.20])]
    table = table.assign(over_target=table["risk"] > table["alpha"])
    columns = {"risk_mean": ("risk", "mean"), "risk_sd": ("risk", "std"), "over_target": ("over_target", "mean"),
               "routed": ("routed", "mean"), "saving": ("saving", "mean")}
    return table.groupby(["alpha", "method"]).agg(**columns).round(3).reset_index()

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = trials()
    table.to_csv(RESULTS_DIR / "trials.csv", index=False)
    table = summary(table)
    table.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()