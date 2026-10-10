import json
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parent / "raw" / "bench-release"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
PARQUET_PATH = PROCESSED_DIR / "data.parquet"

DATASET_SPLITS = {
    "aime": "hybrid",
    "livemathbench": "test",
    "livecodebench": "test",
    "swe-bench": "verified",
    "gpqa": "test",
    "hle": "test",
    "mmlupro": "test_3000",
    "simpleqa": "test",
    "arenahard": "test",
    "tau2": "test",
}

MODELS = [
    "claude-sonnet-4", "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5-chat", "gpt-5",
    "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507", "deepseek-v3-0324",
    "deepseek-v3.1-terminus", "deepseek-r1-0528", "glm-4.6", "kimi-k2-0905", "intern-s1",
]

def load_json(file_path: Path) -> dict:
    """Open and parse one result file. Raise if it is unreadable."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read JSON file {file_path}: {e}") from e

def is_performance_cost(data: dict) -> bool:
    """True if a result file belongs to the performance-cost track and is not a demo run."""
    return (
        DATASET_SPLITS.get(data["dataset_name"]) == data["split"]
        and data["model_name"] in MODELS
        and not data.get("demo", False)
    )

def select_files(raw_data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Return the path of every performance-cost result file. Raise on duplicate runs."""
    selected, seen = [], set()
    for path in sorted(raw_data_dir.rglob("*.json")):
        data = load_json(path)
        if is_performance_cost(data):
            key = (data["dataset_name"], data["model_name"])
            if key in seen:
                raise RuntimeError(f"Duplicate run for {key}: {path}")
            seen.add(key)
            selected.append(path)
    return selected

def read_file(file_path: Path) -> list[dict]:
    """Return one row per record in a result file."""
    data = load_json(file_path)
    rows = []
    for record in data["records"]:
        prompt = record["prompt"]
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt, ensure_ascii=False)
        rows.append({
            "dataset_id": data["dataset_name"],
            "record_index": record["index"],
            "model": data["model_name"],
            "prompt": prompt,
            "score": record.get("score"),
            "cost": record["cost"],
        })
    return rows

def build_table(paths: list[Path]) -> pd.DataFrame:
    """Stack the records of every selected file into one table, one row per (query, model)."""
    rows = []
    for path in paths:
        rows.extend(read_file(path))
    table = pd.DataFrame(rows)

    key = ["dataset_id", "record_index", "model"]
    if table.duplicated(subset=key).any():
        raise RuntimeError("The same (dataset_id, record_index, model) appears more than once")

    table["score"] = table["score"].astype(float)
    table["cost"] = table["cost"].astype(float)
    return table.sort_values(key).reset_index(drop=True)

def main() -> None:
    """Build data/processed/data.parquet from the raw release."""
    paths = select_files()
    table = build_table(paths)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(PARQUET_PATH, index=False)

    n_queries = len(table[["dataset_id", "record_index"]].drop_duplicates())
    print(f"Saved {PARQUET_PATH.name}: {len(table)} rows, {n_queries} queries, {len(paths)} files")

if __name__ == "__main__":
    main()