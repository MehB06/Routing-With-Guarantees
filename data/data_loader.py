from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_processing import MODELS, PARQUET_PATH, PROCESSED_DIR

CORRECT_THRESHOLD = 0.5
QUERY_KEY = ["dataset_id", "record_index"]
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"

@dataclass(frozen=True)
class RoutingData:
    """Everything the router needs, one row per query and one column per model (MODELS order)."""
    queries: pd.DataFrame
    models: list[str]
    correct: np.ndarray
    cost: np.ndarray

def load(parquet_path: Path = PARQUET_PATH) -> RoutingData:
    """Load data.parquet as query x model matrices for routing."""
    table = pd.read_parquet(parquet_path)
    queries = table.drop_duplicates(subset=QUERY_KEY)[QUERY_KEY + ["prompt"]].reset_index(drop=True)
    index = pd.MultiIndex.from_frame(queries[QUERY_KEY])

    def as_matrix(column: str) -> np.ndarray:
        wide = table.pivot(index=QUERY_KEY, columns="model", values=column)
        return wide.reindex(index=index, columns=MODELS).to_numpy(dtype=float)

    correct = as_matrix("score") >= CORRECT_THRESHOLD
    cost = as_matrix("cost")
    return RoutingData(queries, MODELS, correct, cost)

def load_embeddings(prompts: list[str]) -> np.ndarray:
    """MiniLM embedding of every prompt (one row each), computed once and cached to disk"""
    if EMBEDDINGS_PATH.exists():
        return np.load(EMBEDDINGS_PATH)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(prompts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    return embeddings