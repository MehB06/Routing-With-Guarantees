import numpy as np
from data.data_processing import PROCESSED_DIR
from sklearn.neighbors import NearestNeighbors

REFERENCE = "gemini-2.5-pro"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
GRID = np.round(np.linspace(1, 0, 101), 2)

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

def route(scores: np.ndarray, available: np.ndarray, order: list[int], reference: int, lam: float) -> np.ndarray:
    """Chosen model per query: the first available candidate in order with score >= lam, else the reference model"""
    choice = np.full(len(scores), reference)
    undecided = np.ones(len(scores), dtype=bool)
    for j in order:
        accept = undecided & available[:, j] & (scores[:, j] >= lam)
        choice[accept] = j
        undecided &= ~accept
    return choice

def loss_matrix(scores: np.ndarray, available: np.ndarray, order: list[int], reference: int, correct: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Loss per query (rows) and lambda in GRID (columns): the routed model was wrong where target was right"""
    rows = np.arange(len(scores))
    columns = []
    for lam in GRID:
        choice = route(scores, available, order, reference, lam)
        columns.append(~correct[rows, choice] & target)
    return np.column_stack(columns)

def load_embeddings(prompts: list[str]) -> np.ndarray:
    """MiniLM embedding of every prompt (one row each), computed once and cached to disk"""
    if EMBEDDINGS_PATH.exists():
        return np.load(EMBEDDINGS_PATH)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(prompts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    return embeddings

def fit_scorer(embeddings: np.ndarray, correct: np.ndarray, available: np.ndarray, train: np.ndarray, k: int = 40) -> np.ndarray:
    """Score per query and model: the share of its k nearest training queries that the model got right"""
    knn = NearestNeighbors(n_neighbors=k).fit(embeddings[train])
    _, idx = knn.kneighbors(embeddings)
    neighbours = train[idx]
    right = correct[neighbours].sum(axis=1)
    seen = available[neighbours].sum(axis=1)
    return right / np.maximum(seen, 1)