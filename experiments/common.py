import numpy as np

from data.data_loader import RoutingData
from router import REFERENCE, candidate_order, fit_scorer, loss_matrix, route, split

ALPHAS = np.round(np.linspace(0.01, 0.30, 30), 2)
DELTA = 0.10
SEED = 0 

def fit(data: RoutingData, embeddings: np.ndarray, seed: int) -> tuple:
    """Split the queries, then fit the scorer and the candidate order on the training rows"""
    train, cal, test = split(data.queries["dataset_id"].to_numpy(), seed)
    scores = fit_scorer(embeddings, data.correct, ~np.isnan(data.cost), train)
    order = candidate_order(data.cost, train, data.models.index(REFERENCE))
    return train, cal, test, scores, order

def reference_losses(data: RoutingData, scores: np.ndarray, order: list[int], rows: np.ndarray) -> np.ndarray:
    """Reference-regret loss matrix on the given rows, one column per lambda in GRID"""
    ref = data.models.index(REFERENCE)
    available = ~np.isnan(data.cost)
    return loss_matrix(scores[rows], available[rows], order, ref, data.correct[rows], data.correct[rows, ref])

def choose(data: RoutingData, scores: np.ndarray, order: list[int], rows: np.ndarray, lam: float) -> np.ndarray:
    """Chosen model for each of the given rows at one lambda """
    ref = data.models.index(REFERENCE)
    available = ~np.isnan(data.cost)
    return route(scores[rows], available[rows], order, ref, lam)

def evaluate(data: RoutingData, rows: np.ndarray, choice: np.ndarray) -> dict:
    """Risk, conditional risk, accuracy, mean cost, saving and routed fraction on the given rows"""
    ref = data.models.index(REFERENCE)
    right = data.correct[rows, choice]
    loss = ~right & data.correct[rows, ref]
    routed = choice != ref
    mean_cost = np.nanmean(data.cost[rows, choice])
    return {
        "risk": loss.mean(),
        "conditional_risk": loss[routed].mean() if routed.any() else np.nan,
        "accuracy": right.mean(),
        "mean_cost": mean_cost,
        "saving": 1 - mean_cost / data.cost[rows, ref].mean(),
        "routed": routed.mean(),
    }