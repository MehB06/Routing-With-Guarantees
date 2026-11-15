import numpy as np

from data.data_loader import RoutingData
from router import GRID, REFERENCE, candidate_order, fit_scorer, loss_matrix, route, split
from ltt import calibrate, fixed_sequence_test

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

def uncalibrated(losses: np.ndarray, alpha: float) -> int:
    """Index of the last lambda before the calibration risk exceeds alpha: LTT walk without the test"""
    risks = losses.mean(axis=0)
    return fixed_sequence_test(risks, alpha) - 1

def run_split(data: RoutingData, embeddings: np.ndarray, seed: int) -> list[dict]:
    """One split: choose lambda with LTT and with the uncalibrated baseline on a random sample of the held-out queries, then measure the result on all held-out queries"""
    train, cal, test, scores, order = fit(data, embeddings, seed)
    held_out = np.concatenate([cal, test])                                   
    sample = np.random.default_rng(seed).choice(held_out, size=len(cal))     
    losses = reference_losses(data, scores, order, sample)                   
    rows = []
    for alpha in ALPHAS:
        indices = {"ltt": calibrate(losses, alpha, DELTA), "uncalibrated": uncalibrated(losses, alpha)}
        for method, k in indices.items():
            lam = GRID[0] if k is None else GRID[k]
            choice = choose(data, scores, order, held_out, lam)             
            rows.append({"seed": seed, "alpha": alpha, "method": method, "lambda_hat": lam, **evaluate(data, held_out, choice)})   
    return rows