import numpy as np

from data.data_loader import RoutingData
from router import GRID, REFERENCE, candidate_order, fit_scorer, loss_matrix, route, split
from ltt import calibrate

ALPHAS = np.round(np.linspace(0.01, 0.30, 30), 2)
BUDGETS = np.round(np.linspace(0.1, 0.9, 9), 1)
DELTA = 0.10

def fit(data: RoutingData, embeddings: np.ndarray, seed: int) -> tuple:
    """Split the queries, fit the scorer and the candidate order on the training rows, and return the other rows with them"""
    train, cal, test = split(data.queries["dataset_id"].to_numpy(), seed)
    scores = fit_scorer(embeddings, data.correct, ~np.isnan(data.cost), train)
    order = candidate_order(data.cost, train, data.models.index(REFERENCE))
    return cal, test, scores, order

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

def reference_share_losses(data: RoutingData, scores: np.ndarray, order: list[int], rows: np.ndarray) -> np.ndarray:
    """Cost loss matrix on the given rows: True where the query goes to the reference, one column per lambda from 0.00 up to 1.00"""
    ref = data.models.index(REFERENCE)
    return np.column_stack([choose(data, scores, order, rows, lam) == ref for lam in GRID[::-1]])

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

def uncalibrated(losses: np.ndarray, alpha: float) -> int | None:
    """Index of the last lambda before the calibration risk exceeds alpha (LTT walk without the test), or None if the first one already does"""
    passed = losses.mean(axis=0) <= alpha
    k = len(passed) if passed.all() else int(np.argmin(passed))
    return k - 1 if k > 0 else None

def setup(data: RoutingData, embeddings: np.ndarray, seed: int) -> tuple:
    """Fit on train, pool the rest as the held-out queries, and draw the calibration sample from them"""
    cal, test, scores, order = fit(data, embeddings, seed)
    held_out = np.concatenate([cal, test])
    sample = np.random.default_rng(seed).choice(held_out, size=len(cal))
    return held_out, sample, scores, order

def control_regret(data: RoutingData, seed: int, held_out: np.ndarray, sample: np.ndarray, scores: np.ndarray, order: list[int]) -> list[dict]:
    """Control regret: for every target alpha, choose lambda on the sample with each method, then measure on all held-out queries"""
    losses = reference_losses(data, scores, order, sample)
    rows = []
    for alpha in ALPHAS:
        indices = {"ltt": calibrate(losses, alpha, DELTA), "uncalibrated": uncalibrated(losses, alpha)}
        for method, k in indices.items():
            lam = GRID[0] if k is None else GRID[k]
            choice = choose(data, scores, order, held_out, lam)
            rows.append({"seed": seed, "alpha": alpha, "method": method, "lambda_hat": lam, **evaluate(data, held_out, choice)})
    return rows

def control_cost(data: RoutingData, seed: int, held_out: np.ndarray, sample: np.ndarray, scores: np.ndarray, order: list[int]) -> list[dict]:
    """Control cost: for every budget on the share of queries sent to the reference, choose lambda on the sample with each method, then measure on all held-out queries"""
    losses = reference_share_losses(data, scores, order, sample)
    grid = GRID[::-1]
    rows = []
    for budget in BUDGETS:
        indices = {"ltt": calibrate(losses, budget, DELTA), "uncalibrated": uncalibrated(losses, budget)}
        for method, k in indices.items():
            lam = grid[0] if k is None else grid[k]
            choice = choose(data, scores, order, held_out, lam)
            rows.append({"seed": seed, "budget": budget, "method": method, "lambda_hat": lam, **evaluate(data, held_out, choice)})
    return rows