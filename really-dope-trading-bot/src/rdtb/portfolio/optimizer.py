from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdtb.config import TradingBotConfig


@dataclass(slots=True)
class PolicyParameters:
    max_positions: int
    max_weight: float
    risk_penalty: float
    turnover_penalty: float
    concentration_penalty: float
    cash_floor: float
    risk_on_threshold: float
    defensive_threshold: float
    buy_threshold: float
    add_threshold: float
    exit_threshold: float
    trim_threshold: float
    risk_reject_threshold: float


@dataclass(slots=True)
class PortfolioSnapshot:
    date: pd.Timestamp
    market_frame: pd.DataFrame
    holdings: pd.DataFrame
    cash: float


def default_policy(config: TradingBotConfig) -> PolicyParameters:
    return PolicyParameters(
        max_positions=config.max_positions,
        max_weight=config.max_weight,
        risk_penalty=config.optimizer_risk_penalty,
        turnover_penalty=config.optimizer_turnover_penalty,
        concentration_penalty=config.optimizer_concentration_penalty,
        cash_floor=config.optimizer_cash_floor,
        risk_on_threshold=config.regime_risk_on_threshold,
        defensive_threshold=config.regime_defensive_threshold,
        buy_threshold=config.buy_score_threshold,
        add_threshold=config.add_score_threshold,
        exit_threshold=config.exit_score_threshold,
        trim_threshold=config.trim_score_threshold,
        risk_reject_threshold=config.risk_reject_threshold,
    )


def optimize_target_weights(
    snapshot: PortfolioSnapshot,
    config: TradingBotConfig,
    policy: PolicyParameters | None = None,
) -> pd.DataFrame:
    policy = policy or default_policy(config)
    frame = snapshot.market_frame.copy().sort_values("utility_score", ascending=False).reset_index(drop=True)
    holdings = snapshot.holdings.copy()
    if holdings.empty:
        holdings = pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "buy_date"])
    holdings["symbol"] = holdings["symbol"].astype(str)
    reference_prices = frame.set_index("symbol")["close"].astype(float)
    holdings["reference_price"] = holdings["symbol"].map(reference_prices).fillna(holdings.get("avg_cost", 0.0))
    holdings["market_value"] = holdings["quantity"].fillna(0.0) * holdings["reference_price"].fillna(0.0)
    current_positions_value = float(holdings["market_value"].sum())
    equity = max(snapshot.cash + current_positions_value, 1.0)
    current_weights = (holdings.set_index("symbol")["market_value"] / equity).to_dict()

    regime_probability = float(frame["regime_probability"].dropna().iloc[0]) if not frame["regime_probability"].dropna().empty else 0.5
    gross_exposure = _resolve_gross_exposure(policy, regime_probability)

    frame["adjusted_score"] = frame["utility_score"].fillna(0.0) - min(policy.risk_penalty * 0.35, 0.20) * frame["risk_probability"].fillna(0.0)
    frame.loc[frame["risk_probability"] >= policy.risk_reject_threshold, "adjusted_score"] = -1.0
    holdings_symbols = holdings["symbol"].tolist()
    investable_floor = max(policy.buy_threshold - config.hold_buffer, 0.45)
    investable = frame.loc[
        (frame["utility_score"].fillna(0.0) >= investable_floor)
        & (frame["risk_probability"].fillna(1.0) < policy.risk_reject_threshold)
    ].copy()
    if investable.empty:
        investable = frame.head(max(policy.max_positions * 2, policy.max_positions)).copy()
    top_symbols = investable.head(max(policy.max_positions * 2, policy.max_positions))["symbol"].tolist()
    candidate_symbols = []
    for symbol in top_symbols + holdings_symbols:
        if symbol not in candidate_symbols:
            candidate_symbols.append(symbol)
        if len(candidate_symbols) >= max(policy.max_positions * 2, policy.max_positions + len(holdings_symbols)):
            break
    candidate = frame.loc[frame["symbol"].isin(candidate_symbols)].copy().reset_index(drop=True)
    all_symbols = frame[["symbol", "close", "utility_score", "risk_probability", "alpha_prediction", "regime_probability"]].copy()
    all_symbols["current_weight"] = all_symbols["symbol"].map(current_weights).fillna(0.0)
    all_symbols["target_weight"] = 0.0
    if candidate.empty:
        return all_symbols

    optimized = _solve_target_weights(
        candidate=candidate,
        current_weights=np.array([current_weights.get(symbol, 0.0) for symbol in candidate["symbol"]], dtype=float),
        gross_exposure=gross_exposure,
        policy=policy,
        optimizer_backend=config.optimizer_backend,
    )
    target_weights = dict(zip(candidate["symbol"], optimized))
    all_symbols["target_weight"] = all_symbols["symbol"].map(target_weights).fillna(0.0)
    return all_symbols.sort_values(["target_weight", "utility_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)


def _resolve_gross_exposure(policy: PolicyParameters, regime_probability: float) -> float:
    if regime_probability >= policy.risk_on_threshold:
        return 1.0 - policy.cash_floor
    if regime_probability <= policy.defensive_threshold:
        return max(0.35, 0.55 - policy.cash_floor)
    span = policy.risk_on_threshold - policy.defensive_threshold
    blend = 0.0 if span <= 0 else (regime_probability - policy.defensive_threshold) / span
    return (0.55 + 0.35 * blend) - policy.cash_floor


def _solve_target_weights(
    candidate: pd.DataFrame,
    current_weights: np.ndarray,
    gross_exposure: float,
    policy: PolicyParameters,
    optimizer_backend: str,
) -> np.ndarray:
    adjusted_scores = candidate["adjusted_score"].to_numpy(dtype=float)
    conviction_scores = adjusted_scores - adjusted_scores.min() + 1e-3
    if np.all(conviction_scores <= 0):
        return np.zeros(len(candidate), dtype=float)
    conviction_scores = conviction_scores / max(float(conviction_scores.max()), 1e-6)
    if optimizer_backend != "convex":
        breadth_multiplier = 3 if optimizer_backend == "soft_heuristic" else 1
        return _heuristic_weights(
            conviction_scores,
            gross_exposure,
            policy.max_weight,
            policy.max_positions,
            breadth_multiplier=breadth_multiplier,
        )
    try:
        import cvxpy as cp
    except Exception:  # pragma: no cover - optional runtime dependency
        return _heuristic_weights(conviction_scores, gross_exposure, policy.max_weight, policy.max_positions)

    top_count = min(len(candidate), max(policy.max_positions, 1))
    opportunity_score = float(candidate["utility_score"].nlargest(top_count).mean())
    min_required_exposure = gross_exposure * np.clip((opportunity_score - 0.45) / 0.25, 0.0, 0.65)
    max_feasible_exposure = min(gross_exposure, policy.max_positions * policy.max_weight)
    min_required_exposure = min(min_required_exposure, max_feasible_exposure * 0.95)

    weights = cp.Variable(len(candidate))
    objective = cp.Maximize(
        conviction_scores @ weights
        - (policy.turnover_penalty * 0.20) * cp.norm1(weights - current_weights)
        - policy.concentration_penalty * cp.sum_squares(weights)
    )
    constraints = [
        weights >= 0,
        weights <= policy.max_weight,
        cp.sum(weights) <= gross_exposure,
    ]
    if min_required_exposure > 0:
        constraints.append(cp.sum(weights) >= min_required_exposure)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="OSQP", warm_start=True, verbose=False)
    except Exception:  # pragma: no cover - optional runtime dependency
        return _heuristic_weights(conviction_scores, gross_exposure, policy.max_weight, policy.max_positions)
    if weights.value is None:
        return _heuristic_weights(conviction_scores, gross_exposure, policy.max_weight, policy.max_positions)
    solved = np.clip(np.asarray(weights.value, dtype=float), 0.0, policy.max_weight)
    if len(solved) > policy.max_positions:
        keep_indices = np.argsort(solved)[-policy.max_positions:]
        mask = np.zeros(len(solved), dtype=bool)
        mask[keep_indices] = True
        solved = np.where(mask, solved, 0.0)
    if solved.sum() > gross_exposure and solved.sum() > 0:
        solved *= gross_exposure / solved.sum()
    return solved


def _heuristic_weights(
    scores: np.ndarray,
    gross_exposure: float,
    max_weight: float,
    max_positions: int,
    breadth_multiplier: int = 1,
) -> np.ndarray:
    normalized = scores / scores.sum()
    weights = np.minimum(normalized * gross_exposure, max_weight)
    max_slots = max(max_positions * max(breadth_multiplier, 1), max_positions)
    if len(weights) > max_slots:
        keep_indices = np.argsort(weights)[-max_slots:]
        mask = np.zeros(len(weights), dtype=bool)
        mask[keep_indices] = True
        weights = np.where(mask, weights, 0.0)
    if weights.sum() == 0:
        return weights
    return weights * (gross_exposure / weights.sum())
