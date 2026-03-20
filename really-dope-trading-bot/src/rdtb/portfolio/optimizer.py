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
    defensive_gross_exposure: float = 0.20
    min_gross_exposure: float = 0.03
    risk_exit_threshold: float = 0.72
    atr_stop_multiple: float = 2.5
    regime_transition_slope: float = 8.0


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
        defensive_gross_exposure=config.optimizer_defensive_gross_exposure,
        min_gross_exposure=config.optimizer_min_gross_exposure,
        risk_exit_threshold=config.risk_exit_threshold,
        atr_stop_multiple=config.atr_stop_multiple,
        regime_transition_slope=config.regime_transition_slope,
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

    fallback_regime_probability = _aggregate_regime_probability(frame, column="regime_probability")

    frame["adjusted_score"] = (
        frame["utility_score"].fillna(0.0)
        - min(policy.risk_penalty * 0.45, 0.30) * frame["risk_probability"].fillna(0.0)
        - (1.0 - frame["regime_probability"].fillna(fallback_regime_probability)) * 0.08
    )
    frame.loc[frame["risk_probability"] >= policy.risk_reject_threshold, "adjusted_score"] = -1.0
    holdings_symbols = holdings["symbol"].tolist()
    investable_floor = max(policy.buy_threshold - config.hold_buffer, 0.45)
    investable = frame.loc[
        (frame["utility_score"].fillna(0.0) >= investable_floor)
        & (frame["risk_probability"].fillna(1.0) < policy.risk_reject_threshold)
    ].copy()
    top_symbols = investable.head(max(policy.max_positions * 2, policy.max_positions))["symbol"].tolist()
    candidate_symbols = []
    for symbol in holdings_symbols + top_symbols:
        if symbol not in candidate_symbols:
            candidate_symbols.append(symbol)
        if len(candidate_symbols) >= max(policy.max_positions * 2, policy.max_positions + len(holdings_symbols)):
            break
    candidate = frame.loc[frame["symbol"].isin(candidate_symbols)].copy().reset_index(drop=True)
    all_symbols = frame[["symbol", "close", "utility_score", "risk_probability", "alpha_prediction", "regime_probability"]].copy()
    for column in ["regime_anchor_probability", "regime_participation_probability"]:
        if column in frame.columns:
            all_symbols[column] = pd.to_numeric(frame[column], errors="coerce")
    all_symbols["atr_pct"] = frame["atr_pct"].astype(float) if "atr_pct" in frame.columns else np.nan
    all_symbols["current_weight"] = all_symbols["symbol"].map(current_weights).fillna(0.0)
    all_symbols["target_weight"] = 0.0
    if candidate.empty:
        return all_symbols

    gross_exposure = _resolve_gross_exposure(
        policy,
        _aggregate_regime_probability(candidate, column="regime_anchor_probability"),
    )
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


def _aggregate_regime_probability(frame: pd.DataFrame, column: str = "regime_probability") -> float:
    if frame.empty:
        return 0.5
    effective_column = column if column in frame.columns else "regime_probability"
    if effective_column not in frame.columns:
        return 0.5
    regime = pd.to_numeric(frame[effective_column], errors="coerce")
    valid = regime.notna()
    if not valid.any():
        return 0.5
    weights = pd.Series(1.0, index=frame.index, dtype=float)
    if "utility_score" in frame.columns:
        weights = pd.to_numeric(frame["utility_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
    regime = regime.loc[valid]
    weights = weights.loc[valid]
    if float(weights.sum()) > 0:
        return float(np.average(regime.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))
    return float(regime.median())


def _resolve_gross_exposure(policy: PolicyParameters, regime_probability: float) -> float:
    risk_on_exposure = max(1.0 - policy.cash_floor, 0.0)
    defensive_exposure = np.clip(policy.defensive_gross_exposure, 0.0, risk_on_exposure)
    span = policy.risk_on_threshold - policy.defensive_threshold
    if span <= 0:
        blend = float(regime_probability >= policy.risk_on_threshold)
    else:
        midpoint = policy.defensive_threshold + (span * 0.5)
        slope = max(float(policy.regime_transition_slope), 1e-3) / span
        blend = 1.0 / (1.0 + np.exp(-(regime_probability - midpoint) * slope))
    return defensive_exposure + (risk_on_exposure - defensive_exposure) * blend


def _solve_target_weights(
    candidate: pd.DataFrame,
    current_weights: np.ndarray,
    gross_exposure: float,
    policy: PolicyParameters,
    optimizer_backend: str,
) -> np.ndarray:
    if gross_exposure <= 0:
        return np.zeros(len(candidate), dtype=float)
    adjusted_scores = candidate["adjusted_score"].to_numpy(dtype=float)
    if adjusted_scores.size == 0 or float(np.nanmax(adjusted_scores)) <= 0:
        return np.zeros(len(candidate), dtype=float)
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
    opportunity_strength = np.clip((opportunity_score - policy.buy_threshold) / 0.25, 0.0, 1.0)
    min_required_exposure = min(gross_exposure, policy.min_gross_exposure) * opportunity_strength
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
