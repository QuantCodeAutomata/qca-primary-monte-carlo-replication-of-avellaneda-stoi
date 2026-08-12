"""Generic Monte Carlo simulator for the Avellaneda-Stoikov market-making model.

The simulator decouples the quoting strategy from the market mechanics,
allowing Strategy A (inventory-aware) and Strategy B (symmetric) to share
the same simulation engine.
"""
from __future__ import annotations
from typing import Callable, Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from src.config import SimParams

# Type alias: strategy function maps (S, q, t, gamma, params) -> (delta_a, delta_b)
StrategyFn = Callable[[float, int, float, float, SimParams], Tuple[float, float]]


def simulate_path(
    strategy_fn: StrategyFn,
    gamma: float,
    params: SimParams,
    rng: np.random.Generator,
    record_path: bool = False,
) -> Dict[str, Any]:
    """Simulate a single market-making path.

    Parameters
    ----------
    strategy_fn : callable
        Function (S, q, t, gamma, params) -> (delta_a, delta_b).
        delta_a is the ask distance from mid (p_a = S + delta_a).
        delta_b is the bid distance from mid (p_b = S - delta_b).
    gamma : float
        Risk-aversion coefficient for this scenario.
    params : SimParams
        Simulation parameters.
    rng : numpy.random.Generator
        Random number generator (passed in for reproducibility).
    record_path : bool
        If True, store the full time-series path data.

    Returns
    -------
    dict with keys:
        terminal_profit  : float   Pi_T = X_T + q_T * S_T
        terminal_inventory : int   q_T
        negative_delta_count : int  # of steps with negative delta (warning indicator)
        bernoulli_violation_count : int  # of steps where lambda*dt > 1
        path_data : list of dicts (only if record_path=True)
    """
    S = params.S0
    q = params.q0
    X = params.X0
    dt = params.dt
    sigma = params.sigma
    A = params.A
    kk = params.k
    N = params.N

    negative_delta_count = 0
    bernoulli_violation_count = 0
    path_data: List[Dict[str, Any]] = []

    sigma_sqrt_dt = sigma * np.sqrt(dt)

    for n in range(N):
        t = n * dt

        # --- Strategy: compute quote distances ---
        delta_a, delta_b = strategy_fn(S, q, t, gamma, params)

        # Track negative deltas (quotes cross mid-price)
        if delta_a < 0 or delta_b < 0:
            negative_delta_count += 1

        # --- Intensities and execution probabilities ---
        lam_a = A * np.exp(-kk * delta_a)
        lam_b = A * np.exp(-kk * delta_b)
        prob_a = lam_a * dt
        prob_b = lam_b * dt

        if prob_a > 1.0 or prob_b > 1.0:
            bernoulli_violation_count += 1

        # --- Order execution (independent Bernoulli draws) ---
        p_a = S + delta_a
        p_b = S - delta_b
        ask_fill = rng.uniform() < prob_a
        bid_fill = rng.uniform() < prob_b

        # Record reservation price for inventory strategies
        r = S - q * gamma * params.sigma ** 2 * (params.T - t)

        if record_path:
            path_data.append({
                'step': n,
                'time': t,
                'S': S,
                'r': r,
                'p_a': p_a,
                'p_b': p_b,
                'delta_a': delta_a,
                'delta_b': delta_b,
                'q': q,
                'X': X,
                'lam_a': lam_a,
                'lam_b': lam_b,
                'ask_fill': int(ask_fill),
                'bid_fill': int(bid_fill),
            })

        # --- Fill updates (pre-price-move prices) ---
        if ask_fill:
            q -= 1
            X += p_a
        if bid_fill:
            q += 1
            X -= p_b

        # --- Mid-price update: binomial ± sigma*sqrt(dt) ---
        price_move = sigma_sqrt_dt if rng.uniform() < 0.5 else -sigma_sqrt_dt
        S += price_move

    # --- Terminal profit ---
    terminal_profit = X + q * S

    result: Dict[str, Any] = {
        'terminal_profit': terminal_profit,
        'terminal_inventory': q,
        'terminal_S': S,
        'terminal_X': X,
        'negative_delta_count': negative_delta_count,
        'bernoulli_violation_count': bernoulli_violation_count,
    }
    if record_path:
        result['path_data'] = path_data

    return result


def simulate_monte_carlo(
    strategy_fn: StrategyFn,
    strategy_name: str,
    gamma: float,
    params: SimParams,
    seed: int,
    rep_path_index: int = 0,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run Monte Carlo simulation for a given strategy and gamma.

    Parameters
    ----------
    strategy_fn : StrategyFn
        Strategy function (S, q, t, gamma, params) -> (delta_a, delta_b).
    strategy_name : str
        Label for the strategy (used in output columns).
    gamma : float
        Risk-aversion scenario.
    params : SimParams
        Simulation parameters.
    seed : int
        Random seed for this run.
    rep_path_index : int
        Which path index to record as the representative path.

    Returns
    -------
    results_df : pd.DataFrame
        One row per path with columns: path_id, gamma, strategy,
        terminal_profit, terminal_inventory, negative_delta_count,
        bernoulli_violation_count.
    rep_path_df : pd.DataFrame or None
        Full time-series for the representative path.
    """
    rng = np.random.default_rng(seed)
    records = []
    rep_path_df = None

    for m in range(params.n_paths):
        record_this = (m == rep_path_index)
        res = simulate_path(
            strategy_fn=strategy_fn,
            gamma=gamma,
            params=params,
            rng=rng,
            record_path=record_this,
        )
        row = {
            'path_id': m,
            'gamma': gamma,
            'strategy': strategy_name,
            'terminal_profit': res['terminal_profit'],
            'terminal_inventory': res['terminal_inventory'],
            'negative_delta_count': res['negative_delta_count'],
            'bernoulli_violation_count': res['bernoulli_violation_count'],
        }
        records.append(row)
        if record_this:
            rep_path_df = pd.DataFrame(res['path_data'])
            rep_path_df['gamma'] = gamma
            rep_path_df['strategy'] = strategy_name

    results_df = pd.DataFrame(records)
    return results_df, rep_path_df
