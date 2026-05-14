"""
Diagnostic Replication — Spread-Ambiguity Variant (exp_2).

Addresses the paper's internal inconsistency between the theoretically implied
time-varying spread and the constant spread values reported in the published tables.

The paper's tables report spread = 2/gamma * ln(1 + gamma/k), whereas the
finite-horizon formula implies an additional term gamma*sigma^2*(T-t).

This module uses the constant-spread interpretation to attempt table matching.

Custom implementation — Context7 found no library equivalent for this diagnostic.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from src.avellaneda_stoikov import (
    ASParams,
    reservation_price,
    optimal_half_spread_base,
    execution_intensity,
)


# ---------------------------------------------------------------------------
# Constant-spread quoting formulas
# ---------------------------------------------------------------------------

def constant_spread_quote_distances(
    q: int, gamma: float, sigma: float, k: float, tau: float
) -> Tuple[float, float]:
    """
    Diagnostic variant: constant total spread = 2*c_gamma, centered around
    the reservation price.

    delta^a = c_gamma - q*gamma*sigma^2*(T-t)
    delta^b = c_gamma + q*gamma*sigma^2*(T-t)

    This preserves centering around the reservation price while forcing the
    total spread to equal 2*c_gamma (matching the paper's reported table values).

    Parameters
    ----------
    q     : current inventory
    gamma : risk-aversion coefficient
    sigma : volatility
    k     : intensity decay parameter
    tau   : time remaining (T - t)

    Returns
    -------
    (delta_a, delta_b) : ask and bid quote distances from mid-price
    """
    c = optimal_half_spread_base(gamma, k)
    adj = q * gamma * sigma ** 2 * tau
    delta_a = c - adj
    delta_b = c + adj
    return delta_a, delta_b


def constant_spread_symmetric_quotes(
    s: float, gamma: float, k: float
) -> Tuple[float, float]:
    """
    Symmetric benchmark with constant spread = 2*c_gamma centered at mid-price.

    p^a = s + c_gamma
    p^b = s - c_gamma

    Parameters
    ----------
    s     : current mid-price
    gamma : risk-aversion coefficient
    k     : intensity decay parameter

    Returns
    -------
    (p_ask, p_bid) : symmetric ask and bid quotes
    """
    c = optimal_half_spread_base(gamma, k)
    return s + c, s - c


# ---------------------------------------------------------------------------
# Single-path simulation (diagnostic variant)
# ---------------------------------------------------------------------------

def simulate_path_diagnostic(
    params: ASParams,
    strategy: str,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate one Monte Carlo path using the constant-spread (table-matching)
    interpretation.

    Parameters
    ----------
    params   : ASParams instance
    strategy : 'inventory' or 'symmetric'
    rng      : numpy random Generator

    Returns
    -------
    dict with terminal_profit and terminal_inventory
    """
    assert strategy in ("inventory", "symmetric"), \
        f"Unknown strategy '{strategy}'."

    n = params.n_steps
    gamma = params.gamma
    sigma = params.sigma
    k = params.k
    A = params.A
    dt = params.dt
    T = params.T

    s = params.s0
    x = params.x0
    q = params.q0

    for i in range(n):
        tau = T - i * dt

        if strategy == "inventory":
            delta_a, delta_b = constant_spread_quote_distances(q, gamma, sigma, k, tau)
            p_ask = s + delta_a
            p_bid = s - delta_b
        else:  # symmetric
            p_ask, p_bid = constant_spread_symmetric_quotes(s, gamma, k)
            delta_a = p_ask - s
            delta_b = s - p_bid

        lam_a = execution_intensity(delta_a, A, k)
        lam_b = execution_intensity(delta_b, A, k)

        ask_fill = int(rng.random() < lam_a * dt)
        bid_fill = int(rng.random() < lam_b * dt)

        x = x + ask_fill * p_ask - bid_fill * p_bid
        q = q - ask_fill + bid_fill

        price_move = sigma * np.sqrt(dt) * (1 if rng.random() < 0.5 else -1)
        s = s + price_move

    terminal_profit = x + q * s
    return {
        "terminal_profit": terminal_profit,
        "terminal_inventory": q,
    }


# ---------------------------------------------------------------------------
# Monte Carlo runner (diagnostic variant)
# ---------------------------------------------------------------------------

def run_monte_carlo_diagnostic(
    params: ASParams,
    strategy: str,
) -> dict:
    """
    Run Monte Carlo simulation for the diagnostic (constant-spread) variant.

    Parameters
    ----------
    params   : ASParams instance
    strategy : 'inventory' or 'symmetric'

    Returns
    -------
    dict with summary statistics
    """
    rng = np.random.default_rng(params.seed)

    profits = np.zeros(params.n_paths)
    inventories = np.zeros(params.n_paths, dtype=int)

    for i in range(params.n_paths):
        result = simulate_path_diagnostic(params, strategy, rng)
        profits[i] = result["terminal_profit"]
        inventories[i] = result["terminal_inventory"]

    return {
        "profits": profits,
        "inventories": inventories,
        "mean_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "mean_inventory": float(np.mean(inventories)),
        "std_inventory": float(np.std(inventories)),
    }
