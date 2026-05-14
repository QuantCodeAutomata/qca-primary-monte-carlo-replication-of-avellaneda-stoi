"""
Avellaneda-Stoikov (2008) Market Making Model — Core Implementation.

Implements the finite-horizon approximation for inventory-based market making
under arithmetic Brownian motion and exponential execution intensities.

Reference:
    Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book.
    Quantitative Finance, 8(3), 217-224.

Custom implementation — Context7 found no library equivalent for the AS quoting formulas.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ASParams:
    """Parameters for the Avellaneda-Stoikov model."""
    s0: float = 100.0       # initial mid-price
    sigma: float = 2.0      # volatility (arithmetic BM)
    T: float = 1.0          # horizon
    dt: float = 0.005       # time step  → 200 steps
    q0: int = 0             # initial inventory
    x0: float = 0.0         # initial cash
    A: float = 140.0        # intensity scale
    k: float = 1.5          # intensity decay
    gamma: float = 0.1      # risk aversion
    n_paths: int = 1000     # Monte Carlo paths
    seed: int = 42          # random seed

    @property
    def n_steps(self) -> int:
        return round(self.T / self.dt)


# ---------------------------------------------------------------------------
# Quoting formulas — finite-horizon (primary replication, exp_1)
# ---------------------------------------------------------------------------

def reservation_price(s: float, q: int, gamma: float, sigma: float,
                      tau: float) -> float:
    """
    Reservation (indifference) price under CARA utility.

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    gamma : risk-aversion coefficient
    sigma : volatility
    tau   : time remaining (T - t)

    Returns
    -------
    float : reservation price
    """
    return s - q * gamma * sigma ** 2 * tau


def optimal_half_spread_base(gamma: float, k: float) -> float:
    """
    Constant component of the optimal half-spread.

    c = (1/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    gamma : risk-aversion coefficient
    k     : intensity decay parameter

    Returns
    -------
    float : constant half-spread component
    """
    return (1.0 / gamma) * np.log(1.0 + gamma / k)


def finite_horizon_quote_distances(
    q: int, gamma: float, sigma: float, k: float, tau: float
) -> Tuple[float, float]:
    """
    Finite-horizon optimal quote distances from the mid-price.

    delta^a = (1/gamma)*ln(1+gamma/k) + ((1-2q)*gamma*sigma^2*(T-t))/2
    delta^b = (1/gamma)*ln(1+gamma/k) + ((1+2q)*gamma*sigma^2*(T-t))/2

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
    adj = gamma * sigma ** 2 * tau
    delta_a = c + (1 - 2 * q) * adj / 2.0
    delta_b = c + (1 + 2 * q) * adj / 2.0
    return delta_a, delta_b


def symmetric_finite_horizon_quotes(
    s: float, gamma: float, sigma: float, k: float, tau: float
) -> Tuple[float, float]:
    """
    Symmetric benchmark quotes centered at the mid-price.

    spread_t = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)
    p^a = s + spread_t/2
    p^b = s - spread_t/2

    Parameters
    ----------
    s     : current mid-price
    gamma : risk-aversion coefficient
    sigma : volatility
    k     : intensity decay parameter
    tau   : time remaining (T - t)

    Returns
    -------
    (p_ask, p_bid) : symmetric ask and bid quotes
    """
    spread = gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
    half = spread / 2.0
    return s + half, s - half


def execution_intensity(delta: float, A: float, k: float) -> float:
    """
    Exponential execution intensity.

    lambda(delta) = A * exp(-k * delta)

    Parameters
    ----------
    delta : quote distance from mid-price
    A     : intensity scale
    k     : intensity decay

    Returns
    -------
    float : execution intensity (events per unit time)
    """
    return A * np.exp(-k * delta)


# ---------------------------------------------------------------------------
# Single-path simulation
# ---------------------------------------------------------------------------

def simulate_path(
    params: ASParams,
    strategy: str,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate one Monte Carlo path of the AS market-making model.

    Parameters
    ----------
    params   : ASParams instance
    strategy : 'inventory' or 'symmetric'
    rng      : numpy random Generator (for reproducibility)

    Returns
    -------
    dict with keys:
        time, s, x, q, r, p_ask, p_bid, ask_fill, bid_fill,
        terminal_profit, terminal_inventory
    """
    assert strategy in ("inventory", "symmetric"), \
        f"Unknown strategy '{strategy}'. Choose 'inventory' or 'symmetric'."

    n = params.n_steps
    gamma = params.gamma
    sigma = params.sigma
    k = params.k
    A = params.A
    dt = params.dt
    T = params.T

    # Pre-allocate state arrays
    time_arr = np.zeros(n + 1)
    s_arr = np.zeros(n + 1)
    x_arr = np.zeros(n + 1)
    q_arr = np.zeros(n + 1, dtype=int)
    r_arr = np.zeros(n + 1)
    p_ask_arr = np.zeros(n + 1)
    p_bid_arr = np.zeros(n + 1)
    ask_fill_arr = np.zeros(n, dtype=int)
    bid_fill_arr = np.zeros(n, dtype=int)

    # Initial state
    s_arr[0] = params.s0
    x_arr[0] = params.x0
    q_arr[0] = params.q0

    for i in range(n):
        t = i * dt
        tau = T - t
        s = s_arr[i]
        x = x_arr[i]
        q = q_arr[i]

        time_arr[i] = t

        if strategy == "inventory":
            # Reservation price
            r = reservation_price(s, q, gamma, sigma, tau)
            delta_a, delta_b = finite_horizon_quote_distances(q, gamma, sigma, k, tau)
            # Methodological deviation: clip negative quote distances to 0 when
            # extreme inventory causes the formula to produce negative values.
            # This is numerically necessary for large |q|; the paper's main
            # simulation uses q0=0 so this case does not arise in primary results.
            delta_a = max(delta_a, 0.0)
            delta_b = max(delta_b, 0.0)
            p_ask = s + delta_a
            p_bid = s - delta_b
        else:  # symmetric
            r = s  # symmetric strategy centers at mid-price
            p_ask, p_bid = symmetric_finite_horizon_quotes(s, gamma, sigma, k, tau)
            delta_a = p_ask - s
            delta_b = s - p_bid

        r_arr[i] = r
        p_ask_arr[i] = p_ask
        p_bid_arr[i] = p_bid

        # Execution intensities
        lam_a = execution_intensity(delta_a, A, k)
        lam_b = execution_intensity(delta_b, A, k)

        # Validate lambda*dt < 1 (Bernoulli approximation validity)
        assert lam_a * dt <= 1.0, \
            f"lambda_a*dt={lam_a*dt:.4f} >= 1; Bernoulli approx invalid."
        assert lam_b * dt <= 1.0, \
            f"lambda_b*dt={lam_b*dt:.4f} >= 1; Bernoulli approx invalid."

        # Bernoulli execution events (independent bid and ask fills)
        ask_fill = int(rng.random() < lam_a * dt)
        bid_fill = int(rng.random() < lam_b * dt)

        ask_fill_arr[i] = ask_fill
        bid_fill_arr[i] = bid_fill

        # Update cash and inventory
        x_new = x + ask_fill * p_ask - bid_fill * p_bid
        q_new = q - ask_fill + bid_fill

        # Mid-price binomial update: ±sigma*sqrt(dt) with equal probability
        price_move = sigma * np.sqrt(dt) * (1 if rng.random() < 0.5 else -1)
        s_new = s + price_move

        s_arr[i + 1] = s_new
        x_arr[i + 1] = x_new
        q_arr[i + 1] = q_new

    # Terminal step
    time_arr[n] = T
    r_arr[n] = s_arr[n]  # tau=0 → r=s
    p_ask_arr[n] = s_arr[n]
    p_bid_arr[n] = s_arr[n]

    terminal_profit = x_arr[n] + q_arr[n] * s_arr[n]

    return {
        "time": time_arr,
        "s": s_arr,
        "x": x_arr,
        "q": q_arr,
        "r": r_arr,
        "p_ask": p_ask_arr,
        "p_bid": p_bid_arr,
        "ask_fill": ask_fill_arr,
        "bid_fill": bid_fill_arr,
        "terminal_profit": terminal_profit,
        "terminal_inventory": int(q_arr[n]),
    }


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    params: ASParams,
    strategy: str,
) -> dict:
    """
    Run Monte Carlo simulation for a given strategy.

    Parameters
    ----------
    params   : ASParams instance
    strategy : 'inventory' or 'symmetric'

    Returns
    -------
    dict with keys:
        profits       : array of terminal profits (n_paths,)
        inventories   : array of terminal inventories (n_paths,)
        mean_profit   : float
        std_profit    : float
        mean_inventory: float
        std_inventory : float
        sample_path   : dict from simulate_path for path 0 (for plotting)
    """
    rng = np.random.default_rng(params.seed)

    profits = np.zeros(params.n_paths)
    inventories = np.zeros(params.n_paths, dtype=int)
    sample_path = None

    for i in range(params.n_paths):
        result = simulate_path(params, strategy, rng)
        profits[i] = result["terminal_profit"]
        inventories[i] = result["terminal_inventory"]
        if i == 0:
            sample_path = result

    return {
        "profits": profits,
        "inventories": inventories,
        "mean_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "mean_inventory": float(np.mean(inventories)),
        "std_inventory": float(np.std(inventories)),
        "sample_path": sample_path,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_reservation_price_properties(
    params: ASParams,
) -> None:
    """
    Validate key properties of the reservation price:
    1. r(s, 0, t) = s  (zero inventory → reservation price equals mid-price)
    2. Positive inventory → r < s
    3. Negative inventory → r > s
    4. Adjustment vanishes as t → T (tau → 0)
    """
    s = 100.0
    gamma = params.gamma
    sigma = params.sigma
    T = params.T

    # Property 1: zero inventory
    r_zero = reservation_price(s, 0, gamma, sigma, T / 2)
    assert abs(r_zero - s) < 1e-12, \
        f"r(s,0,t) should equal s, got {r_zero}"

    # Property 2: positive inventory → r < s
    r_pos = reservation_price(s, 5, gamma, sigma, T / 2)
    assert r_pos < s, \
        f"Positive inventory should give r < s, got r={r_pos}, s={s}"

    # Property 3: negative inventory → r > s
    r_neg = reservation_price(s, -5, gamma, sigma, T / 2)
    assert r_neg > s, \
        f"Negative inventory should give r > s, got r={r_neg}, s={s}"

    # Property 4: adjustment vanishes as tau → 0
    r_terminal = reservation_price(s, 10, gamma, sigma, 0.0)
    assert abs(r_terminal - s) < 1e-12, \
        f"At T, reservation price should equal s, got {r_terminal}"


def validate_lambda_dt(params: ASParams) -> None:
    """
    Validate that lambda*dt < 1 for the Bernoulli approximation to be valid.
    Worst case is delta → 0, giving lambda_max = A.
    """
    lambda_max = params.A
    assert lambda_max * params.dt < 1.0, (
        f"lambda_max*dt = {lambda_max * params.dt:.4f} >= 1; "
        "Bernoulli approximation is invalid."
    )
