"""
Core Monte Carlo simulation engine for the Avellaneda-Stoikov (2008) market-making model.

Implements the finite-horizon approximation under:
  - Arithmetic Brownian Motion mid-price dynamics (binomial discretisation)
  - Symmetric exponential execution intensities: lambda(delta) = A * exp(-k * delta)
  - CARA (exponential utility) optimal quoting rules

Reference: Avellaneda & Stoikov, "High-frequency trading in a limit order book",
           Quantitative Finance, 2008.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# Custom — Context7 found no library equivalent for AS finite-horizon quoting rules (paper Eq. 3.8 / 3.9)


@dataclass
class SimParams:
    """Simulation parameters matching the paper's numerical experiments."""

    s0: float = 100.0       # Initial mid-price
    sigma: float = 2.0      # Volatility (arithmetic BM)
    T: float = 1.0          # Horizon
    dt: float = 0.005       # Time step  → 200 steps
    q0: int = 0             # Initial inventory
    x0: float = 0.0         # Initial cash (CARA controls are cash-independent)
    A: float = 140.0        # Intensity scale parameter
    k: float = 1.5          # Intensity decay parameter
    n_paths: int = 1000     # Monte Carlo paths
    seed: int = 42          # Base random seed

    @property
    def n_steps(self) -> int:
        return round(self.T / self.dt)


@dataclass
class PathRecord:
    """Per-path state arrays for debugging and analysis."""

    time: np.ndarray
    s: np.ndarray       # mid-price
    x: np.ndarray       # cash
    q: np.ndarray       # inventory
    r: np.ndarray       # reservation price (NaN for symmetric strategy)
    pa: np.ndarray      # ask quote
    pb: np.ndarray      # bid quote
    ask_fill: np.ndarray   # 1 if ask filled at this step
    bid_fill: np.ndarray   # 1 if bid filled at this step


@dataclass
class SimResults:
    """Aggregated simulation results across all Monte Carlo paths."""

    gamma: float
    strategy: str                   # "inventory" or "symmetric"
    mean_profit: float
    std_profit: float
    mean_final_q: float
    std_final_q: float
    profits: np.ndarray             # shape (n_paths,)
    final_inventories: np.ndarray   # shape (n_paths,)
    sample_path: PathRecord = field(default=None)  # first path for plotting


# ---------------------------------------------------------------------------
# Quoting rules
# ---------------------------------------------------------------------------

def reservation_price(s: float, q: int, gamma: float, sigma: float, tau: float) -> float:
    """
    Compute the inventory-adjusted reservation price.

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s     : current mid-price
    q     : current inventory
    gamma : risk-aversion coefficient
    sigma : volatility
    tau   : time remaining (T - t)
    """
    return s - q * gamma * sigma ** 2 * tau


def inventory_quote_distances(
    q: int, gamma: float, sigma: float, k: float, tau: float
) -> Tuple[float, float]:
    """
    Compute optimal ask and bid quote distances from the mid-price using the
    paper's finite-horizon closed-form approximation (Eq. 3.8 / 3.9).

    delta^a = (1/gamma) * ln(1 + gamma/k) + ((1 - 2q) * gamma * sigma^2 * tau) / 2
    delta^b = (1/gamma) * ln(1 + gamma/k) + ((1 + 2q) * gamma * sigma^2 * tau) / 2

    Parameters
    ----------
    q     : current inventory
    gamma : risk-aversion coefficient
    sigma : volatility
    k     : intensity decay parameter
    tau   : time remaining (T - t)

    Returns
    -------
    (delta_a, delta_b) : ask and bid distances from mid-price
    """
    base = (1.0 / gamma) * np.log(1.0 + gamma / k)
    adj = gamma * sigma ** 2 * tau
    delta_a = base + (1.0 - 2.0 * q) * adj / 2.0
    delta_b = base + (1.0 + 2.0 * q) * adj / 2.0
    return delta_a, delta_b


def symmetric_spread(gamma: float, sigma: float, k: float, tau: float) -> float:
    """
    Compute the theoretically implied finite-horizon total spread for the
    symmetric benchmark strategy.

    spread(t) = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

    Parameters
    ----------
    gamma : risk-aversion coefficient
    sigma : volatility
    k     : intensity decay parameter
    tau   : time remaining (T - t)
    """
    return gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)


def execution_intensity(delta: float, A: float, k: float) -> float:
    """
    Compute execution intensity for a quote at distance delta.

    lambda(delta) = A * exp(-k * delta)

    Parameters
    ----------
    delta : quote distance from mid-price (should be >= 0 for meaningful intensity)
    A     : intensity scale
    k     : intensity decay
    """
    return A * np.exp(-k * delta)


# ---------------------------------------------------------------------------
# Single-path simulation
# ---------------------------------------------------------------------------

def simulate_path(
    params: SimParams,
    gamma: float,
    strategy: str,
    price_moves: np.ndarray,
    ask_uniforms: np.ndarray,
    bid_uniforms: np.ndarray,
    record: bool = False,
) -> Tuple[float, int, PathRecord | None]:
    """
    Simulate one Monte Carlo path for a given strategy.

    Bid and ask Bernoulli execution events are simulated independently within
    each dt interval. Simultaneous fills are therefore possible but rare because
    lambda * dt << 1 under the chosen parameters. This is the faithful choice
    described in the implementation notes.

    Parameters
    ----------
    params        : SimParams instance
    gamma         : risk-aversion coefficient
    strategy      : "inventory" or "symmetric"
    price_moves   : pre-drawn ±1 array of length n_steps for mid-price increments
    ask_uniforms  : pre-drawn U[0,1] array of length n_steps for ask fill decisions
    bid_uniforms  : pre-drawn U[0,1] array of length n_steps for bid fill decisions
    record        : if True, store full state arrays in a PathRecord

    Returns
    -------
    (terminal_profit, terminal_inventory, path_record_or_None)
    """
    n = params.n_steps
    s = params.s0
    x = params.x0
    q = params.q0
    dt = params.dt
    sigma = params.sigma
    A = params.A
    k = params.k
    T = params.T

    if record:
        times = np.empty(n + 1)
        s_arr = np.empty(n + 1)
        x_arr = np.empty(n + 1)
        q_arr = np.empty(n + 1, dtype=int)
        r_arr = np.full(n + 1, np.nan)
        pa_arr = np.empty(n + 1)
        pb_arr = np.empty(n + 1)
        af_arr = np.zeros(n + 1, dtype=int)
        bf_arr = np.zeros(n + 1, dtype=int)
        times[0] = 0.0
        s_arr[0] = s
        x_arr[0] = x
        q_arr[0] = q

    for i in range(n):
        t = i * dt
        tau = T - t  # time remaining

        if strategy == "inventory":
            delta_a, delta_b = inventory_quote_distances(q, gamma, sigma, k, tau)
            # Quotes are placed at distance delta from the mid-price
            pa = s + delta_a
            pb = s - delta_b
            r = reservation_price(s, q, gamma, sigma, tau)
        else:  # symmetric benchmark
            spread = symmetric_spread(gamma, sigma, k, tau)
            half = spread / 2.0
            pa = s + half
            pb = s - half
            delta_a = half
            delta_b = half
            r = np.nan

        # Execution intensities
        lam_a = execution_intensity(delta_a, A, k)
        lam_b = execution_intensity(delta_b, A, k)

        # Bernoulli execution probabilities (paper's approximation)
        prob_a = lam_a * dt
        prob_b = lam_b * dt

        # Independent Bernoulli draws for ask and bid fills
        ask_filled = ask_uniforms[i] < prob_a
        bid_filled = bid_uniforms[i] < prob_b

        if ask_filled:
            q -= 1
            x += pa
        if bid_filled:
            q += 1
            x -= pb

        # Mid-price binomial update: ±sigma*sqrt(dt) with equal probability
        s += price_moves[i] * sigma * np.sqrt(dt)

        if record:
            times[i + 1] = t + dt
            s_arr[i + 1] = s
            x_arr[i + 1] = x
            q_arr[i + 1] = q
            r_arr[i] = r
            pa_arr[i] = pa
            pb_arr[i] = pb
            af_arr[i] = int(ask_filled)
            bf_arr[i] = int(bid_filled)

    terminal_profit = x + q * s

    if record:
        r_arr[n] = np.nan
        pa_arr[n] = np.nan
        pb_arr[n] = np.nan
        path = PathRecord(
            time=times,
            s=s_arr,
            x=x_arr,
            q=q_arr,
            r=r_arr,
            pa=pa_arr,
            pb=pb_arr,
            ask_fill=af_arr,
            bid_fill=bf_arr,
        )
        return terminal_profit, q, path

    return terminal_profit, q, None


# ---------------------------------------------------------------------------
# Full Monte Carlo run
# ---------------------------------------------------------------------------

def run_monte_carlo(
    params: SimParams,
    gamma: float,
    strategy: str,
    shared_price_moves: np.ndarray | None = None,
    shared_ask_uniforms: np.ndarray | None = None,
    shared_bid_uniforms: np.ndarray | None = None,
) -> SimResults:
    """
    Run the full Monte Carlo simulation for a given strategy and gamma.

    Parameters
    ----------
    params               : SimParams instance
    gamma                : risk-aversion coefficient
    strategy             : "inventory" or "symmetric"
    shared_price_moves   : optional pre-drawn price moves (n_paths × n_steps)
                           for common random numbers across strategies
    shared_ask_uniforms  : optional pre-drawn ask uniforms (n_paths × n_steps)
    shared_bid_uniforms  : optional pre-drawn bid uniforms (n_paths × n_steps)

    Returns
    -------
    SimResults with aggregated statistics and a sample path
    """
    rng = np.random.default_rng(params.seed)
    n = params.n_steps
    m = params.n_paths

    if shared_price_moves is not None:
        price_moves = shared_price_moves
    else:
        price_moves = rng.choice([-1, 1], size=(m, n))

    if shared_ask_uniforms is not None:
        ask_uniforms = shared_ask_uniforms
    else:
        ask_uniforms = rng.uniform(size=(m, n))

    if shared_bid_uniforms is not None:
        bid_uniforms = shared_bid_uniforms
    else:
        bid_uniforms = rng.uniform(size=(m, n))

    profits = np.empty(m)
    final_qs = np.empty(m, dtype=int)
    sample_path = None

    for j in range(m):
        record = j == 0
        profit, final_q, path = simulate_path(
            params,
            gamma,
            strategy,
            price_moves[j],
            ask_uniforms[j],
            bid_uniforms[j],
            record=record,
        )
        profits[j] = profit
        final_qs[j] = final_q
        if record:
            sample_path = path

    return SimResults(
        gamma=gamma,
        strategy=strategy,
        mean_profit=float(np.mean(profits)),
        std_profit=float(np.std(profits, ddof=1)),
        mean_final_q=float(np.mean(final_qs)),
        std_final_q=float(np.std(final_qs, ddof=1)),
        profits=profits,
        final_inventories=final_qs,
        sample_path=sample_path,
    )


def draw_shared_randoms(params: SimParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw shared random arrays for common random numbers across strategies.

    Returns
    -------
    (price_moves, ask_uniforms, bid_uniforms) each of shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(params.seed)
    n = params.n_steps
    m = params.n_paths
    price_moves = rng.choice([-1, 1], size=(m, n))
    ask_uniforms = rng.uniform(size=(m, n))
    bid_uniforms = rng.uniform(size=(m, n))
    return price_moves, ask_uniforms, bid_uniforms


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_parameters(params: SimParams, gamma: float) -> None:
    """
    Validate key parameter relationships before running simulations.

    Checks:
    1. r(s, 0, t) = s  (zero inventory → reservation price equals mid-price)
    2. Positive inventory → r < s
    3. Negative inventory → r > s
    4. Reservation-price adjustment vanishes as t → T
    5. lambda * dt < 1 (Bernoulli approximation validity)
    """
    s = params.s0
    sigma = params.sigma
    T = params.T
    A = params.A
    k = params.k
    dt = params.dt

    # Check 1: zero inventory
    r_zero = reservation_price(s, 0, gamma, sigma, T / 2)
    assert abs(r_zero - s) < 1e-12, f"r(s,0,t) != s: got {r_zero}"

    # Check 2: positive inventory
    r_pos = reservation_price(s, 5, gamma, sigma, T / 2)
    assert r_pos < s, f"Positive inventory should give r < s, got r={r_pos}"

    # Check 3: negative inventory
    r_neg = reservation_price(s, -5, gamma, sigma, T / 2)
    assert r_neg > s, f"Negative inventory should give r > s, got r={r_neg}"

    # Check 4: adjustment vanishes at T
    r_at_T = reservation_price(s, 10, gamma, sigma, 0.0)
    assert abs(r_at_T - s) < 1e-12, f"Reservation price should equal s at T, got {r_at_T}"

    # Check 5: lambda * dt < 1
    # Worst case: delta = 0 → lambda = A
    max_lam_dt = A * dt
    assert max_lam_dt < 1.0, (
        f"lambda*dt = {max_lam_dt:.4f} >= 1; Bernoulli approximation invalid"
    )
