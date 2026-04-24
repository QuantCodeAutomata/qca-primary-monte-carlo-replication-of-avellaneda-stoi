"""
Experiment 2: Diagnostic Replication — Spread-Ambiguity Variant.

Addresses the paper's internal inconsistency between the theoretically implied
time-varying spread and the constant spread values reported in the published
summary tables.

The paper's tables report spread = 2/gamma * ln(1 + gamma/k), whereas the
finite-horizon formula implies an additional term gamma*sigma^2*(T-t).

This diagnostic uses the constant-spread interpretation to test whether it
better matches the published table values.

Reference: Avellaneda & Stoikov (2008), Quantitative Finance.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.simulation import (
    SimParams,
    SimResults,
    PathRecord,
    execution_intensity,
    reservation_price,
    draw_shared_randoms,
    validate_parameters,
)

# Custom — Context7 found no library equivalent for AS constant-spread diagnostic variant


GAMMAS = [0.1, 0.01, 0.5]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def constant_half_spread(gamma: float, k: float) -> float:
    """
    Compute the constant half-spread component used in the table-matching variant.

    c_gamma = (1/gamma) * ln(1 + gamma/k)

    This matches the spread values reported in the paper's published tables.

    Parameters
    ----------
    gamma : risk-aversion coefficient
    k     : intensity decay parameter
    """
    return (1.0 / gamma) * np.log(1.0 + gamma / k)


def diagnostic_quote_distances(
    q: int, gamma: float, sigma: float, k: float, tau: float
) -> Tuple[float, float]:
    """
    Compute diagnostic (constant-spread) quote distances.

    Forces total spread = 2 * c_gamma while preserving centering around
    the reservation price:

    delta^a = c_gamma - q * gamma * sigma^2 * tau
    delta^b = c_gamma + q * gamma * sigma^2 * tau

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
    c = constant_half_spread(gamma, k)
    adj = q * gamma * sigma ** 2 * tau
    delta_a = c - adj
    delta_b = c + adj
    return delta_a, delta_b


def simulate_path_diagnostic(
    params: SimParams,
    gamma: float,
    strategy: str,
    price_moves: np.ndarray,
    ask_uniforms: np.ndarray,
    bid_uniforms: np.ndarray,
    record: bool = False,
) -> Tuple[float, int, PathRecord | None]:
    """
    Simulate one Monte Carlo path using the diagnostic (constant-spread) variant.

    For the inventory strategy, uses constant-spread quote distances that
    preserve reservation-price centering but fix total spread to 2*c_gamma.
    For the symmetric benchmark, uses the same constant spread centered at mid.

    Parameters
    ----------
    params        : SimParams instance
    gamma         : risk-aversion coefficient
    strategy      : "inventory" or "symmetric"
    price_moves   : pre-drawn ±1 array of length n_steps
    ask_uniforms  : pre-drawn U[0,1] array of length n_steps
    bid_uniforms  : pre-drawn U[0,1] array of length n_steps
    record        : if True, store full state arrays

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

    c_gamma = constant_half_spread(gamma, k)

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
        tau = T - t

        if strategy == "inventory":
            delta_a, delta_b = diagnostic_quote_distances(q, gamma, sigma, k, tau)
            pa = s + delta_a
            pb = s - delta_b
            r = reservation_price(s, q, gamma, sigma, tau)
        else:  # symmetric benchmark with constant spread
            pa = s + c_gamma
            pb = s - c_gamma
            delta_a = c_gamma
            delta_b = c_gamma
            r = np.nan

        lam_a = execution_intensity(delta_a, A, k)
        lam_b = execution_intensity(delta_b, A, k)

        prob_a = lam_a * dt
        prob_b = lam_b * dt

        ask_filled = ask_uniforms[i] < prob_a
        bid_filled = bid_uniforms[i] < prob_b

        if ask_filled:
            q -= 1
            x += pa
        if bid_filled:
            q += 1
            x -= pb

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


def run_monte_carlo_diagnostic(
    params: SimParams,
    gamma: float,
    strategy: str,
    shared_price_moves: np.ndarray | None = None,
    shared_ask_uniforms: np.ndarray | None = None,
    shared_bid_uniforms: np.ndarray | None = None,
) -> SimResults:
    """
    Run the diagnostic Monte Carlo simulation for a given strategy and gamma.

    Parameters
    ----------
    params               : SimParams instance
    gamma                : risk-aversion coefficient
    strategy             : "inventory" or "symmetric"
    shared_price_moves   : optional pre-drawn price moves (n_paths × n_steps)
    shared_ask_uniforms  : optional pre-drawn ask uniforms (n_paths × n_steps)
    shared_bid_uniforms  : optional pre-drawn bid uniforms (n_paths × n_steps)

    Returns
    -------
    SimResults with aggregated statistics
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
        profit, final_q, path = simulate_path_diagnostic(
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


def run_experiment_2(
    params: SimParams | None = None,
    gammas: List[float] | None = None,
    results_dir: str = RESULTS_DIR,
) -> Dict[str, Dict[str, SimResults]]:
    """
    Run Experiment 2: diagnostic constant-spread replication.

    Uses the same simulation engine as Experiment 1 but replaces the
    time-varying finite-horizon spread with the constant spread 2*c_gamma
    that matches the paper's published table values.

    Parameters
    ----------
    params      : SimParams (defaults to paper's baseline values)
    gammas      : list of risk-aversion values
    results_dir : directory to save outputs

    Returns
    -------
    Nested dict: results[gamma_str][strategy] = SimResults
    """
    if params is None:
        params = SimParams()
    if gammas is None:
        gammas = GAMMAS

    os.makedirs(results_dir, exist_ok=True)

    for g in gammas:
        validate_parameters(params, g)

    all_results: Dict[str, Dict[str, SimResults]] = {}

    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        all_results[g_key] = {}

        # Use common random numbers — same seed as Exp 1 for direct comparison
        price_moves, ask_uniforms, bid_uniforms = draw_shared_randoms(params)

        for strategy in ["inventory", "symmetric"]:
            res = run_monte_carlo_diagnostic(
                params,
                gamma,
                strategy,
                shared_price_moves=price_moves,
                shared_ask_uniforms=ask_uniforms,
                shared_bid_uniforms=bid_uniforms,
            )
            all_results[g_key][strategy] = res

    _print_summary_table_diag(all_results, gammas, params)
    _save_summary_csv_diag(all_results, gammas, results_dir)
    _plot_profit_histograms_diag(all_results, gammas, results_dir)

    return all_results


def _print_summary_table_diag(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    params: SimParams,
) -> None:
    """Print diagnostic summary table with reported spread values."""
    print(f"\n{'='*80}")
    print("Exp 2 (Constant-Spread Diagnostic) — Summary Statistics")
    print(f"{'='*80}")
    fmt = "{:<10} {:<12} {:>12} {:>12} {:>12} {:>12} {:>12}"
    print(fmt.format("gamma", "strategy", "E[Pi_T]", "Std[Pi_T]", "E[q_T]", "Std[q_T]", "spread"))
    print("-" * 80)
    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        c = constant_half_spread(gamma, params.k)
        spread = 2 * c
        for strategy in ["inventory", "symmetric"]:
            r = results[g_key][strategy]
            print(
                fmt.format(
                    f"{gamma:.3f}",
                    strategy,
                    f"{r.mean_profit:.2f}",
                    f"{r.std_profit:.2f}",
                    f"{r.mean_final_q:.2f}",
                    f"{r.std_final_q:.2f}",
                    f"{spread:.4f}",
                )
            )
    print("=" * 80)


def _save_summary_csv_diag(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    results_dir: str,
) -> None:
    """Save diagnostic summary statistics to CSV."""
    rows = []
    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        for strategy in ["inventory", "symmetric"]:
            r = results[g_key][strategy]
            rows.append(
                {
                    "gamma": gamma,
                    "strategy": strategy,
                    "mean_profit": round(r.mean_profit, 4),
                    "std_profit": round(r.std_profit, 4),
                    "mean_final_q": round(r.mean_final_q, 4),
                    "std_final_q": round(r.std_final_q, 4),
                }
            )
    df = pd.DataFrame(rows)
    path = os.path.join(results_dir, "exp2_summary.csv")
    df.to_csv(path, index=False)
    print(f"Saved diagnostic summary CSV → {path}")


def _plot_profit_histograms_diag(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    results_dir: str,
) -> None:
    """Plot overlaid profit histograms for the diagnostic variant."""
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 5))
    if len(gammas) == 1:
        axes = [axes]

    for ax, gamma in zip(axes, gammas):
        g_key = f"gamma_{gamma}"
        inv_profits = results[g_key]["inventory"].profits
        sym_profits = results[g_key]["symmetric"].profits

        bins = np.linspace(
            min(inv_profits.min(), sym_profits.min()),
            max(inv_profits.max(), sym_profits.max()),
            50,
        )
        ax.hist(inv_profits, bins=bins, alpha=0.55, color="steelblue",
                label="Inventory", density=True)
        ax.hist(sym_profits, bins=bins, alpha=0.55, color="tomato",
                label="Symmetric", density=True)
        ax.set_title(f"Terminal Profit Distribution\nγ={gamma} (Constant Spread)")
        ax.set_xlabel("$\\Pi_T = X_T + q_T S_T$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(np.mean(inv_profits), color="steelblue", lw=1.5, ls="--")
        ax.axvline(np.mean(sym_profits), color="tomato", lw=1.5, ls="--")

    plt.suptitle("Exp 2: Terminal Profit Histograms (Constant-Spread Diagnostic)", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "exp2_profit_histograms.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic profit histogram → {fig_path}")


def build_results_markdown(
    results: Dict[str, Dict[str, SimResults]],
    gammas: List[float],
    params: SimParams,
) -> str:
    """Build a Markdown table string for the diagnostic results."""
    lines = [
        "## Experiment 2: Diagnostic Replication (Constant-Spread Variant)",
        "",
        "| gamma | strategy | spread | E[Pi_T] | Std[Pi_T] | E[q_T] | Std[q_T] |",
        "|-------|----------|--------|---------|-----------|--------|----------|",
    ]
    for gamma in gammas:
        g_key = f"gamma_{gamma}"
        c = constant_half_spread(gamma, params.k)
        spread = 2 * c
        for strategy in ["inventory", "symmetric"]:
            r = results[g_key][strategy]
            lines.append(
                f"| {gamma} | {strategy} | {spread:.4f} | {r.mean_profit:.2f} | "
                f"{r.std_profit:.2f} | {r.mean_final_q:.2f} | {r.std_final_q:.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment_2()
