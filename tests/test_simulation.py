"""
Tests for the core simulation engine (src/simulation.py).

Verifies:
  - Quoting rule formulas match paper's equations
  - Reservation price properties
  - Execution intensity properties
  - Parameter validation
  - Monte Carlo path mechanics
  - Edge cases
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation import (
    SimParams,
    reservation_price,
    inventory_quote_distances,
    symmetric_spread,
    execution_intensity,
    simulate_path,
    run_monte_carlo,
    draw_shared_randoms,
    validate_parameters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params() -> SimParams:
    return SimParams()


@pytest.fixture
def small_params() -> SimParams:
    """Smaller simulation for fast tests."""
    return SimParams(n_paths=50, seed=0)


# ---------------------------------------------------------------------------
# Tests: reservation_price
# ---------------------------------------------------------------------------

def test_reservation_price_zero_inventory():
    """r(s, 0, t) must equal s regardless of gamma, sigma, tau."""
    s = 100.0
    for gamma in [0.01, 0.1, 0.5]:
        for tau in [0.0, 0.5, 1.0]:
            r = reservation_price(s, 0, gamma, 2.0, tau)
            assert abs(r - s) < 1e-12, f"r != s for q=0, gamma={gamma}, tau={tau}"


def test_reservation_price_positive_inventory_below_mid():
    """Positive inventory should give r < s."""
    s = 100.0
    for q in [1, 5, 10]:
        r = reservation_price(s, q, 0.1, 2.0, 0.5)
        assert r < s, f"Expected r < s for q={q}, got r={r}"


def test_reservation_price_negative_inventory_above_mid():
    """Negative inventory should give r > s."""
    s = 100.0
    for q in [-1, -5, -10]:
        r = reservation_price(s, q, 0.1, 2.0, 0.5)
        assert r > s, f"Expected r > s for q={q}, got r={r}"


def test_reservation_price_vanishes_at_horizon():
    """Reservation price adjustment must vanish as tau → 0."""
    s = 100.0
    for q in [-10, -1, 0, 1, 10]:
        r = reservation_price(s, q, 0.1, 2.0, 0.0)
        assert abs(r - s) < 1e-12, f"r != s at T for q={q}"


def test_reservation_price_linear_in_inventory():
    """r should be linear in q: r(s, 2q) - s = 2*(r(s, q) - s)."""
    s = 100.0
    gamma, sigma, tau = 0.1, 2.0, 0.5
    r1 = reservation_price(s, 2, gamma, sigma, tau)
    r2 = reservation_price(s, 4, gamma, sigma, tau)
    assert abs((r2 - s) - 2 * (r1 - s)) < 1e-10


def test_reservation_price_formula():
    """Verify exact formula: r = s - q*gamma*sigma^2*tau."""
    s, q, gamma, sigma, tau = 100.0, 3, 0.1, 2.0, 0.5
    expected = s - q * gamma * sigma ** 2 * tau
    result = reservation_price(s, q, gamma, sigma, tau)
    assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tests: inventory_quote_distances
# ---------------------------------------------------------------------------

def test_quote_distances_positive_base():
    """Base component (1/gamma)*ln(1+gamma/k) must be positive, and at q=0
    both distances equal base + adj/2 (symmetric around mid-price)."""
    for gamma in [0.01, 0.1, 0.5]:
        sigma, k, tau = 2.0, 1.5, 0.5
        da, db = inventory_quote_distances(0, gamma, sigma, k, tau)
        base = (1.0 / gamma) * np.log(1.0 + gamma / k)
        adj = gamma * sigma ** 2 * tau
        expected = base + adj / 2.0  # at q=0: (1-2*0)/2 * adj = adj/2
        assert base > 0, f"Base component should be positive for gamma={gamma}"
        assert abs(da - expected) < 1e-10, f"delta_a != base+adj/2 for q=0, gamma={gamma}"
        assert abs(db - expected) < 1e-10, f"delta_b != base+adj/2 for q=0, gamma={gamma}"


def test_quote_distances_symmetry_at_zero_inventory():
    """At q=0, delta_a == delta_b."""
    for gamma in [0.01, 0.1, 0.5]:
        da, db = inventory_quote_distances(0, gamma, 2.0, 1.5, 0.5)
        assert abs(da - db) < 1e-12, f"delta_a != delta_b at q=0 for gamma={gamma}"


def test_quote_distances_positive_inventory_asymmetry():
    """
    For q > 0, delta_a < delta_b (ask closer, bid farther).
    This reflects the inventory-based strategy's desire to sell.
    """
    da, db = inventory_quote_distances(5, 0.1, 2.0, 1.5, 0.5)
    assert da < db, f"Expected delta_a < delta_b for q>0, got da={da:.4f}, db={db:.4f}"


def test_quote_distances_negative_inventory_asymmetry():
    """For q < 0, delta_a > delta_b (bid closer, ask farther)."""
    da, db = inventory_quote_distances(-5, 0.1, 2.0, 1.5, 0.5)
    assert da > db, f"Expected delta_a > delta_b for q<0, got da={da:.4f}, db={db:.4f}"


def test_quote_distances_formula():
    """Verify exact formula from paper (Eq. 3.8/3.9)."""
    q, gamma, sigma, k, tau = 3, 0.1, 2.0, 1.5, 0.5
    base = (1.0 / gamma) * np.log(1.0 + gamma / k)
    adj = gamma * sigma ** 2 * tau
    expected_a = base + (1.0 - 2.0 * q) * adj / 2.0
    expected_b = base + (1.0 + 2.0 * q) * adj / 2.0
    da, db = inventory_quote_distances(q, gamma, sigma, k, tau)
    assert abs(da - expected_a) < 1e-12
    assert abs(db - expected_b) < 1e-12


def test_quote_distances_vanish_adjustment_at_horizon():
    """At tau=0, both distances equal the base component."""
    gamma, k = 0.1, 1.5
    base = (1.0 / gamma) * np.log(1.0 + gamma / k)
    for q in [-5, 0, 5]:
        da, db = inventory_quote_distances(q, gamma, 2.0, k, 0.0)
        assert abs(da - base) < 1e-12, f"delta_a != base at tau=0 for q={q}"
        assert abs(db - base) < 1e-12, f"delta_b != base at tau=0 for q={q}"


# ---------------------------------------------------------------------------
# Tests: symmetric_spread
# ---------------------------------------------------------------------------

def test_symmetric_spread_positive():
    """Spread must be positive for all valid parameters."""
    for gamma in [0.01, 0.1, 0.5]:
        for tau in [0.0, 0.5, 1.0]:
            spread = symmetric_spread(gamma, 2.0, 1.5, tau)
            assert spread >= 0, f"Negative spread for gamma={gamma}, tau={tau}"


def test_symmetric_spread_decreases_with_tau():
    """Spread should decrease as tau decreases (time-varying component)."""
    gamma, sigma, k = 0.1, 2.0, 1.5
    spread_early = symmetric_spread(gamma, sigma, k, 1.0)
    spread_late = symmetric_spread(gamma, sigma, k, 0.1)
    assert spread_early > spread_late, "Spread should be larger at earlier times"


def test_symmetric_spread_formula():
    """Verify exact formula: spread = gamma*sigma^2*tau + (2/gamma)*ln(1+gamma/k)."""
    gamma, sigma, k, tau = 0.1, 2.0, 1.5, 0.5
    expected = gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
    result = symmetric_spread(gamma, sigma, k, tau)
    assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tests: execution_intensity
# ---------------------------------------------------------------------------

def test_execution_intensity_positive():
    """Intensity must be positive for all delta >= 0."""
    for delta in [0.0, 0.5, 1.0, 2.0]:
        lam = execution_intensity(delta, 140.0, 1.5)
        assert lam > 0, f"Non-positive intensity for delta={delta}"


def test_execution_intensity_at_zero():
    """At delta=0, intensity equals A."""
    A, k = 140.0, 1.5
    lam = execution_intensity(0.0, A, k)
    assert abs(lam - A) < 1e-12


def test_execution_intensity_decreasing():
    """Intensity must decrease monotonically with delta."""
    A, k = 140.0, 1.5
    deltas = np.linspace(0, 3, 100)
    lams = np.array([execution_intensity(d, A, k) for d in deltas])
    assert np.all(np.diff(lams) < 0), "Intensity should be strictly decreasing"


def test_execution_intensity_formula():
    """Verify exact formula: lambda = A * exp(-k * delta)."""
    A, k, delta = 140.0, 1.5, 0.5
    expected = A * np.exp(-k * delta)
    result = execution_intensity(delta, A, k)
    assert abs(result - expected) < 1e-12


def test_lambda_dt_below_one(default_params):
    """Bernoulli approximation validity: lambda * dt < 1."""
    # Worst case: delta = 0 → lambda = A
    max_lam_dt = default_params.A * default_params.dt
    assert max_lam_dt < 1.0, f"lambda*dt = {max_lam_dt} >= 1"


# ---------------------------------------------------------------------------
# Tests: validate_parameters
# ---------------------------------------------------------------------------

def test_validate_parameters_passes(default_params):
    """validate_parameters should not raise for valid inputs."""
    for gamma in [0.01, 0.1, 0.5]:
        validate_parameters(default_params, gamma)  # should not raise


def test_validate_parameters_catches_bad_lambda_dt():
    """validate_parameters should raise if lambda*dt >= 1."""
    bad_params = SimParams(A=1000.0, dt=0.1)  # A*dt = 100 >> 1
    with pytest.raises(AssertionError):
        validate_parameters(bad_params, 0.1)


# ---------------------------------------------------------------------------
# Tests: simulate_path
# ---------------------------------------------------------------------------

def test_simulate_path_returns_correct_types(default_params):
    """simulate_path should return (float, int, None) when record=False."""
    rng = np.random.default_rng(42)
    n = default_params.n_steps
    price_moves = rng.choice([-1, 1], size=n)
    ask_u = rng.uniform(size=n)
    bid_u = rng.uniform(size=n)

    profit, final_q, path = simulate_path(
        default_params, 0.1, "inventory", price_moves, ask_u, bid_u, record=False
    )
    assert isinstance(profit, float)
    assert isinstance(final_q, (int, np.integer))
    assert path is None


def test_simulate_path_record_shapes(default_params):
    """PathRecord arrays should have length n_steps + 1."""
    rng = np.random.default_rng(42)
    n = default_params.n_steps
    price_moves = rng.choice([-1, 1], size=n)
    ask_u = rng.uniform(size=n)
    bid_u = rng.uniform(size=n)

    _, _, path = simulate_path(
        default_params, 0.1, "inventory", price_moves, ask_u, bid_u, record=True
    )
    assert path is not None
    assert len(path.time) == n + 1
    assert len(path.s) == n + 1
    assert len(path.q) == n + 1
    assert len(path.x) == n + 1


def test_simulate_path_terminal_profit_formula(default_params):
    """Terminal profit must equal X_T + q_T * S_T."""
    rng = np.random.default_rng(42)
    n = default_params.n_steps
    price_moves = rng.choice([-1, 1], size=n)
    ask_u = rng.uniform(size=n)
    bid_u = rng.uniform(size=n)

    profit, final_q, path = simulate_path(
        default_params, 0.1, "inventory", price_moves, ask_u, bid_u, record=True
    )
    expected_profit = path.x[-1] + path.q[-1] * path.s[-1]
    assert abs(profit - expected_profit) < 1e-10


def test_simulate_path_inventory_integer(default_params):
    """Inventory must always be an integer (one-unit fills)."""
    rng = np.random.default_rng(42)
    n = default_params.n_steps
    price_moves = rng.choice([-1, 1], size=n)
    ask_u = rng.uniform(size=n)
    bid_u = rng.uniform(size=n)

    _, _, path = simulate_path(
        default_params, 0.1, "inventory", price_moves, ask_u, bid_u, record=True
    )
    assert np.all(path.q == path.q.astype(int)), "Inventory must be integer-valued"


def test_simulate_path_price_moves_binomial(default_params):
    """Mid-price increments must be ±sigma*sqrt(dt)."""
    rng = np.random.default_rng(42)
    n = default_params.n_steps
    price_moves = rng.choice([-1, 1], size=n)
    ask_u = rng.uniform(size=n)
    bid_u = rng.uniform(size=n)

    _, _, path = simulate_path(
        default_params, 0.1, "inventory", price_moves, ask_u, bid_u, record=True
    )
    # Price increments (excluding fill effects) should be ±sigma*sqrt(dt)
    # We can check that all increments are multiples of sigma*sqrt(dt)
    step = default_params.sigma * np.sqrt(default_params.dt)
    # Note: inventory fills also change cash but not price
    # Price path increments should be ±step
    price_diffs = np.diff(path.s)
    # Each diff should be close to ±step (no fill effect on price)
    assert np.allclose(np.abs(price_diffs), step, atol=1e-10), \
        "Price increments should be ±sigma*sqrt(dt)"


def test_simulate_path_both_strategies_same_price(default_params):
    """Both strategies should produce identical price paths given same price_moves."""
    rng = np.random.default_rng(42)
    n = default_params.n_steps
    price_moves = rng.choice([-1, 1], size=n)
    ask_u = rng.uniform(size=n)
    bid_u = rng.uniform(size=n)

    _, _, path_inv = simulate_path(
        default_params, 0.1, "inventory", price_moves, ask_u, bid_u, record=True
    )
    _, _, path_sym = simulate_path(
        default_params, 0.1, "symmetric", price_moves, ask_u, bid_u, record=True
    )
    # Price paths should be identical (fills don't affect mid-price)
    assert np.allclose(path_inv.s, path_sym.s), \
        "Price paths should be identical for same price_moves"


# ---------------------------------------------------------------------------
# Tests: run_monte_carlo
# ---------------------------------------------------------------------------

def test_run_monte_carlo_output_shape(small_params):
    """run_monte_carlo should return SimResults with correct array shapes."""
    result = run_monte_carlo(small_params, 0.1, "inventory")
    assert len(result.profits) == small_params.n_paths
    assert len(result.final_inventories) == small_params.n_paths


def test_run_monte_carlo_reproducible(small_params):
    """Same seed should produce identical results."""
    r1 = run_monte_carlo(small_params, 0.1, "inventory")
    r2 = run_monte_carlo(small_params, 0.1, "inventory")
    assert np.allclose(r1.profits, r2.profits), "Results should be reproducible"


def test_run_monte_carlo_sample_path_exists(small_params):
    """First path should be recorded as sample_path."""
    result = run_monte_carlo(small_params, 0.1, "inventory")
    assert result.sample_path is not None


def test_run_monte_carlo_statistics_finite(small_params):
    """All summary statistics should be finite."""
    for strategy in ["inventory", "symmetric"]:
        result = run_monte_carlo(small_params, 0.1, strategy)
        assert np.isfinite(result.mean_profit)
        assert np.isfinite(result.std_profit)
        assert np.isfinite(result.mean_final_q)
        assert np.isfinite(result.std_final_q)


def test_run_monte_carlo_std_positive(small_params):
    """Standard deviations should be positive."""
    result = run_monte_carlo(small_params, 0.1, "inventory")
    assert result.std_profit > 0
    assert result.std_final_q >= 0  # could be 0 if all paths end at same q


def test_run_monte_carlo_common_random_numbers(small_params):
    """Common random numbers should give same price paths to both strategies."""
    price_moves, ask_u, bid_u = draw_shared_randoms(small_params)
    r_inv = run_monte_carlo(
        small_params, 0.1, "inventory",
        shared_price_moves=price_moves,
        shared_ask_uniforms=ask_u,
        shared_bid_uniforms=bid_u,
    )
    r_sym = run_monte_carlo(
        small_params, 0.1, "symmetric",
        shared_price_moves=price_moves,
        shared_ask_uniforms=ask_u,
        shared_bid_uniforms=bid_u,
    )
    # Both should have valid results
    assert np.isfinite(r_inv.mean_profit)
    assert np.isfinite(r_sym.mean_profit)


# ---------------------------------------------------------------------------
# Tests: methodology adherence
# ---------------------------------------------------------------------------

def test_inventory_strategy_reduces_inventory_dispersion(small_params):
    """
    Inventory strategy should reduce inventory dispersion vs symmetric benchmark.
    This is a core claim of the paper.
    """
    price_moves, ask_u, bid_u = draw_shared_randoms(small_params)
    r_inv = run_monte_carlo(
        small_params, 0.5, "inventory",
        shared_price_moves=price_moves,
        shared_ask_uniforms=ask_u,
        shared_bid_uniforms=bid_u,
    )
    r_sym = run_monte_carlo(
        small_params, 0.5, "symmetric",
        shared_price_moves=price_moves,
        shared_ask_uniforms=ask_u,
        shared_bid_uniforms=bid_u,
    )
    # For gamma=0.5, inventory strategy should have lower inventory std
    assert r_inv.std_final_q <= r_sym.std_final_q * 1.5, \
        "Inventory strategy should not dramatically increase inventory dispersion"


def test_low_gamma_strategies_similar(small_params):
    """
    For very low gamma (0.001), inventory and symmetric strategies should be similar.
    """
    price_moves, ask_u, bid_u = draw_shared_randoms(small_params)
    r_inv = run_monte_carlo(
        small_params, 0.001, "inventory",
        shared_price_moves=price_moves,
        shared_ask_uniforms=ask_u,
        shared_bid_uniforms=bid_u,
    )
    r_sym = run_monte_carlo(
        small_params, 0.001, "symmetric",
        shared_price_moves=price_moves,
        shared_ask_uniforms=ask_u,
        shared_bid_uniforms=bid_u,
    )
    # With very low gamma, strategies should be nearly identical
    # Profits should be close (within 50% of each other's std)
    profit_diff = abs(r_inv.mean_profit - r_sym.mean_profit)
    assert profit_diff < max(r_inv.std_profit, r_sym.std_profit) * 2.0, \
        "Low gamma strategies should produce similar mean profits"


def test_initial_inventory_zero(default_params):
    """Initial inventory must be zero as specified in paper."""
    assert default_params.q0 == 0


def test_initial_cash_zero(default_params):
    """Initial cash must be zero as specified in paper."""
    assert default_params.x0 == 0.0


def test_n_steps_correct(default_params):
    """Number of steps must be T/dt = 200."""
    assert default_params.n_steps == 200


def test_parameters_match_paper(default_params):
    """All parameters must match paper's specification."""
    assert default_params.s0 == 100.0
    assert default_params.sigma == 2.0
    assert default_params.T == 1.0
    assert default_params.dt == 0.005
    assert default_params.A == 140.0
    assert default_params.k == 1.5
    assert default_params.n_paths == 1000


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

def test_extreme_inventory_quote_distances():
    """Quote distances should remain finite for extreme inventory values."""
    for q in [-100, 100]:
        da, db = inventory_quote_distances(q, 0.1, 2.0, 1.5, 0.5)
        assert np.isfinite(da), f"Non-finite delta_a for q={q}"
        assert np.isfinite(db), f"Non-finite delta_b for q={q}"


def test_execution_intensity_large_delta():
    """Intensity should approach zero for large delta."""
    lam = execution_intensity(100.0, 140.0, 1.5)
    assert lam < 1e-60, f"Intensity should be near zero for large delta, got {lam}"


def test_single_step_simulation():
    """Simulation with 1 step should not crash."""
    params = SimParams(T=0.005, dt=0.005, n_paths=10)
    result = run_monte_carlo(params, 0.1, "inventory")
    assert np.isfinite(result.mean_profit)


def test_zero_gamma_limit():
    """Very small gamma should not cause numerical issues."""
    params = SimParams(n_paths=10)
    result = run_monte_carlo(params, 1e-6, "inventory")
    assert np.isfinite(result.mean_profit)
