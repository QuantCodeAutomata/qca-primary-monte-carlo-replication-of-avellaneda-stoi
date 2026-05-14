"""
Tests for the Avellaneda-Stoikov core model (src/avellaneda_stoikov.py).

Tests verify:
  - Mathematical properties of reservation price and quote formulas
  - Bernoulli approximation validity
  - Monte Carlo simulation correctness
  - Edge cases and boundary conditions
"""

from __future__ import annotations

import numpy as np
import pytest

from src.avellaneda_stoikov import (
    ASParams,
    reservation_price,
    optimal_half_spread_base,
    finite_horizon_quote_distances,
    symmetric_finite_horizon_quotes,
    execution_intensity,
    simulate_path,
    run_monte_carlo,
    validate_reservation_price_properties,
    validate_lambda_dt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params() -> ASParams:
    return ASParams(gamma=0.1, seed=42)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Tests: reservation_price
# ---------------------------------------------------------------------------

class TestReservationPrice:
    def test_zero_inventory_equals_mid_price(self):
        """r(s, 0, t) = s for any s, t."""
        s = 100.0
        r = reservation_price(s, 0, gamma=0.1, sigma=2.0, tau=0.5)
        assert abs(r - s) < 1e-12, f"Expected r=s={s}, got {r}"

    def test_positive_inventory_below_mid_price(self):
        """Positive inventory → r < s."""
        r = reservation_price(100.0, 5, gamma=0.1, sigma=2.0, tau=0.5)
        assert r < 100.0

    def test_negative_inventory_above_mid_price(self):
        """Negative inventory → r > s."""
        r = reservation_price(100.0, -5, gamma=0.1, sigma=2.0, tau=0.5)
        assert r > 100.0

    def test_vanishes_at_terminal_time(self):
        """Adjustment vanishes as tau → 0."""
        r = reservation_price(100.0, 10, gamma=0.1, sigma=2.0, tau=0.0)
        assert abs(r - 100.0) < 1e-12

    def test_linear_in_inventory(self):
        """r is linear in q."""
        s, gamma, sigma, tau = 100.0, 0.1, 2.0, 0.5
        r1 = reservation_price(s, 1, gamma, sigma, tau)
        r2 = reservation_price(s, 2, gamma, sigma, tau)
        r3 = reservation_price(s, 3, gamma, sigma, tau)
        # Differences should be equal
        assert abs((r2 - r1) - (r3 - r2)) < 1e-12

    def test_formula_exact(self):
        """Verify exact formula: r = s - q*gamma*sigma^2*tau."""
        s, q, gamma, sigma, tau = 100.0, 3, 0.1, 2.0, 0.5
        expected = s - q * gamma * sigma ** 2 * tau
        result = reservation_price(s, q, gamma, sigma, tau)
        assert abs(result - expected) < 1e-12

    def test_increases_with_risk_aversion(self):
        """Higher gamma → larger adjustment for same positive inventory."""
        s, q, sigma, tau = 100.0, 5, 2.0, 0.5
        r_low = reservation_price(s, q, gamma=0.01, sigma=sigma, tau=tau)
        r_high = reservation_price(s, q, gamma=0.5, sigma=sigma, tau=tau)
        # Both below s, but r_high should be further below
        assert r_high < r_low


# ---------------------------------------------------------------------------
# Tests: optimal_half_spread_base
# ---------------------------------------------------------------------------

class TestHalfSpreadBase:
    def test_positive(self):
        """Half-spread base should be positive."""
        c = optimal_half_spread_base(gamma=0.1, k=1.5)
        assert c > 0

    def test_formula_exact(self):
        """Verify exact formula: c = (1/gamma)*ln(1+gamma/k)."""
        gamma, k = 0.1, 1.5
        expected = (1.0 / gamma) * np.log(1.0 + gamma / k)
        result = optimal_half_spread_base(gamma, k)
        assert abs(result - expected) < 1e-12

    def test_decreases_with_gamma(self):
        """Higher gamma → smaller constant half-spread (for fixed k)."""
        c_low = optimal_half_spread_base(gamma=0.01, k=1.5)
        c_high = optimal_half_spread_base(gamma=0.5, k=1.5)
        assert c_low > c_high

    def test_limit_as_gamma_to_zero(self):
        """As gamma → 0, c → 1/k (L'Hopital)."""
        gamma_small = 1e-8
        c = optimal_half_spread_base(gamma_small, k=1.5)
        expected_limit = 1.0 / 1.5
        assert abs(c - expected_limit) < 1e-5


# ---------------------------------------------------------------------------
# Tests: finite_horizon_quote_distances
# ---------------------------------------------------------------------------

class TestFiniteHorizonQuoteDistances:
    def test_zero_inventory_symmetric(self):
        """At q=0, delta_a = delta_b (symmetric quotes)."""
        da, db = finite_horizon_quote_distances(0, gamma=0.1, sigma=2.0, k=1.5, tau=0.5)
        assert abs(da - db) < 1e-12

    def test_positive_inventory_ask_closer(self):
        """
        Positive inventory → ask closer to mid-price (delta_a < delta_b).
        With positive inventory the market maker wants to sell, so the ask
        is posted closer to the mid-price to attract buyers.
        Formula: delta_a = c + (1-2q)*adj/2  →  smaller for q>0.
        """
        da, db = finite_horizon_quote_distances(5, gamma=0.1, sigma=2.0, k=1.5, tau=0.5)
        assert da < db

    def test_negative_inventory_bid_closer(self):
        """
        Negative inventory → bid closer to mid-price (delta_b < delta_a).
        With negative inventory the market maker wants to buy, so the bid
        is posted closer to the mid-price to attract sellers.
        Formula: delta_b = c + (1+2q)*adj/2  →  smaller for q<0.
        """
        da, db = finite_horizon_quote_distances(-5, gamma=0.1, sigma=2.0, k=1.5, tau=0.5)
        assert db < da

    def test_total_spread_formula(self):
        """Total spread = 2*c + gamma*sigma^2*tau (finite-horizon formula)."""
        q, gamma, sigma, k, tau = 0, 0.1, 2.0, 1.5, 0.5
        da, db = finite_horizon_quote_distances(q, gamma, sigma, k, tau)
        total_spread = da + db
        expected = 2 * optimal_half_spread_base(gamma, k) + gamma * sigma ** 2 * tau
        assert abs(total_spread - expected) < 1e-12

    def test_vanishes_at_terminal_time(self):
        """At tau=0, quote distances reduce to constant half-spread."""
        q, gamma, sigma, k = 5, 0.1, 2.0, 1.5
        da, db = finite_horizon_quote_distances(q, gamma, sigma, k, tau=0.0)
        c = optimal_half_spread_base(gamma, k)
        assert abs(da - c) < 1e-12
        assert abs(db - c) < 1e-12


# ---------------------------------------------------------------------------
# Tests: execution_intensity
# ---------------------------------------------------------------------------

class TestExecutionIntensity:
    def test_positive(self):
        """Intensity should be positive."""
        lam = execution_intensity(0.5, A=140.0, k=1.5)
        assert lam > 0

    def test_decreasing_in_delta(self):
        """Intensity decreases as quote distance increases."""
        lam1 = execution_intensity(0.1, A=140.0, k=1.5)
        lam2 = execution_intensity(0.5, A=140.0, k=1.5)
        lam3 = execution_intensity(1.0, A=140.0, k=1.5)
        assert lam1 > lam2 > lam3

    def test_formula_exact(self):
        """Verify exact formula: lambda = A*exp(-k*delta)."""
        delta, A, k = 0.5, 140.0, 1.5
        expected = A * np.exp(-k * delta)
        result = execution_intensity(delta, A, k)
        assert abs(result - expected) < 1e-12

    def test_lambda_dt_below_one(self):
        """lambda*dt < 1 for Bernoulli approximation validity."""
        params = ASParams()
        lam_max = params.A  # worst case: delta=0
        assert lam_max * params.dt < 1.0


# ---------------------------------------------------------------------------
# Tests: simulate_path
# ---------------------------------------------------------------------------

class TestSimulatePath:
    def test_inventory_strategy_runs(self, default_params, rng):
        """Inventory strategy simulation completes without error."""
        result = simulate_path(default_params, "inventory", rng)
        assert "terminal_profit" in result
        assert "terminal_inventory" in result

    def test_symmetric_strategy_runs(self, default_params, rng):
        """Symmetric strategy simulation completes without error."""
        result = simulate_path(default_params, "symmetric", rng)
        assert "terminal_profit" in result

    def test_output_arrays_correct_length(self, default_params, rng):
        """Output arrays have correct length (n_steps+1)."""
        result = simulate_path(default_params, "inventory", rng)
        n = default_params.n_steps
        assert len(result["time"]) == n + 1
        assert len(result["s"]) == n + 1
        assert len(result["q"]) == n + 1
        assert len(result["ask_fill"]) == n
        assert len(result["bid_fill"]) == n

    def test_initial_state_correct(self, default_params, rng):
        """Initial state matches parameters."""
        result = simulate_path(default_params, "inventory", rng)
        assert result["s"][0] == default_params.s0
        assert result["x"][0] == default_params.x0
        assert result["q"][0] == default_params.q0

    def test_terminal_profit_formula(self, default_params, rng):
        """Terminal profit = X_T + q_T * S_T."""
        result = simulate_path(default_params, "inventory", rng)
        expected = result["x"][-1] + result["q"][-1] * result["s"][-1]
        assert abs(result["terminal_profit"] - expected) < 1e-10

    def test_invalid_strategy_raises(self, default_params, rng):
        """Invalid strategy name raises AssertionError."""
        with pytest.raises(AssertionError):
            simulate_path(default_params, "invalid_strategy", rng)

    def test_inventory_changes_by_one(self, default_params, rng):
        """Inventory changes by exactly ±1 per fill event."""
        result = simulate_path(default_params, "inventory", rng)
        q = result["q"]
        ask_fills = result["ask_fill"]
        bid_fills = result["bid_fill"]
        for i in range(default_params.n_steps):
            expected_dq = bid_fills[i] - ask_fills[i]
            actual_dq = q[i + 1] - q[i]
            assert actual_dq == expected_dq, \
                f"Step {i}: expected dq={expected_dq}, got {actual_dq}"

    def test_cash_updates_correctly(self, default_params, rng):
        """Cash updates: +p_ask on ask fill, -p_bid on bid fill."""
        result = simulate_path(default_params, "inventory", rng)
        x = result["x"]
        p_ask = result["p_ask"]
        p_bid = result["p_bid"]
        ask_fills = result["ask_fill"]
        bid_fills = result["bid_fill"]
        for i in range(default_params.n_steps):
            expected_dx = ask_fills[i] * p_ask[i] - bid_fills[i] * p_bid[i]
            actual_dx = x[i + 1] - x[i]
            assert abs(actual_dx - expected_dx) < 1e-10, \
                f"Step {i}: expected dx={expected_dx:.4f}, got {actual_dx:.4f}"

    def test_price_moves_by_sigma_sqrt_dt(self, default_params, rng):
        """Price moves by exactly ±sigma*sqrt(dt) at each step."""
        result = simulate_path(default_params, "inventory", rng)
        s = result["s"]
        sigma = default_params.sigma
        dt = default_params.dt
        expected_move = sigma * np.sqrt(dt)
        for i in range(default_params.n_steps):
            actual_move = abs(s[i + 1] - s[i])
            assert abs(actual_move - expected_move) < 1e-10, \
                f"Step {i}: expected |ds|={expected_move:.6f}, got {actual_move:.6f}"


# ---------------------------------------------------------------------------
# Tests: run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def test_returns_correct_keys(self, default_params):
        """Monte Carlo result contains all expected keys."""
        result = run_monte_carlo(default_params, "inventory")
        for key in ["profits", "inventories", "mean_profit", "std_profit",
                    "mean_inventory", "std_inventory", "sample_path"]:
            assert key in result

    def test_profits_array_length(self, default_params):
        """Profits array has length n_paths."""
        result = run_monte_carlo(default_params, "inventory")
        assert len(result["profits"]) == default_params.n_paths

    def test_std_profit_positive(self, default_params):
        """Standard deviation of profit should be positive."""
        result = run_monte_carlo(default_params, "inventory")
        assert result["std_profit"] > 0

    def test_reproducibility(self, default_params):
        """Same seed produces identical results."""
        r1 = run_monte_carlo(default_params, "inventory")
        r2 = run_monte_carlo(default_params, "inventory")
        np.testing.assert_array_equal(r1["profits"], r2["profits"])

    def test_inventory_strategy_lower_std_than_symmetric(self):
        """
        Inventory strategy should have lower std of terminal inventory
        than symmetric strategy (key paper result).
        """
        params = ASParams(gamma=0.1, seed=42)
        inv = run_monte_carlo(params, "inventory")
        sym = run_monte_carlo(params, "symmetric")
        assert inv["std_inventory"] < sym["std_inventory"], (
            f"Inventory std_inventory={inv['std_inventory']:.2f} should be < "
            f"symmetric std_inventory={sym['std_inventory']:.2f}"
        )

    def test_low_gamma_strategies_similar(self):
        """
        For very low gamma (0.01), inventory and symmetric strategies
        should produce similar results (paper's convergence property).
        """
        params = ASParams(gamma=0.01, seed=42)
        inv = run_monte_carlo(params, "inventory")
        sym = run_monte_carlo(params, "symmetric")
        # Mean profits should be close
        assert abs(inv["mean_profit"] - sym["mean_profit"]) < 5.0, (
            f"For gamma=0.01, strategies should be similar. "
            f"inv={inv['mean_profit']:.2f}, sym={sym['mean_profit']:.2f}"
        )

    def test_high_gamma_inventory_more_conservative(self):
        """
        For high gamma (0.5), inventory strategy should be more conservative
        (lower std_profit) than symmetric.
        """
        params = ASParams(gamma=0.5, seed=42)
        inv = run_monte_carlo(params, "inventory")
        sym = run_monte_carlo(params, "symmetric")
        assert inv["std_profit"] < sym["std_profit"], (
            f"For gamma=0.5, inventory should have lower std_profit. "
            f"inv={inv['std_profit']:.2f}, sym={sym['std_profit']:.2f}"
        )


# ---------------------------------------------------------------------------
# Tests: validation functions
# ---------------------------------------------------------------------------

class TestValidationFunctions:
    def test_validate_reservation_price_properties(self, default_params):
        """Reservation price validation passes for default params."""
        validate_reservation_price_properties(default_params)  # should not raise

    def test_validate_lambda_dt(self, default_params):
        """Lambda*dt validation passes for default params."""
        validate_lambda_dt(default_params)  # should not raise

    def test_validate_lambda_dt_fails_for_large_A(self):
        """Lambda*dt validation fails when A is too large."""
        params = ASParams(A=1000.0, dt=0.005)  # 1000*0.005 = 5 > 1
        with pytest.raises(AssertionError):
            validate_lambda_dt(params)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_step_simulation(self):
        """Simulation with a single time step completes correctly."""
        params = ASParams(T=0.005, dt=0.005, n_paths=10, seed=0)
        result = run_monte_carlo(params, "inventory")
        assert len(result["profits"]) == 10

    def test_zero_gamma_raises_or_handles(self):
        """gamma=0 would cause division by zero; test that it raises."""
        with pytest.raises((ZeroDivisionError, ValueError, AssertionError)):
            params = ASParams(gamma=0.0)
            run_monte_carlo(params, "inventory")

    def test_extreme_positive_inventory_initial(self):
        """Large initial inventory still produces valid simulation."""
        params = ASParams(q0=50, gamma=0.1, seed=0, n_paths=10)
        result = run_monte_carlo(params, "inventory")
        assert np.all(np.isfinite(result["profits"]))

    def test_extreme_negative_inventory_initial(self):
        """Large negative initial inventory still produces valid simulation."""
        params = ASParams(q0=-50, gamma=0.1, seed=0, n_paths=10)
        result = run_monte_carlo(params, "inventory")
        assert np.all(np.isfinite(result["profits"]))

    def test_profits_are_finite(self, default_params):
        """All terminal profits should be finite."""
        result = run_monte_carlo(default_params, "inventory")
        assert np.all(np.isfinite(result["profits"]))

    def test_inventories_are_integers(self, default_params):
        """Terminal inventories should be integers."""
        result = run_monte_carlo(default_params, "inventory")
        assert result["inventories"].dtype in [np.int32, np.int64, int]
