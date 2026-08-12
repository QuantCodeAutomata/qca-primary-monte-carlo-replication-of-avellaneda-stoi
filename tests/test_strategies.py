"""Unit tests for Avellaneda-Stoikov quoting strategies.

Tests validate:
- Mathematical properties of quote distance formulas
- Edge cases (q=0, t=T, extreme inventory)
- Strategy interface compliance
- Directional correctness of inventory adjustments
"""
import pytest
import numpy as np
from src.config import SimParams
from exp.strategies_exp1 import inventory_full_spread, symmetric_full_spread, compute_reservation_price
from exp.strategies_exp2 import inventory_const_spread, symmetric_const_spread, compute_half_spread


@pytest.fixture
def params():
    return SimParams()


class TestInventoryFullSpread:
    """Tests for Experiment 1, Strategy A: inventory-aware full spread."""

    def test_zero_inventory_equals_half_spread(self, params):
        """When q=0, delta_a and delta_b should both equal the symmetric half-spread."""
        gamma, S, t = 0.1, 100.0, 0.5
        delta_a, delta_b = inventory_full_spread(S, 0, t, gamma, params)
        delta_a_sym, delta_b_sym = symmetric_full_spread(S, 0, t, gamma, params)
        assert np.isclose(delta_a, delta_b_sym, rtol=1e-10)
        assert np.isclose(delta_b, delta_a_sym, rtol=1e-10)
        assert np.isclose(delta_a, delta_b, rtol=1e-10)

    def test_positive_inventory_widens_ask_tightens_bid(self, params):
        """When q>0, ask distance shrinks and bid distance widens to reduce inventory."""
        gamma, S, t, q = 0.1, 100.0, 0.5, 3
        delta_a_pos, delta_b_pos = inventory_full_spread(S, q, t, gamma, params)
        delta_a_zero, delta_b_zero = inventory_full_spread(S, 0, t, gamma, params)
        # Positive inventory: ask should be closer to mid (smaller delta_a)
        assert delta_a_pos < delta_a_zero, 'Positive q should reduce ask distance'
        # Bid should be further from mid (larger delta_b)
        assert delta_b_pos > delta_b_zero, 'Positive q should increase bid distance'

    def test_negative_inventory_widens_bid_tightens_ask(self, params):
        """When q<0, bid distance shrinks and ask distance widens."""
        gamma, S, t, q = 0.1, 100.0, 0.5, -3
        delta_a_neg, delta_b_neg = inventory_full_spread(S, q, t, gamma, params)
        delta_a_zero, delta_b_zero = inventory_full_spread(S, 0, t, gamma, params)
        assert delta_a_neg > delta_a_zero, 'Negative q should increase ask distance'
        assert delta_b_neg < delta_b_zero, 'Negative q should reduce bid distance'

    def test_spread_formula_matches_total(self, params):
        """Total spread delta_a + delta_b should equal the full formula."""
        gamma, S, t, q = 0.1, 100.0, 0.3, 2
        delta_a, delta_b = inventory_full_spread(S, q, t, gamma, params)
        total = delta_a + delta_b
        tau = params.T - t
        expected = gamma * params.sigma**2 * tau + (2.0/gamma) * np.log1p(gamma/params.k)
        assert np.isclose(total, expected, rtol=1e-10), f'Total spread {total} != expected {expected}'

    def test_spread_at_maturity(self, params):
        """At t=T, the time-varying term vanishes; only the constant term remains."""
        gamma, S, t = 0.1, 100.0, params.T
        delta_a, delta_b = inventory_full_spread(S, 5, t, gamma, params)
        const_term = np.log1p(gamma / params.k) / gamma
        # Both should equal the constant term
        assert np.isclose(delta_a, const_term, rtol=1e-10)
        assert np.isclose(delta_b, const_term, rtol=1e-10)

    def test_log_term_numerical_stability(self, params):
        """Test that log1p is used for stability (gamma/k near zero)."""
        # Very small gamma - log1p(gamma/k) should be more stable than log(1 + gamma/k)
        gamma = 0.001
        h_log1p = np.log1p(gamma / params.k) / gamma
        h_direct = np.log(1 + gamma / params.k) / gamma
        assert np.isclose(h_log1p, h_direct, rtol=1e-6)


class TestSymmetricFullSpread:
    """Tests for Experiment 1, Strategy B: symmetric full spread."""

    def test_ignores_inventory(self, params):
        """Symmetric strategy gives same quotes regardless of inventory."""
        gamma, S, t = 0.1, 100.0, 0.5
        da0, db0 = symmetric_full_spread(S, 0, t, gamma, params)
        da5, db5 = symmetric_full_spread(S, 5, t, gamma, params)
        da_neg, db_neg = symmetric_full_spread(S, -5, t, gamma, params)
        assert np.isclose(da0, da5) and np.isclose(da0, da_neg)
        assert np.isclose(db0, db5) and np.isclose(db0, db_neg)

    def test_symmetric_equal_distances(self, params):
        """Ask and bid distances are equal."""
        gamma, S, t = 0.5, 100.0, 0.2
        da, db = symmetric_full_spread(S, 3, t, gamma, params)
        assert np.isclose(da, db, rtol=1e-12)

    def test_spread_decreases_with_time(self, params):
        """As t increases toward T, full spread decreases."""
        gamma, S, q = 0.1, 100.0, 0
        times = np.linspace(0, params.T - params.dt, 10)
        spreads = [sum(symmetric_full_spread(S, q, t, gamma, params)) for t in times]
        diffs = np.diff(spreads)
        assert (diffs < 0).all(), 'Full spread should be monotonically decreasing over time'


class TestReservationPrice:
    """Tests for the reservation price function."""

    def test_zero_inventory_equals_mid(self, params):
        """r_t = S_t when q_t = 0."""
        for gamma in params.gammas:
            for t in [0.0, 0.5, 0.9]:
                r = compute_reservation_price(100.0, 0, t, gamma, params)
                assert np.isclose(r, 100.0), f'r != S when q=0 at t={t}, gamma={gamma}'

    def test_converges_to_mid_at_maturity(self, params):
        """r_t -> S_t as t -> T regardless of inventory."""
        t_near_T = params.T - 1e-9
        for gamma in params.gammas:
            r = compute_reservation_price(100.0, 100, t_near_T, gamma, params)
            assert np.isclose(r, 100.0, atol=1e-3)

    def test_positive_inventory_below_mid(self, params):
        """When q>0, reservation price is below mid-price."""
        gamma, S, q, t = 0.1, 100.0, 5, 0.5
        r = compute_reservation_price(S, q, t, gamma, params)
        assert r < S, 'Reservation price should be below mid when q>0'

    def test_negative_inventory_above_mid(self, params):
        """When q<0, reservation price is above mid-price."""
        gamma, S, q, t = 0.1, 100.0, -5, 0.5
        r = compute_reservation_price(S, q, t, gamma, params)
        assert r > S, 'Reservation price should be above mid when q<0'


class TestConstantSpreadStrategies:
    """Tests for Experiment 2 constant-spread strategies."""

    def test_half_spread_positive(self, params):
        """Half-spread must be positive for all gamma scenarios."""
        for gamma in params.gammas:
            h = compute_half_spread(gamma, params.k)
            assert h > 0, f'Half-spread should be positive for gamma={gamma}'

    def test_inventory_const_zero_inventory_equals_symmetric(self, params):
        """Inventory strategy equals symmetric when q=0."""
        for gamma in params.gammas:
            for t in [0.0, 0.5, 1.0]:
                da_inv, db_inv = inventory_const_spread(100.0, 0, t, gamma, params)
                da_sym, db_sym = symmetric_const_spread(100.0, 0, t, gamma, params)
                assert np.isclose(da_inv, da_sym, rtol=1e-10)
                assert np.isclose(db_inv, db_sym, rtol=1e-10)

    def test_constant_spread_is_time_invariant(self, params):
        """Symmetric constant spread does not vary with time."""
        gamma = 0.1
        for t in np.linspace(0, params.T, 10):
            da, db = symmetric_const_spread(100.0, 0, t, gamma, params)
            h = compute_half_spread(gamma, params.k)
            assert np.isclose(da, h, rtol=1e-12)
            assert np.isclose(db, h, rtol=1e-12)

    def test_inventory_const_spread_total_is_constant(self, params):
        """Total spread for inventory_const_spread should equal 2h (constant)."""
        gamma, S = 0.1, 100.0
        h = compute_half_spread(gamma, params.k)
        for t in np.linspace(0, params.T * 0.9, 5):
            for q in [-5, 0, 5]:
                da, db = inventory_const_spread(S, q, t, gamma, params)
                total = da + db
                assert np.isclose(total, 2 * h, rtol=1e-10), \
                    f'Total spread {total} != 2h={2*h} for q={q}, t={t}'

    def test_symmetric_const_ignores_inventory(self, params):
        """Symmetric strategy gives same quotes for all inventories."""
        gamma, S, t = 0.5, 100.0, 0.5
        for q in [-10, 0, 10]:
            da, db = symmetric_const_spread(S, q, t, gamma, params)
            h = compute_half_spread(gamma, params.k)
            assert np.isclose(da, h) and np.isclose(db, h)
