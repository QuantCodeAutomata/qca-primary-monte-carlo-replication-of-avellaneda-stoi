# AGENTS.md — Repository Knowledge Base

## Project Overview

Replication of the Avellaneda-Stoikov (2008) finite-horizon inventory-based
market-making model. Four experiments cover:
1. Primary Monte Carlo replication (finite-horizon ABM + exponential utility)
2. Diagnostic spread-ambiguity variant (constant vs. time-varying spread)
3. Microstructure calibration mapping (power-law orders → execution intensity)
4. Appendix extension (GBM + mean-variance objective)

## Repository Structure

```
project/
├── data/                        # Synthetic data generation (no external data needed)
│   └── generate_synthetic_data.py
├── src/                         # Core implementations
│   ├── avellaneda_stoikov.py    # Main AS model: reservation price, quote distances, MC sim
│   ├── diagnostic_variant.py   # Constant-spread (table-matching) variant
│   ├── microstructure_calibration.py  # Intensity derivations and calibration
│   └── appendix_gbm.py         # GBM mean-variance appendix formulas
├── exp/                         # Experiment runners
│   ├── exp1_primary_replication.py
│   ├── exp2_diagnostic_variant.py
│   ├── exp3_microstructure_calibration.py
│   └── exp4_appendix_gbm.py
├── tests/                       # pytest test suite (105 tests)
│   ├── test_avellaneda_stoikov.py
│   ├── test_diagnostic_variant.py
│   ├── test_microstructure_calibration.py
│   └── test_appendix_gbm.py
├── results/                     # Generated artefacts
│   └── RESULTS.md               # Summary of all findings
├── conftest.py                  # pytest fixtures
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Parameters (Paper-Exact)

| Parameter | Value | Description |
|-----------|-------|-------------|
| s₀        | 100   | Initial mid-price |
| σ         | 2     | ABM volatility |
| T         | 1     | Horizon |
| dt        | 0.005 | Time step (200 steps) |
| q₀        | 0     | Initial inventory |
| x₀        | 0     | Initial cash |
| A         | 140   | Intensity pre-factor |
| k         | 1.5   | Intensity decay rate |
| γ         | {0.1, 0.01, 0.5} | Risk-aversion values |
| N_paths   | 1000  | Monte Carlo paths |
| seed      | 42    | Random seed |

## Core Formulas (src/avellaneda_stoikov.py)

- **Reservation price**: r_t = s_t − q_t·γ·σ²·(T−t)
- **Ask half-spread**: δᵃ = (1/γ)·ln(1+γ/k) + (1−2q)·γ·σ²·(T−t)/2
- **Bid half-spread**: δᵇ = (1/γ)·ln(1+γ/k) + (1+2q)·γ·σ²·(T−t)/2
- **Symmetric spread**: spread_t = γ·σ²·(T−t) + (2/γ)·ln(1+γ/k)
- **Execution intensity**: λ(δ) = A·exp(−k·δ)
- **Bernoulli fill prob**: P(fill) = λ(δ)·dt
- **Terminal profit**: Π_T = X_T + q_T·S_T
- **Mid-price step**: s_{t+dt} = s_t ± σ·√(dt) (equal probability)

## Diagnostic Variant (src/diagnostic_variant.py)

Uses constant half-spread c_γ = (1/γ)·ln(1+γ/k):
- δᵃ = c_γ − q·γ·σ²·(T−t)
- δᵇ = c_γ + q·γ·σ²·(T−t)
- Symmetric: p^a = s + c_γ, p^b = s − c_γ

## Microstructure Calibration (src/microstructure_calibration.py)

- Log-impact: λ(δ) = A·exp(−k·δ), A = Λ/α, k = α·K
- Power-law impact: λ(δ) = B·δ^{−α/β}
- Representative α: US stocks 1.53, NASDAQ 1.40, Paris 1.50
- Representative β: square-root 0.50, empirical 0.76

## Appendix GBM (src/appendix_gbm.py)

- Value function: V = x + qs + (γq²s²/2)·(exp(σ²(T−t))−1)
- Reservation ask: Rᵃ = s + ((1−2q)/2)·γ·s²·(exp(σ²(T−t))−1)
- Reservation bid: Rᵇ = s + ((−1−2q)/2)·γ·s²·(exp(σ²(T−t))−1)
- Reservation midpoint: R = s − q·γ·s²·(exp(σ²(T−t))−1)

## Running Experiments

```bash
# Run all experiments
python -m exp.exp1_primary_replication
python -m exp.exp2_diagnostic_variant
python -m exp.exp3_microstructure_calibration
python -m exp.exp4_appendix_gbm

# Run tests
pytest tests/ -v
```

## Implementation Notes

- Bid and ask Bernoulli fills are **independent** within each dt step
  (simultaneous fills possible but rare; documented as a methodological choice)
- Negative quote distances are **not clipped** (paper does not specify clipping)
- Common random numbers: price paths and execution uniforms use the same seed
  across inventory and symmetric strategies for fair comparison
- ABM only (no GBM in main simulation); GBM appears only in Exp 4 appendix
- Zero interest rate, no transaction costs, no self-impact, no tick size

## Libraries Used

- **numpy**: Monte Carlo simulation, random number generation
- **pandas**: Data aggregation and CSV output
- **matplotlib**: All visualisations (Agg backend for headless rendering)
- **scipy.stats**: Statistical utilities in tests
- No portfolio-optimisation libraries needed (custom AS implementation)

## Test Suite

105 tests across 4 files. Run with `pytest tests/ -v`.
Key test categories:
- Reservation price properties (sign, magnitude, boundary conditions)
- Quote distance formulas (symmetry, positivity, gamma→0 limit)
- Execution intensity (monotonicity, Bernoulli probability bounds)
- Monte Carlo simulation (shape, terminal profit computation)
- Diagnostic variant (constant spread, inventory centering)
- Microstructure calibration (exponential/power-law derivations)
- Appendix GBM (value function, reservation price properties)
