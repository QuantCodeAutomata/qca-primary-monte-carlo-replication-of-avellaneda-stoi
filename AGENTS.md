# Avellaneda-Stoikov Monte Carlo Replication — Agent Memory

## Project Purpose
Monte Carlo replication of the Avellaneda & Stoikov (2008) market-making model.
Compares Strategy A (inventory-aware, reservation-price centred) against Strategy B
(symmetric, mid-price centred).

## Directory Layout
```
src/          # Core shared library — config, simulator, metrics
exp/          # Experiment scripts and strategies (run_exp1.py, run_exp2.py, strategies_*.py, plotting.py)
tests/        # Unit + integration tests (test_simulator.py, test_strategies.py)
results/      # Output CSVs and figures (git-tracked via .gitkeep placeholder)
data/         # Input data (git-tracked via .gitkeep placeholder)
```

## Core Module Contracts

### src/config.py
- `SimParams` dataclass: all model parameters (S0, T, sigma, dt, A, k, q0, X0, n_paths, master_seed, gammas)
- `DEFAULT_PARAMS = SimParams()` — single source of truth used in all experiments
- `SimParams.N` property: `round(T / dt)` = 200 steps with defaults
- `SimParams.time_grid()`: returns `np.arange(N) * dt`

### src/simulator.py
- `StrategyFn` = `Callable[[float, int, float, float, SimParams], Tuple[float, float]]`
  Signature: `(S, q, t, gamma, params) -> (delta_a, delta_b)`
- `simulate_path(strategy_fn, gamma, params, rng, record_path=False) -> dict`
  Keys: terminal_profit, terminal_inventory, terminal_S, terminal_X,
        negative_delta_count, bernoulli_violation_count, path_data (optional)
- `simulate_monte_carlo(strategy_fn, strategy_name, gamma, params, seed, rep_path_index=0)`
  Returns `(results_df, rep_path_df)` — results_df has 1000 rows (one per path)

### src/metrics.py
- `compute_summary_stats(results_df)` — groups by (gamma, strategy), returns mean/std/pct columns
- `validate_terminal_profit(results_df)` — checks all profits finite
- `validate_lambda_monotonic(params, gamma)` — checks λ(δ) decreasing
- `validate_reservation_price(params, gamma)` — checks r=S when q=0, r→S as t→T
- `compute_full_spread_formula(gamma, sigma, k, T_minus_t)` — eq (3.18) of paper
- `compute_constant_spread(gamma, k)` — constant term `(2/γ) ln(1 + γ/k)`
- `compute_initial_full_spread(gamma, sigma, k, T)` — full spread at t=0
- `print_diagnostics(results_df, summary_df, params)` — prints formatted summary

## Model Mechanics (in simulate_path)
1. Strategy called → `(delta_a, delta_b)` quote distances from mid
2. Intensities: `λ_a = A exp(-k δ_a)`, arrival probs: `p = λ dt` (Bernoulli)
3. Fills happen at `p_a = S + δ_a` (ask) and `p_b = S - δ_b` (bid)
4. Inventory and cash updated; then mid-price moves `±σ√dt` with prob 0.5 each
5. Terminal profit: `X_T + q_T * S_T`

## Key Parameter Values (defaults)
- S0=100, T=1.0, σ=2.0, dt=0.005 → N=200 steps
- A=140, k=1.5
- n_paths=1000, master_seed=42
- gammas=[0.01, 0.1, 0.5]

## Verified Behaviour
- No Bernoulli violations with default params (λ_max·dt < 1)
- validate_lambda_monotonic → True
- validate_reservation_price → True
- End-to-end smoke test with symmetric strategy produces finite profits

## exp/plotting.py — Visualization Module
- `matplotlib.use('Agg')` at import time — safe for headless/file-only rendering
- `RESULTS_DIR = Path('results')` created at module load
- `plot_representative_path(rep_path_df, gamma, strategy_name, exp_label, save_path)`
  — 3-panel figure: price+quotes, inventory, quote distances
- `plot_profit_histograms(all_results_df, gamma, exp_label, save_path)`
  — 2-panel figure: terminal profit density, terminal inventory histogram
- `plot_summary_comparison(summary_df, exp_label, save_path)`
  — 2×2 bar chart grid over gammas: mean_profit, std_profit, mean_final_q, std_final_q
- `generate_all_plots(all_results_df, summary_df, rep_paths, exp_label, params)`
  — orchestrates all three plot types; rep_paths keyed by `(gamma, strat_name)`

## exp/main.py — Experiment Runner
- `run_all_experiments(params=None)` — runs Exp1 + Exp2, generates all plots, writes RESULTS.md
- `compare_experiments(summary1, summary2, params)` — prints directional validation table
- `write_results_md(summary1, summary2, params)` — writes `results/RESULTS.md` with markdown tables
- Entry point: `python -m exp.main` or `python exp/main.py`

## tests/ — Test Suite
- `tests/__init__.py` — empty (package marker)
- `tests/test_strategies.py` — 23 tests across 4 classes
  - `TestInventoryFullSpread`: spread formula, maturity, inventory-direction correctness
  - `TestSymmetricFullSpread`: inventory-invariance, symmetry, time-monotonicity
  - `TestReservationPrice`: q=0 case, maturity convergence, sign correctness
  - `TestConstantSpreadStrategies`: positivity, time-invariance, total spread identity
- `tests/test_simulator.py` — 18 tests across 3 classes
  - `TestSimulatePath`: finite profit, required keys, path recording, column coverage
  - `TestSimulateMonteCarlo`: shape, columns, rep path, reproducibility, finite profits
  - `TestIntensityProperties`: λ monotone, Bernoulli prob validity
- `tests/test_metrics.py` — 17 tests across 3 classes
  - `TestComputeSummaryStats`: shape, columns, non-negative std, n_paths
  - `TestValidations`: finite-profit gate, NaN detection, λ-monotonic, reservation price
  - `TestSpreadFormulas`: full-spread at maturity, positivity, constant formula, known values

## Confirmed Experimental Results (run on 2026-08-12)

### Experiment 1 – Full Time-Varying Spread
| gamma | Strategy | std(Profit) | std(q_T) |
|-------|----------|-------------|----------|
| 0.01 | inventory_full | 8.94 | 5.07 |
| 0.01 | symmetric_full | 13.55 | 8.86 |
| 0.1  | inventory_full | 6.38 | 2.94 |
| 0.1  | symmetric_full | 13.32 | 8.46 |
| 0.5  | inventory_full | 5.93 | 1.94 |
| 0.5  | symmetric_full | 11.45 | 7.25 |

### Experiment 2 – Constant Spread (Table-Faithful)
| gamma | constant_spread | Strategy | std(Profit) | std(q_T) |
|-------|-----------------|----------|-------------|----------|
| 0.01 | 1.3289 | inventory_const | 8.91 | 5.08 |
| 0.01 | 1.3289 | symmetric_const | 13.77 | 8.73 |
| 0.1  | 1.2908 | inventory_const | 6.08 | 2.89 |
| 0.1  | 1.2908 | symmetric_const | 13.59 | 8.71 |
| 0.5  | 1.1507 | inventory_const | 5.06 | 1.94 |
| 0.5  | 1.1507 | symmetric_const | 13.53 | 9.41 |

### Key Notes
- All 6 directional hypotheses passed: inventory strategy consistently lower std(Profit) and std(q_T)
- Exp2 constant spread at gamma=0.01 (1.3289) and gamma=0.5 (1.1507) match paper table references exactly
- gamma=0.5 with inventory_full generates Bernoulli violations (prob>1) on nearly all paths — expected at high gamma with large inventory skew pushing delta_a negative; flagged but not clipped per methodology
- pandas DeprecationWarning: use `include_groups=False` in `groupby.apply()` (already applied in src/metrics.py)

## Python Environment
- numpy, pandas, matplotlib, seaborn, pytest, tabulate are all available
- Run from /workspace/project/ so `from src.xxx import yyy` works
- All tests are run with `pytest tests/` from project root
