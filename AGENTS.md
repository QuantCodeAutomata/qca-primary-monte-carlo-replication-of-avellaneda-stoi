# Repository Knowledge — Avellaneda-Stoikov Market Making Replication

## Project Overview
Replication of Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book".
Four experiments: primary MC replication, diagnostic spread-ambiguity variant,
microstructure calibration, and appendix GBM mean-variance extension.

## Key Architecture
- `src/simulation.py`: Core engine — SimParams, simulate_path, run_monte_carlo
- `src/exp1_primary_replication.py`: Finite-horizon quoting rules (Eq. 3.8/3.9)
- `src/exp2_diagnostic_replication.py`: Constant-spread diagnostic variant
- `src/exp3_microstructure_calibration.py`: Analytical intensity derivations
- `src/exp4_appendix_gbm_mv.py`: GBM + mean-variance appendix formulas
- `run_experiments.py`: Main runner, writes results/RESULTS.md
- `tests/`: pytest suite for all modules

## Critical Implementation Details
- Use ARITHMETIC Brownian motion (not GBM) for main experiments
- Binomial price discretisation: s ± sigma*sqrt(dt) with equal probability
- Bid and ask Bernoulli fills are INDEPENDENT within each dt step
- Common random numbers used across strategies (shared price_moves, uniforms)
- x0=0, q0=0, one-unit fills only, zero interest rate
- lambda*dt must be < 1 (validated in validate_parameters)

## Paper Ambiguity (Exp 2)
The paper's tables report spread = (2/gamma)*ln(1+gamma/k) (constant),
but the finite-horizon formula implies spread = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k).
Exp 2 tests the constant-spread interpretation.

## Parameter Values (Paper's Baseline)
s0=100, sigma=2, T=1, dt=0.005, q0=0, x0=0, A=140, k=1.5, n_paths=1000
gammas = [0.1, 0.01, 0.5]

## Testing
Run: `pytest tests/ -v`
All tests use real code paths (no mocks).
Fast tests use SimParams(n_paths=100, seed=42).

## Results
All outputs saved to results/ directory.
RESULTS.md contains all summary statistics and key findings.
