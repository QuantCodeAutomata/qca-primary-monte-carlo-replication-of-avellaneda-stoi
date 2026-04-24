# Avellaneda-Stoikov Market Making Replication

A complete Python implementation replicating the numerical experiments from:

> Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.

## Overview

This repository implements four experiments:

| Experiment | Description |
|-----------|-------------|
| **Exp 1** | Primary Monte Carlo replication of finite-horizon inventory-based market making |
| **Exp 2** | Diagnostic replication addressing the paper's spread-ambiguity inconsistency |
| **Exp 3** | Microstructure-to-intensity calibration mapping (analytical derivation) |
| **Exp 4** | Appendix extension: mean-variance market making under GBM |

## Model Summary

The Avellaneda-Stoikov model describes an optimal market-making strategy where:

- Mid-price follows **arithmetic Brownian motion**: `dS = σ dW`
- Execution intensities are **exponential**: `λ(δ) = A·exp(-k·δ)`
- The dealer maximises **CARA utility** (exponential utility)
- Optimal quotes are centered around an **inventory-adjusted reservation price**:
  `r(s, q, t) = s - q·γ·σ²·(T-t)`

### Finite-Horizon Quoting Rules (Eq. 3.8/3.9)

```
δᵃ = (1/γ)·ln(1+γ/k) + ((1-2q)·γ·σ²·(T-t))/2
δᵇ = (1/γ)·ln(1+γ/k) + ((1+2q)·γ·σ²·(T-t))/2
```

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| s₀ | 100 | Initial mid-price |
| σ | 2 | Volatility |
| T | 1 | Horizon |
| dt | 0.005 | Time step (200 steps) |
| q₀ | 0 | Initial inventory |
| x₀ | 0 | Initial cash |
| A | 140 | Intensity scale |
| k | 1.5 | Intensity decay |
| N | 1000 | Monte Carlo paths |
| γ | {0.1, 0.01, 0.5} | Risk-aversion values |

## Repository Structure

```
├── src/
│   ├── simulation.py              # Core simulation engine
│   ├── exp1_primary_replication.py
│   ├── exp2_diagnostic_replication.py
│   ├── exp3_microstructure_calibration.py
│   └── exp4_appendix_gbm_mv.py
├── tests/
│   ├── test_simulation.py
│   └── test_experiments.py
├── results/
│   └── RESULTS.md                 # All experiment results
├── run_experiments.py             # Main runner
└── requirements.txt
```

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python run_experiments.py

# Run tests
pytest tests/ -v
```

## Key Findings

1. **Inventory strategy reduces risk**: Lower profit std and inventory std vs symmetric benchmark
2. **High γ effect**: For γ=0.5, inventory strategy is much more conservative
3. **Low γ convergence**: For γ=0.01, both strategies behave similarly
4. **Spread ambiguity**: Paper's tables may use constant spread `(2/γ)·ln(1+γ/k)` rather than time-varying formula
5. **Microstructure link**: Exponential intensity emerges naturally from power-law order sizes + log impact

## Implementation Notes

- **Arithmetic BM only** (not GBM) for main experiments
- **Binomial discretisation**: `s_{t+dt} = s_t ± σ√dt` with equal probability
- **Independent Bernoulli fills**: bid and ask fills simulated independently per step
- **Common random numbers**: shared across strategies for fair comparison
- **Zero interest rate**, no transaction costs, no market impact
- **One-unit fills** only
