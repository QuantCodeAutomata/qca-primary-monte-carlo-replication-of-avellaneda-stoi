# Avellaneda-Stoikov Market Making: Monte Carlo Replication

Replication of the Avellaneda & Stoikov (2008) paper *"High-frequency trading in a limit order book"*, implementing the finite-horizon inventory-based market-making model and several diagnostic/extension experiments.

## Overview

This repository contains four experiments:

| Experiment | Description |
|---|---|
| **exp_1** | Primary Monte Carlo replication of the finite-horizon AS model (ABM + exponential utility) |
| **exp_2** | Diagnostic replication addressing the paper's spread-ambiguity (constant vs. time-varying spread) |
| **exp_3** | Microstructure calibration: power-law order sizes → exponential/power-law execution intensities |
| **exp_4** | Appendix extension: GBM + mean-variance objective, reservation-price formulas |

## Repository Structure

```
repo/
├── data/                        # Synthetic data generation
│   └── generate_synthetic_data.py
├── src/                         # Core implementations
│   ├── avellaneda_stoikov.py    # Main AS finite-horizon model
│   ├── diagnostic_variant.py   # Constant-spread diagnostic variant
│   ├── microstructure_calibration.py  # Intensity calibration
│   └── appendix_gbm.py         # GBM mean-variance appendix
├── exp/                         # Experiment runners
│   ├── exp1_primary_replication.py
│   ├── exp2_diagnostic_variant.py
│   ├── exp3_microstructure_calibration.py
│   └── exp4_appendix_gbm.py
├── tests/                       # pytest test suite
│   ├── test_avellaneda_stoikov.py
│   ├── test_diagnostic_variant.py
│   ├── test_microstructure_calibration.py
│   └── test_appendix_gbm.py
├── results/                     # Output artifacts
│   └── RESULTS.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Model Parameters

| Parameter | Value | Description |
|---|---|---|
| `s0` | 100 | Initial mid-price |
| `sigma` | 2 | Volatility |
| `T` | 1 | Horizon |
| `dt` | 0.005 | Time step (200 steps) |
| `q0` | 0 | Initial inventory |
| `x0` | 0 | Initial cash |
| `A` | 140 | Intensity scale |
| `k` | 1.5 | Intensity decay |
| `gamma` | {0.1, 0.01, 0.5} | Risk-aversion values |
| `n_paths` | 1000 | Monte Carlo paths |

## Key Formulas

**Reservation price** (main model, ABM):
```
r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
```

**Finite-horizon quote distances**:
```
delta_a = (1/gamma)*ln(1 + gamma/k) + (1 - 2q)*gamma*sigma^2*(T-t)/2
delta_b = (1/gamma)*ln(1 + gamma/k) + (1 + 2q)*gamma*sigma^2*(T-t)/2
```

**Execution intensity**:
```
lambda(delta) = A * exp(-k * delta)
```

**Terminal profit**:
```
Pi_T = X_T + q_T * S_T
```

## Usage

### Run all experiments

```bash
python -m exp.exp1_primary_replication
python -m exp.exp2_diagnostic_variant
python -m exp.exp3_microstructure_calibration
python -m exp.exp4_appendix_gbm
```

### Run tests

```bash
pytest tests/ -v
```

### Generate synthetic data

```bash
python data/generate_synthetic_data.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Results

See `results/RESULTS.md` for a summary of all experiment outputs, tables, and figures.

## Reference

Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
