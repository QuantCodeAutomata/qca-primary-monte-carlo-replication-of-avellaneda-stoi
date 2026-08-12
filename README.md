# Avellaneda-Stoikov Monte Carlo Replication

A Monte Carlo replication of the Avellaneda & Stoikov (2008) market-making model,
implemented in Python with a modular architecture that cleanly separates simulation
mechanics from quoting strategies.

## Project Structure

```
project/
├── src/                  # Shared core library
│   ├── __init__.py
│   ├── config.py         # SimParams dataclass and DEFAULT_PARAMS
│   ├── simulator.py      # Generic Monte Carlo engine (simulate_path, simulate_monte_carlo)
│   └── metrics.py        # Summary statistics, validation checks, diagnostics
├── exp/                  # Experiment scripts (Strategy A and Strategy B)
├── tests/                # Unit and integration tests
├── results/              # Output artefacts (CSVs, figures)
├── data/                 # Input data (if any)
└── README.md
```

## Model Overview

The Avellaneda-Stoikov (2008) model describes a market maker who continuously
posts bid and ask limit orders around the mid-price of a risky asset. The
mid-price follows a random walk and order arrivals follow a Poisson process
whose intensity decays exponentially with the distance of the quote from mid.

**Key model equations:**

- Mid-price dynamics: `S_{n+1} = S_n ± σ√dt` (binomial approximation)
- Arrival intensity: `λ(δ) = A · exp(−k · δ)`
- Reservation price: `r_t = S_t − q · γ · σ² · (T − t)`
- Optimal spread: `ψ* = γ σ² (T−t) + (2/γ) ln(1 + γ/k)`

**Strategies compared:**

| Strategy | Description |
|----------|-------------|
| A | Inventory-aware: quotes centred on the reservation price `r_t` with the full time-varying spread |
| B | Symmetric: quotes centred on mid-price `S_t` with a constant half-spread |

## Parameters (defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| S0 | 100.0 | Initial mid-price |
| T | 1.0 | Trading horizon |
| σ | 2.0 | Mid-price volatility |
| dt | 0.005 | Time step size |
| A | 140.0 | Baseline order-arrival intensity |
| k | 1.5 | Intensity decay coefficient |
| n_paths | 1000 | Monte Carlo paths per scenario |
| γ | 0.01, 0.1, 0.5 | Risk-aversion scenarios |

## Quick Start

```python
from src.config import DEFAULT_PARAMS
from src.simulator import simulate_monte_carlo
from src.metrics import compute_summary_stats, print_diagnostics

# Define a simple symmetric strategy
def symmetric_strategy(S, q, t, gamma, params):
    half_spread = 0.5  # fixed half-spread
    return half_spread, half_spread

params = DEFAULT_PARAMS
results_df, rep_path_df = simulate_monte_carlo(
    strategy_fn=symmetric_strategy,
    strategy_name="symmetric",
    gamma=0.1,
    params=params,
    seed=42,
)

summary_df = compute_summary_stats(results_df)
print_diagnostics(results_df, summary_df, params)
```

## References

Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
*Quantitative Finance*, 8(3), 217–224.
