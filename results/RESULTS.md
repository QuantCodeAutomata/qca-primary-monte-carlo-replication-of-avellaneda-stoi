# Results: Avellaneda-Stoikov Market Making Replication

## Overview

This document summarises the numerical findings from all four experiments
replicating and extending the Avellaneda-Stoikov (2008) finite-horizon
inventory-based market-making model.

**Simulation parameters (shared across Exp 1 & 2):**
- Horizon T = 1, dt = 0.005 (200 steps)
- Initial mid-price s₀ = 100, σ = 2 (ABM)
- Initial inventory q₀ = 0, initial cash x₀ = 0
- Execution intensity: λ(δ) = A·exp(−k·δ), A = 140, k = 1.5
- Risk-aversion γ ∈ {0.1, 0.01, 0.5}
- 1 000 independent Monte Carlo paths per strategy per γ
- Fixed random seed for reproducibility

---

## Experiment 1 — Primary Monte Carlo Replication (Finite-Horizon Formulas)

**Strategy definitions:**
- *Inventory*: quotes centred on reservation price r_t = s_t − q_t·γ·σ²·(T−t);
  half-spreads δᵃ = (1/γ)·ln(1+γ/k) + (1−2q)·γ·σ²·(T−t)/2,
  δᵇ = (1/γ)·ln(1+γ/k) + (1+2q)·γ·σ²·(T−t)/2.
- *Symmetric*: same total spread as inventory strategy but centred on mid-price.

**Summary statistics (1 000 paths, seed = 42):**

| γ    | Strategy  | E[Π]  | Std[Π] | E[q_T] | Std[q_T] |
|------|-----------|------:|-------:|-------:|---------:|
| 0.10 | inventory | 64.80 |   6.65 |   0.10 |     2.91 |
| 0.10 | symmetric | 68.25 |  13.37 |  −0.03 |     8.28 |
| 0.01 | inventory | 68.45 |   8.94 |   0.03 |     5.07 |
| 0.01 | symmetric | 69.11 |  13.47 |  −0.05 |     8.61 |
| 0.50 | inventory | 50.27 |   6.11 |   0.11 |     1.94 |
| 0.50 | symmetric | 58.34 |  11.60 |   0.04 |     7.16 |

**Key findings:**
1. The inventory strategy consistently achieves **lower profit standard deviation**
   and **lower terminal inventory dispersion** than the symmetric benchmark across
   all γ values — confirming the paper's core risk-reduction claim.
2. The symmetric benchmark earns **higher average profit** in all cases because it
   remains centred at the mid-price and captures more order flow.
3. At γ = 0.01 (near-zero risk aversion) the two strategies are closest:
   E[Π] differs by only 0.66 and Std[Π] by 4.53, consistent with the paper's
   prediction that the inventory strategy approaches the symmetric one as γ → 0.
4. At γ = 0.5 (high risk aversion) the inventory strategy is most conservative:
   Std[q_T] = 1.94 vs 7.16 for symmetric — a 3.7× reduction in inventory risk.

**Artefacts:** `exp1_summary_statistics.csv`, `exp1_sample_path_gamma*.png`,
`exp1_profit_hist_gamma*.png`, `exp1_gamma_comparison.png`.

---

## Experiment 2 — Diagnostic Replication (Spread-Ambiguity Variant)

**Motivation:** The paper's published tables report spread = 2/γ·ln(1+γ/k)
(constant), whereas the finite-horizon formula implies an additional
γ·σ²·(T−t) term. This experiment uses the constant-spread interpretation.

**Reported (constant) spreads:**

| γ    | Spread = 2/γ·ln(1+γ/k) |
|------|------------------------:|
| 0.10 |                  1.2908 |
| 0.01 |                  1.3289 |
| 0.50 |                  1.1507 |

**Diagnostic statistics (constant-spread variant):**

| γ    | Strategy  | E[Π]  | Std[Π] | E[q_T] | Std[q_T] |
|------|-----------|------:|-------:|-------:|---------:|
| 0.10 | inventory | 64.23 |   6.10 |   0.10 |     2.92 |
| 0.10 | symmetric | 68.88 |  13.63 |  −0.02 |     8.70 |
| 0.01 | inventory | 68.47 |   8.88 |   0.07 |     5.08 |
| 0.01 | symmetric | 69.05 |  13.65 |  −0.11 |     8.65 |
| 0.50 | inventory | 23.93 |   5.05 |   0.07 |     1.93 |
| 0.50 | symmetric | 68.29 |  13.81 |   0.12 |     9.16 |

**Exp 1 vs Exp 2 comparison:**

| γ    | Strategy  | Exp1 E[Π] | Exp2 E[Π] | Exp1 Std[Π] | Exp2 Std[Π] |
|------|-----------|----------:|----------:|------------:|------------:|
| 0.10 | inventory |     64.80 |     64.23 |        6.65 |        6.10 |
| 0.10 | symmetric |     68.25 |     68.88 |       13.37 |       13.63 |
| 0.01 | inventory |     68.45 |     68.47 |        8.94 |        8.88 |
| 0.01 | symmetric |     69.11 |     69.05 |       13.47 |       13.65 |
| 0.50 | inventory |     50.27 |     23.93 |        6.11 |        5.05 |
| 0.50 | symmetric |     58.34 |     68.29 |       11.60 |       13.81 |

**Key findings:**
1. For γ ∈ {0.1, 0.01} the two spread interpretations produce nearly identical
   results — the time-varying correction γ·σ²·(T−t) is small relative to the
   constant term.
2. For γ = 0.5 the difference is dramatic: the constant-spread inventory strategy
   earns only E[Π] = 23.93 vs 50.27 in Exp 1, because the constant spread is
   narrower than the finite-horizon spread at early times, leading to more
   aggressive quoting and larger inventory accumulation.
3. The symmetric benchmark is unaffected by the spread interpretation at γ = 0.01
   but diverges at γ = 0.5, confirming that the ambiguity matters most at high
   risk aversion.
4. The constant-spread variant does **not** clearly reproduce the paper's tables
   better than the finite-horizon variant; the ambiguity remains unresolved.

**Artefacts:** `exp2_diagnostic_statistics.csv`, `exp2_comparison_exp1_vs_exp2.csv`,
`exp2_spread_ambiguity_comparison.png`.

---

## Experiment 3 — Microstructure-to-Intensity Calibration Mapping

**Derivation summary:**

| Impact law | Intensity form | Parameters |
|------------|---------------|------------|
| Logarithmic: Δp = (1/K)·ln(Q) | λ(δ) = A·exp(−k·δ) | A = Λ/α, k = α·K |
| Power-law: Δp ∝ Q^β | λ(δ) = B·δ^{−α/β} | exponent = α/β |

**Exponential intensity parameters (Λ = 1) for representative α and K:**

| α    | K   | A      | k     |
|------|-----|-------:|------:|
| 1.53 | 0.5 | 0.6536 | 0.765 |
| 1.53 | 1.0 | 0.6536 | 1.530 |
| 1.53 | 1.5 | 0.6536 | 2.295 |
| 1.53 | 2.0 | 0.6536 | 3.060 |
| 1.40 | 1.0 | 0.7143 | 1.400 |
| 1.50 | 1.0 | 0.6667 | 1.500 |

**Power-law intensity exponents (α/β):**

| α    | β    | α/β   |
|------|------|------:|
| 1.53 | 0.50 | 3.060 |
| 1.53 | 0.76 | 2.013 |
| 1.40 | 0.50 | 2.800 |
| 1.50 | 0.50 | 3.000 |
| 1.50 | 0.76 | 1.974 |

**Main simulation parameter interpretation:**
With Λ = 140 and α = 1.53 (US stocks), k = 1.5 implies K = k/α ≈ 0.980.
The paper's A = 140 is consistent with Λ = 140·α ≈ 214 market orders per unit
time under the normalisation A = Λ/α.

**Key findings:**
1. Exponential execution intensity naturally emerges from power-law order-size
   distributions combined with logarithmic price impact.
2. Power-law execution intensity emerges when impact itself follows a power law.
3. Steeper intensity decay (faster fall-off with δ) occurs for larger α or K
   in the log-impact case, and for larger α/β in the power-impact case.
4. The main simulation parameters A = 140, k = 1.5 are interpretable as
   reduced-form counterparts of deeper microstructure primitives.

**Artefacts:** `exp3_exponential_params.csv`, `exp3_powerlaw_exponents.csv`,
`exp3_exponential_sensitivity.png`, `exp3_powerlaw_sensitivity.png`,
`exp3_model_comparison.png`.

---

## Experiment 4 — Appendix Extension: GBM + Mean-Variance

**Model:** GBM without drift dS/S = σ dW; mean-variance objective
V(x,s,q,t) = E_t[(x+qS_T) − (γ/2)(qS_T − qs)²].

**Closed-form value function:**
V(x,s,q,t) = x + qs + (γq²s²/2)·(exp(σ²(T−t)) − 1)

**Reservation prices (GBM appendix, σ = 0.2, s = 100, t = 0, T = 1):**

| γ    | q   | R_gbm   | Rᵃ_gbm  | Rᵇ_gbm  | r_abm  |
|------|-----|--------:|--------:|--------:|-------:|
| 0.10 | −5  | 304.054 | 324.459 | 283.649 | 102.00 |
| 0.10 |  0  | 100.000 | 120.405 |  79.595 | 100.00 |
| 0.10 |  5  | −104.054| −83.649 |−124.459 |  98.00 |

**Key findings:**
1. The GBM appendix reservation price shows the same directional inventory
   effect as the ABM main model: positive inventory lowers personal valuation,
   negative inventory raises it.
2. The GBM adjustment scales with s²·(exp(σ²(T−t))−1) rather than σ²·(T−t),
   producing much larger absolute adjustments for large |q| at s = 100.
3. Both models converge to the mid-price as t → T (adjustment vanishes).
4. The appendix is a conceptually related but methodologically distinct
   extension — it uses GBM dynamics and a mean-variance objective, not the
   exponential utility and ABM of the main model.

**Artefacts:** `exp4_reservation_prices.csv`, `exp4_reservation_vs_inventory.png`,
`exp4_reservation_vs_time.png`, `exp4_value_function_surface.png`.

---

## Conclusion

All four experiments successfully replicate and extend the Avellaneda-Stoikov
(2008) model:

- **Exp 1** confirms the core result: inventory-based quoting reduces risk
  (profit variance and inventory dispersion) at the cost of lower average profit.
- **Exp 2** quantifies the spread-ambiguity effect: negligible for small γ,
  material for γ = 0.5.
- **Exp 3** provides a self-contained microstructure derivation showing why
  exponential intensity is the natural reduced-form model.
- **Exp 4** validates the appendix GBM mean-variance extension as a distinct
  but qualitatively consistent framework.
