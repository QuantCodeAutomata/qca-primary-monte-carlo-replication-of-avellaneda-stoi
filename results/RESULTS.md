# Avellaneda-Stoikov Market Making Replication — Results

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| s0 | 100.0 |
| sigma | 2.0 |
| T | 1.0 |
| dt | 0.005 |
| n_steps | 200 |
| q0 | 0 |
| x0 | 0.0 |
| A | 140.0 |
| k | 1.5 |
| n_paths | 1000 |
| seed | 42 |

---

## Experiment 1: Primary Finite-Horizon Replication

| gamma | strategy | E[Pi_T] | Std[Pi_T] | E[q_T] | Std[q_T] |
|-------|----------|---------|-----------|--------|----------|
| 0.1 | inventory | 65.24 | 6.33 | -0.09 | 2.93 |
| 0.1 | symmetric | 68.29 | 13.42 | 0.03 | 8.34 |
| 0.01 | inventory | 68.74 | 9.10 | 0.03 | 5.19 |
| 0.01 | symmetric | 68.92 | 13.94 | 0.11 | 8.65 |
| 0.5 | inventory | 48.59 | 5.91 | -0.05 | 1.98 |
| 0.5 | symmetric | 58.62 | 11.77 | 0.12 | 7.03 |


### Experiment 1 Notes

- Uses finite-horizon closed-form quoting rules (Eq. 3.8/3.9 from paper)
- Symmetric benchmark uses time-varying spread: γσ²(T-t) + (2/γ)ln(1+γ/k)
- Common random numbers used across strategies for fair comparison
- Bid and ask Bernoulli events simulated independently within each dt
- Simultaneous fills possible but rare (λ·dt << 1)

---

## Experiment 2: Diagnostic Replication (Constant-Spread Variant)

| gamma | strategy | spread | E[Pi_T] | Std[Pi_T] | E[q_T] | Std[q_T] |
|-------|----------|--------|---------|-----------|--------|----------|
| 0.1 | inventory | 1.2908 | 64.37 | 5.97 | -0.09 | 2.93 |
| 0.1 | symmetric | 1.2908 | 68.72 | 13.92 | 0.12 | 8.73 |
| 0.01 | inventory | 1.3289 | 68.77 | 9.09 | 0.05 | 5.20 |
| 0.01 | symmetric | 1.3289 | 68.84 | 13.92 | 0.12 | 8.71 |
| 0.5 | inventory | 1.1507 | 23.82 | 4.91 | -0.04 | 1.96 |
| 0.5 | symmetric | 1.1507 | 67.82 | 14.22 | 0.18 | 9.15 |


### Experiment 2 Notes

- Diagnostic variant: constant spread = 2·c_γ = (2/γ)·ln(1+γ/k)
- Tests whether paper's published tables used constant-spread interpretation
- Same random seeds as Exp 1 for direct comparison
- Inventory strategy still centers around reservation price

---

## Experiment 3: Microstructure-to-Intensity Calibration

### Key Derivations

**Case 1: Power-law sizes + Logarithmic impact → Exponential intensity**

λ(δ) = A · exp(-k · δ)  where  A = Λ/α,  k = α·K

**Case 2: Power-law sizes + Power-law impact → Power-law intensity**

λ(δ) = B · δ^(-α/β)

### Parameter Mapping Table

| Market | α | K | A=Λ/α | k=α·K |
|--------|---|---|-------|-------|
| US stocks | 1.53 | 0.5 | 91.5033 | 0.7650 |
| US stocks | 1.53 | 1.0 | 91.5033 | 1.5300 |
| US stocks | 1.53 | 1.5 | 91.5033 | 2.2950 |
| NASDAQ | 1.4 | 0.5 | 100.0000 | 0.7000 |
| NASDAQ | 1.4 | 1.0 | 100.0000 | 1.4000 |
| NASDAQ | 1.4 | 1.5 | 100.0000 | 2.1000 |
| Paris Bourse | 1.5 | 0.5 | 93.3333 | 0.7500 |
| Paris Bourse | 1.5 | 1.0 | 93.3333 | 1.5000 |
| Paris Bourse | 1.5 | 1.5 | 93.3333 | 2.2500 |


### Experiment 3 Notes

- Standalone analytical/calibration experiment
- No trading simulation required
- Exponential intensity: λ(δ) = A·exp(-k·δ) where A=Λ/α, k=α·K
- Power-law intensity: λ(δ) = B·δ^(-α/β), singular at δ=0

---

## Experiment 4: Appendix Extension (GBM + Mean-Variance)

### Appendix Formulas

**Value function:**
V(x, s, q, t) = x + q·s + (γ·q²·s²/2) · (exp(σ²·(T-t)) - 1)

**Reservation ask:**
R^a(s, q, t) = s + ((1-2q)/2) · γ · s² · (exp(σ²·(T-t)) - 1)

**Reservation bid:**
R^b(s, q, t) = s + ((-1-2q)/2) · γ · s² · (exp(σ²·(T-t)) - 1)

**Reservation midpoint:**
R(s, q, t) = s - q · γ · s² · (exp(σ²·(T-t)) - 1)

### Key Differences from Main Model

| Feature | Main Model | Appendix |
|---------|-----------|----------|
| Price dynamics | Arithmetic BM | Geometric BM |
| Objective | CARA (exponential utility) | Mean-Variance |
| Reservation price adjustment | q·γ·σ²·(T-t) | q·γ·s²·(exp(σ²(T-t))-1) |
| Scales with price level | No | Yes (s²) |


### Experiment 4 Notes

- Appendix model: GBM price dynamics + mean-variance objective
- Completely independent from main Monte Carlo simulation
- Reservation price adjustment scales with s² and exp(σ²(T-t))-1
- Positive inventory → R < s; negative inventory → R > s (same direction as main model)

---

## Key Findings

### Experiment 1 (Primary Replication)

- **γ=0.1**: Inventory strategy reduces profit std by 52.9% and inventory std by 64.8% vs symmetric benchmark
- **γ=0.01**: Inventory strategy reduces profit std by 34.7% and inventory std by 40.0% vs symmetric benchmark
- **γ=0.5**: Inventory strategy reduces profit std by 49.8% and inventory std by 71.9% vs symmetric benchmark

### Experiment 2 (Diagnostic)

- **γ=0.1**: Constant-spread variant gives E[Pi_T]=64.37 vs finite-horizon E[Pi_T]=65.24
- **γ=0.01**: Constant-spread variant gives E[Pi_T]=68.77 vs finite-horizon E[Pi_T]=68.74
- **γ=0.5**: Constant-spread variant gives E[Pi_T]=23.82 vs finite-horizon E[Pi_T]=48.59

### Experiment 3 (Microstructure)

- Exponential intensity naturally emerges from power-law order sizes + log impact
- Power-law intensity emerges from power-law order sizes + power-law impact
- Main simulation parameters A=140, k=1.5 correspond to Λ/α and α·K respectively

### Experiment 4 (Appendix)

- Appendix reservation price has same directional inventory effect as main model
- Key difference: adjustment scales with s² and exp(σ²(T-t))-1 instead of σ²(T-t)
- GBM model is methodologically distinct from main ABM simulation

---

## Output Files

| File | Description |
|------|-------------|
| exp1_summary.csv | Exp 1 aggregated statistics |
| exp1_sample_path_gamma*.png | Representative path plots |
| exp1_profit_histograms.png | Terminal profit distributions |
| exp2_summary.csv | Exp 2 diagnostic statistics |
| exp2_profit_histograms.png | Diagnostic profit distributions |
| exp3_param_table.csv | Microstructure parameter mapping |
| exp3_exponential_intensity.png | Exponential intensity curves |
| exp3_powerlaw_intensity.png | Power-law intensity curves |
| exp3_intensity_comparison.png | Intensity model comparison |
| exp4_comparison.csv | Appendix vs main model comparison |
| exp4_reservation_vs_inventory.png | Appendix reservation price vs q |
| exp4_reservation_vs_time.png | Appendix reservation price vs t |
| exp4_model_comparison.png | Cross-model reservation price comparison |
