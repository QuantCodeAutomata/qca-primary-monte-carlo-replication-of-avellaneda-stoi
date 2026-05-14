"""
Main runner for all Avellaneda-Stoikov replication experiments.

Runs all four experiments and saves results to the results/ directory,
including a comprehensive RESULTS.md file.
"""

from __future__ import annotations

import os
import sys
import time

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.simulation import SimParams
from src.exp1_primary_replication import run_experiment_1, build_results_markdown as exp1_md
from src.exp2_diagnostic_replication import (
    run_experiment_2,
    build_results_markdown as exp2_md,
)
from src.exp3_microstructure_calibration import (
    run_experiment_3,
    build_results_markdown as exp3_md,
)
from src.exp4_appendix_gbm_mv import (
    run_experiment_4,
    build_results_markdown as exp4_md,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
GAMMAS = [0.1, 0.01, 0.5]


def main() -> None:
    """Run all experiments and write RESULTS.md."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    params = SimParams()

    print("\n" + "=" * 70)
    print("Avellaneda-Stoikov Market Making Replication")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Experiment 1: Primary finite-horizon Monte Carlo replication
    # -----------------------------------------------------------------------
    print("\n[1/4] Running Experiment 1: Primary Monte Carlo Replication...")
    t0 = time.time()
    results_1 = run_experiment_1(params=params, gammas=GAMMAS, results_dir=RESULTS_DIR)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Experiment 2: Diagnostic constant-spread replication
    # -----------------------------------------------------------------------
    print("\n[2/4] Running Experiment 2: Diagnostic Spread-Ambiguity Variant...")
    t0 = time.time()
    results_2 = run_experiment_2(params=params, gammas=GAMMAS, results_dir=RESULTS_DIR)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Experiment 3: Microstructure calibration mapping
    # -----------------------------------------------------------------------
    print("\n[3/4] Running Experiment 3: Microstructure Calibration Mapping...")
    t0 = time.time()
    results_3 = run_experiment_3(results_dir=RESULTS_DIR)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Experiment 4: Appendix GBM mean-variance extension
    # -----------------------------------------------------------------------
    print("\n[4/4] Running Experiment 4: Appendix GBM Mean-Variance Extension...")
    t0 = time.time()
    results_4 = run_experiment_4(results_dir=RESULTS_DIR)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Write RESULTS.md
    # -----------------------------------------------------------------------
    _write_results_md(params, results_1, results_2, results_3, results_4)

    print("\n" + "=" * 70)
    print(f"All experiments complete. Results saved to: {RESULTS_DIR}")
    print("=" * 70)


def _write_results_md(params, results_1, results_2, results_3, results_4) -> None:
    """Write comprehensive RESULTS.md file."""
    lines = [
        "# Avellaneda-Stoikov Market Making Replication — Results",
        "",
        "## Simulation Parameters",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| s0 | {params.s0} |",
        f"| sigma | {params.sigma} |",
        f"| T | {params.T} |",
        f"| dt | {params.dt} |",
        f"| n_steps | {params.n_steps} |",
        f"| q0 | {params.q0} |",
        f"| x0 | {params.x0} |",
        f"| A | {params.A} |",
        f"| k | {params.k} |",
        f"| n_paths | {params.n_paths} |",
        f"| seed | {params.seed} |",
        "",
        "---",
        "",
        exp1_md(results_1, GAMMAS, label="Experiment 1: Primary Finite-Horizon Replication"),
        "",
        "### Experiment 1 Notes",
        "",
        "- Uses finite-horizon closed-form quoting rules (Eq. 3.8/3.9 from paper)",
        "- Symmetric benchmark uses time-varying spread: γσ²(T-t) + (2/γ)ln(1+γ/k)",
        "- Common random numbers used across strategies for fair comparison",
        "- Bid and ask Bernoulli events simulated independently within each dt",
        "- Simultaneous fills possible but rare (λ·dt << 1)",
        "",
        "---",
        "",
        exp2_md(results_2, GAMMAS, params),
        "",
        "### Experiment 2 Notes",
        "",
        "- Diagnostic variant: constant spread = 2·c_γ = (2/γ)·ln(1+γ/k)",
        "- Tests whether paper's published tables used constant-spread interpretation",
        "- Same random seeds as Exp 1 for direct comparison",
        "- Inventory strategy still centers around reservation price",
        "",
        "---",
        "",
        exp3_md(results_3),
        "",
        "### Experiment 3 Notes",
        "",
        "- Standalone analytical/calibration experiment",
        "- No trading simulation required",
        "- Exponential intensity: λ(δ) = A·exp(-k·δ) where A=Λ/α, k=α·K",
        "- Power-law intensity: λ(δ) = B·δ^(-α/β), singular at δ=0",
        "",
        "---",
        "",
        exp4_md(results_4, GAMMAS),
        "",
        "### Experiment 4 Notes",
        "",
        "- Appendix model: GBM price dynamics + mean-variance objective",
        "- Completely independent from main Monte Carlo simulation",
        "- Reservation price adjustment scales with s² and exp(σ²(T-t))-1",
        "- Positive inventory → R < s; negative inventory → R > s (same direction as main model)",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### Experiment 1 (Primary Replication)",
        "",
    ]

    # Add key findings from Exp 1
    for gamma in GAMMAS:
        g_key = f"gamma_{gamma}"
        inv = results_1[g_key]["inventory"]
        sym = results_1[g_key]["symmetric"]
        risk_reduction = (sym.std_profit - inv.std_profit) / sym.std_profit * 100
        inv_risk_reduction = (sym.std_final_q - inv.std_final_q) / sym.std_final_q * 100
        lines.append(
            f"- **γ={gamma}**: Inventory strategy reduces profit std by "
            f"{risk_reduction:.1f}% and inventory std by {inv_risk_reduction:.1f}% "
            f"vs symmetric benchmark"
        )

    lines += [
        "",
        "### Experiment 2 (Diagnostic)",
        "",
    ]

    for gamma in GAMMAS:
        g_key = f"gamma_{gamma}"
        inv1 = results_1[g_key]["inventory"]
        inv2 = results_2[g_key]["inventory"]
        lines.append(
            f"- **γ={gamma}**: Constant-spread variant gives E[Pi_T]={inv2.mean_profit:.2f} "
            f"vs finite-horizon E[Pi_T]={inv1.mean_profit:.2f}"
        )

    lines += [
        "",
        "### Experiment 3 (Microstructure)",
        "",
        "- Exponential intensity naturally emerges from power-law order sizes + log impact",
        "- Power-law intensity emerges from power-law order sizes + power-law impact",
        "- Main simulation parameters A=140, k=1.5 correspond to Λ/α and α·K respectively",
        "",
        "### Experiment 4 (Appendix)",
        "",
        "- Appendix reservation price has same directional inventory effect as main model",
        "- Key difference: adjustment scales with s² and exp(σ²(T-t))-1 instead of σ²(T-t)",
        "- GBM model is methodologically distinct from main ABM simulation",
        "",
        "---",
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| exp1_summary.csv | Exp 1 aggregated statistics |",
        "| exp1_sample_path_gamma*.png | Representative path plots |",
        "| exp1_profit_histograms.png | Terminal profit distributions |",
        "| exp2_summary.csv | Exp 2 diagnostic statistics |",
        "| exp2_profit_histograms.png | Diagnostic profit distributions |",
        "| exp3_param_table.csv | Microstructure parameter mapping |",
        "| exp3_exponential_intensity.png | Exponential intensity curves |",
        "| exp3_powerlaw_intensity.png | Power-law intensity curves |",
        "| exp3_intensity_comparison.png | Intensity model comparison |",
        "| exp4_comparison.csv | Appendix vs main model comparison |",
        "| exp4_reservation_vs_inventory.png | Appendix reservation price vs q |",
        "| exp4_reservation_vs_time.png | Appendix reservation price vs t |",
        "| exp4_model_comparison.png | Cross-model reservation price comparison |",
        "",
    ]

    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved RESULTS.md → {md_path}")


if __name__ == "__main__":
    main()
