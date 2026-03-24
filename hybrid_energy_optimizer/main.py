"""
main.py
-------
Orchestrates the full hybrid energy system optimization pipeline:

  1. Generate synthetic energy profiles + Monte Carlo scenarios
  2. Solve deterministic model (Pyomo & Linopy)
  3. Solve stochastic model with CVaR (Pyomo & Linopy)
  4. Compare frameworks on cost and solve time
  5. Produce all plots and a summary CSV report

Run:
    python main.py
    python main.py --n_scenarios 50 --n_hours 24
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_generator import build_base_scenario, generate_scenarios
from system_params import SystemParams
from model_pyomo import solve_deterministic as pyomo_det, solve_stochastic as pyomo_stoch
from model_linopy import (
    build_and_solve_deterministic as linopy_det,
    build_and_solve_stochastic as linopy_stoch,
)
from visualizer import (
    plot_dispatch,
    plot_storage_soc,
    plot_cost_comparison,
    plot_scenario_cost_distribution,
    plot_pv_scenario_fan,
)

PLOTS_DIR = Path("plots")
RESULTS_DIR = Path("results")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    result["solve_time_s"] = elapsed
    return result


def print_summary(result: dict):
    fw  = result["framework"]
    mod = result["mode"]
    obj = result.get("objective_eur", result.get("objective", "N/A"))
    t   = result.get("solve_time_s", 0)
    print(f"  [{fw:6s} | {mod:13s}]  cost = €{obj:.4f}   time = {t:.3f}s")


def main(args):
    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    params = SystemParams(n_scenarios=args.n_scenarios)

    print("=" * 65)
    print("  Hybrid Energy System MILP Optimizer")
    print("  Pyomo vs Linopy | Deterministic + Stochastic + CVaR")
    print("=" * 65)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print("\n[1/5] Generating profiles and Monte Carlo scenarios …")
    base = build_base_scenario()
    scenarios = generate_scenarios(n_scenarios=args.n_scenarios)
    print(f"      Base scenario: 24h  |  MC scenarios: {args.n_scenarios}")

    # PV fan chart
    plot_pv_scenario_fan(scenarios, n_show=50,
                         save_path=str(PLOTS_DIR / "pv_scenarios.png"))
    print("      ✓ PV scenario fan chart saved")

    # ── 2. Deterministic ──────────────────────────────────────────────────────
    print("\n[2/5] Solving deterministic MILP …")

    r_pyomo_det = timed(solve_deterministic_pyomo, base, params)
    print_summary(r_pyomo_det)

    r_linopy_det = timed(linopy_det, base, params)
    print_summary(r_linopy_det)

    # ── 3. Stochastic ─────────────────────────────────────────────────────────
    print(f"\n[3/5] Solving stochastic MILP ({args.n_scenarios} scenarios, CVaR {params.cvar_alpha:.0%}) …")
    print("      (This may take ~1-3 minutes depending on hardware)")

    r_pyomo_stoch = timed(pyomo_stoch, scenarios, params)
    print_summary(r_pyomo_stoch)

    r_linopy_stoch = timed(linopy_stoch, scenarios, params)
    print_summary(r_linopy_stoch)

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("\n[4/5] Generating plots …")

    plot_dispatch(r_pyomo_det, base,
                  save_path=str(PLOTS_DIR / "dispatch_pyomo_det.png"))
    print("      ✓ Pyomo deterministic dispatch")

    plot_dispatch(r_linopy_det, base,
                  save_path=str(PLOTS_DIR / "dispatch_linopy_det.png"))
    print("      ✓ Linopy deterministic dispatch")

    plot_storage_soc(r_pyomo_det, params,
                     save_path=str(PLOTS_DIR / "soc_pyomo_det.png"))
    print("      ✓ Storage SoC (Pyomo det)")

    plot_cost_comparison(
        [r_pyomo_det, r_linopy_det, r_pyomo_stoch, r_linopy_stoch],
        save_path=str(PLOTS_DIR / "framework_comparison.png"),
    )
    print("      ✓ Framework cost & time comparison")

    if "scenario_costs" in r_pyomo_stoch:
        plot_scenario_cost_distribution(
            np.array(r_pyomo_stoch["scenario_costs"]),
            cvar_value=r_pyomo_stoch.get("cvar_eur", 0),
            alpha=params.cvar_alpha,
            framework="Pyomo",
            save_path=str(PLOTS_DIR / "cost_distribution_pyomo.png"),
        )
        print("      ✓ Monte Carlo cost distribution (Pyomo)")

    # ── 5. Summary CSV ────────────────────────────────────────────────────────
    print("\n[5/5] Saving results …")

    summary = pd.DataFrame([
        {
            "framework":     r["framework"],
            "mode":          r["mode"],
            "objective_eur": r.get("objective_eur", r.get("objective")),
            "solve_time_s":  r.get("solve_time_s"),
            "solver_status": r.get("status"),
        }
        for r in [r_pyomo_det, r_linopy_det, r_pyomo_stoch, r_linopy_stoch]
    ])
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print("      ✓ results/summary.csv")

    r_pyomo_det["hourly"].to_csv(RESULTS_DIR / "hourly_pyomo_det.csv", index=False)
    r_linopy_det["hourly"].to_csv(RESULTS_DIR / "hourly_linopy_det.csv", index=False)
    r_pyomo_stoch["hourly_mean"].to_csv(RESULTS_DIR / "hourly_pyomo_stoch.csv", index=False)
    r_linopy_stoch["hourly_mean"].to_csv(RESULTS_DIR / "hourly_linopy_stoch.csv", index=False)
    print("      ✓ hourly dispatch CSVs")

    print("\n" + "=" * 65)
    print("  All done! Plots in ./plots/  |  Results in ./results/")
    print("=" * 65)
    print("\nKey results:")
    print(summary.to_string(index=False))

    return summary


def solve_deterministic_pyomo(base, params):
    """Wrapper so timed() can wrap it cleanly."""
    return pyomo_det(base, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid EMS Optimizer")
    parser.add_argument("--n_scenarios", type=int, default=30,
                        help="Number of Monte Carlo scenarios (default 30)")
    parser.add_argument("--n_hours", type=int, default=24,
                        help="Optimization horizon in hours (default 24)")
    args = parser.parse_args()
    main(args)
