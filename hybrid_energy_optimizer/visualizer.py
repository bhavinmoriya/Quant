"""
visualizer.py
-------------
Publication-quality plots for hybrid EMS results:
  1. Energy dispatch stack (per hour)
  2. BESS & thermal storage SoC
  3. Framework comparison (Pyomo vs Linopy cost/time)
  4. Monte Carlo cost distribution with CVaR annotation
  5. CHP commitment schedule heatmap
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

PALETTE = {
    "pv":       "#F4A621",
    "bess_d":   "#4CAF50",
    "chp":      "#E53935",
    "grid_imp": "#1565C0",
    "hp":       "#9C27B0",
    "demand":   "#212121",
    "export":   "#80CBC4",
    "bess_c":   "#A5D6A7",
}


def plot_dispatch(result: dict, scenario: pd.DataFrame, save_path: str | None = None):
    """Stacked area / bar dispatch chart."""
    h = result["hourly"] if "hourly" in result else result["hourly_mean"]
    hours = h["hour"].values

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax1, ax2 = axes

    # ── Generation side ───────────────────────────────────────────────────────
    gen_pv   = h["p_pv_kw"].values    if "p_pv_kw"   in h.columns else np.zeros(24)
    gen_bess = h["p_bess_d_kw"].values
    gen_chp  = h["p_chp_e_kw"].values
    gen_grid = h["p_import_kw"].values

    bottom = np.zeros(24)
    for label, vals, color in [
        ("PV", gen_pv, PALETTE["pv"]),
        ("BESS discharge", gen_bess, PALETTE["bess_d"]),
        ("CHP (elec)", gen_chp, PALETTE["chp"]),
        ("Grid import", gen_grid, PALETTE["grid_imp"]),
    ]:
        ax1.bar(hours, vals, bottom=bottom, label=label, color=color, alpha=0.85, width=0.8)
        bottom += vals

    demand = scenario["demand_elec_kw"].values + scenario["ev_demand_kw"].values
    ax1.step(hours, demand, where="mid", color=PALETTE["demand"], lw=2.5,
             linestyle="--", label="Total elec demand")
    ax1.set_ylabel("Power [kW]", fontsize=12)
    ax1.set_title(f"Electricity Dispatch — {result['framework']} ({result['mode']})", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    # ── Consumption / storage ────────────────────────────────────────────────
    ax2.bar(hours, h["p_bess_c_kw"].values, label="BESS charge",
            color=PALETTE["bess_c"], alpha=0.85, width=0.8)
    ax2.bar(hours, h["p_export_kw"].values if "p_export_kw" in h.columns else 0,
            bottom=h["p_bess_c_kw"].values,
            label="Grid export", color=PALETTE["export"], alpha=0.85, width=0.8)
    ax2.bar(hours, h["p_hp_kw"].values / 3.5,   # HP electrical draw
            bottom=h["p_bess_c_kw"].values + (h["p_export_kw"].values if "p_export_kw" in h.columns else 0),
            label="Heat pump (elec draw)", color=PALETTE["hp"], alpha=0.85, width=0.8)
    ax2.set_ylabel("Power [kW]", fontsize=12)
    ax2.set_xlabel("Hour of day", fontsize=12)
    ax2.set_title("Sinks", fontsize=13)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(hours)

    # CHP on/off shading
    for ax in axes:
        for t, u in enumerate(h["u_chp"].values):
            if u > 0.5:
                ax.axvspan(t - 0.4, t + 0.4, alpha=0.08, color=PALETTE["chp"], zorder=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_storage_soc(result: dict, params, save_path: str | None = None):
    """Dual-axis SoC plot: BESS kWh and thermal storage kWh."""
    h = result["hourly"] if "hourly" in result else result["hourly_mean"]
    hours = h["hour"].values

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.fill_between(hours, h["soc_bess_kwh"].values, alpha=0.4, color="#4CAF50", step="mid")
    ax1.step(hours, h["soc_bess_kwh"].values, color="#2E7D32", lw=2, where="mid", label="BESS SoC")
    ax1.axhline(params.bess.soc_min * params.bess.capacity_kwh, ls=":", color="#2E7D32", alpha=0.5)
    ax1.axhline(params.bess.soc_max * params.bess.capacity_kwh, ls=":", color="#2E7D32", alpha=0.5)
    ax1.set_ylabel("BESS SoC [kWh]", color="#2E7D32", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#2E7D32")

    if "soc_th_kwh" in h.columns:
        ax2.fill_between(hours, h["soc_th_kwh"].values, alpha=0.3, color="#E53935", step="mid")
        ax2.step(hours, h["soc_th_kwh"].values, color="#B71C1C", lw=2, where="mid", label="Thermal SoC")
        ax2.set_ylabel("Thermal Storage SoC [kWh]", color="#B71C1C", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#B71C1C")

    ax1.set_xlabel("Hour of day", fontsize=12)
    ax1.set_title(f"Storage State-of-Charge — {result['framework']}", fontsize=13)
    ax1.set_xticks(hours)
    ax1.grid(alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cost_comparison(results: list[dict], save_path: str | None = None):
    """Bar chart comparing Pyomo vs Linopy cost and solve time."""
    labels  = [f"{r['framework']}\n({r['mode']})" for r in results]
    costs   = [r.get("objective_eur", r.get("objective", 0)) for r in results]
    times   = [r.get("solve_time_s", 0) for r in results]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    bars = ax1.bar(x, costs, color=["#1565C0", "#E53935", "#2E7D32", "#F4A621"][:len(labels)],
                   alpha=0.85)
    ax1.bar_label(bars, fmt="€%.2f", padding=3, fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Objective [€]", fontsize=12)
    ax1.set_title("Optimized Operating Cost", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, times, color=["#1565C0", "#E53935", "#2E7D32", "#F4A621"][:len(labels)],
                    alpha=0.85)
    ax2.bar_label(bars2, fmt="%.2fs", padding=3, fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Solve time [s]", fontsize=12)
    ax2.set_title("Solver Wall-Clock Time", fontsize=13)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Framework Comparison: Pyomo vs Linopy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scenario_cost_distribution(
    scenario_costs: np.ndarray,
    cvar_value: float,
    alpha: float = 0.95,
    framework: str = "Pyomo",
    save_path: str | None = None,
):
    """Histogram of Monte Carlo scenario costs with VaR/CVaR markers."""
    fig, ax = plt.subplots(figsize=(10, 5))

    var_value = np.quantile(scenario_costs, alpha)

    ax.hist(scenario_costs, bins=30, color="#1565C0", alpha=0.7, edgecolor="white", label="Scenario costs")
    ax.axvline(var_value, color="#F4A621", lw=2.5, linestyle="--",
               label=f"VaR ({int(alpha*100)}%) = €{var_value:.2f}")
    ax.axvline(cvar_value, color="#E53935", lw=2.5, linestyle="-",
               label=f"CVaR ({int(alpha*100)}%) = €{cvar_value:.2f}")
    ax.fill_betweenx(
        [0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10],
        var_value, max(scenario_costs),
        alpha=0.15, color="#E53935", label="Tail risk region"
    )

    ax.set_xlabel("Scenario total cost [€]", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Monte Carlo Cost Distribution — {framework} Stochastic Model", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pv_scenario_fan(scenarios: dict, n_show: int = 30, save_path: str | None = None):
    """Fan chart of PV generation scenarios."""
    pv = scenarios["pv"]
    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(min(n_show, len(pv))):
        ax.plot(hours, pv[i], color=PALETTE["pv"], alpha=0.15, lw=0.8)
    ax.plot(hours, pv.mean(axis=0), color="#E65100", lw=2.5, label="Mean PV")
    ax.fill_between(hours, np.percentile(pv, 10, axis=0), np.percentile(pv, 90, axis=0),
                    alpha=0.25, color=PALETTE["pv"], label="10-90th percentile")
    ax.set_xlabel("Hour of day", fontsize=12)
    ax.set_ylabel("PV generation [kW]", fontsize=12)
    ax.set_title(f"Monte Carlo PV Scenarios (n={len(pv)})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
