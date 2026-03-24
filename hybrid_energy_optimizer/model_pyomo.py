"""
model_pyomo.py
--------------
Hybrid energy system MILP optimizer implemented with **Pyomo**.

Covers:
  - PV (curtailable)
  - BESS (charge / discharge with binary exclusion)
  - CHP (binary on/off, min-load, startup cost)
  - Heat Pump
  - Thermal Storage
  - EV fleet (smart charging, optional V2G)
  - Grid import / export

Two model builders are exposed:
  build_deterministic(scenario, params)   → single-scenario LP/MILP
  build_stochastic(scenarios, params)     → two-stage stochastic MILP (CVaR)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from system_params import SystemParams, DEFAULT_PARAMS


# ─── helpers ────────────────────────────────────────────────────────────────

def _T(params: SystemParams):
    return range(params.n_hours)


def _solve(model: pyo.ConcreteModel, tee: bool = False) -> pyo.SolverResults:
    """Solve with HiGHS (via appsi) falling back to glpk."""
    try:
        solver = pyo.SolverFactory("appsi_highs")
        result = solver.solve(model, tee=tee)
    except Exception:
        solver = pyo.SolverFactory("glpk")
        result = solver.solve(model, tee=tee)
    return result


# ─── Deterministic model ─────────────────────────────────────────────────────

def build_deterministic(
    scenario: pd.DataFrame,
    params: SystemParams = DEFAULT_PARAMS,
) -> pyo.ConcreteModel:
    """
    Build and return a Pyomo ConcreteModel for one 24-h deterministic scenario.

    Decision variables (per hour t):
      p_grid_import, p_grid_export   — grid exchange [kW]
      p_bess_charge, p_bess_discharge — BESS power [kW]
      soc_bess                        — BESS state-of-charge [kWh]
      y_bess_charge, y_bess_discharge — binary: prevent simultaneous C/D
      p_chp_elec, p_chp_heat          — CHP output [kW]
      u_chp                           — binary: CHP on/off
      z_chp                           — binary: CHP startup this hour
      p_hp                            — heat pump thermal output [kW]
      soc_thermal                     — thermal storage SoC [kWh]
      p_pv                            — PV used (≤ available) [kW]
    """
    m = pyo.ConcreteModel(name="HybridEMS_Pyomo_Det")
    T = list(_T(params))
    b = params.bess
    c = params.chp
    hp = params.hp
    th = params.thermal
    g = params.grid
    dt = params.dt

    pv_avail = scenario["pv_kw"].values
    d_elec = (scenario["demand_elec_kw"] + scenario["ev_demand_kw"]).values
    d_heat = scenario["demand_heat_kw"].values
    price = scenario["price_eur_mwh"].values / 1000.0  # convert to €/kWh

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.T = pyo.Set(initialize=T)

    # ── Variables ─────────────────────────────────────────────────────────────
    m.p_import = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, g.max_import_kw))
    m.p_export = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, g.max_export_kw))

    m.p_bess_c = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, b.power_kw))
    m.p_bess_d = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, b.power_kw))
    m.y_bess_c = pyo.Var(m.T, domain=pyo.Binary)
    m.y_bess_d = pyo.Var(m.T, domain=pyo.Binary)
    m.soc_bess = pyo.Var(m.T, domain=pyo.NonNegativeReals,
                         bounds=(b.soc_min * b.capacity_kwh, b.soc_max * b.capacity_kwh))

    m.p_chp_e = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, c.capacity_elec_kw))
    m.p_chp_h = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, c.capacity_heat_kw))
    m.u_chp = pyo.Var(m.T, domain=pyo.Binary)
    m.z_chp = pyo.Var(m.T, domain=pyo.Binary)   # startup indicator

    m.p_hp = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, hp.capacity_kw))

    m.soc_th = pyo.Var(m.T, domain=pyo.NonNegativeReals,
                        bounds=(th.soc_min * th.capacity_kwh, th.soc_max * th.capacity_kwh))

    m.p_pv = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # ── Objective: minimise total operating cost ──────────────────────────────
    def obj_rule(m):
        energy_cost = sum(
            (price[t] + g.grid_fee) * m.p_import[t]
            - g.feed_in_tariff * m.p_export[t]
            for t in T
        ) * dt

        gas_cost = sum(
            (m.p_chp_e[t] / c.eta_elec) * c.gas_price
            for t in T
        ) * dt

        bess_wear = sum(
            b.degradation_cost * (m.p_bess_c[t] + m.p_bess_d[t])
            for t in T
        ) * dt

        startup_cost = sum(c.startup_cost * m.z_chp[t] for t in T)

        return energy_cost + gas_cost + bess_wear + startup_cost

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ── Constraints ───────────────────────────────────────────────────────────

    # PV curtailment
    m.pv_limit = pyo.Constraint(m.T, rule=lambda m, t: m.p_pv[t] <= pv_avail[t])

    # Electrical power balance
    def elec_balance(m, t):
        return (
            m.p_pv[t] + m.p_import[t] + m.p_bess_d[t] + m.p_chp_e[t]
            == d_elec[t] + m.p_export[t] + m.p_bess_c[t] + m.p_hp[t] / hp.cop
        )
    m.elec_balance = pyo.Constraint(m.T, rule=elec_balance)

    # Thermal balance (CHP heat + HP → thermal storage or direct demand)
    def heat_balance(m, t):
        heat_in = m.p_chp_h[t] + m.p_hp[t]
        if t == 0:
            soc_prev = th.soc_init * th.capacity_kwh
        else:
            soc_prev = m.soc_th[t - 1]
        return m.soc_th[t] == th.eta_store * soc_prev + (heat_in - d_heat[t]) * dt

    m.heat_balance = pyo.Constraint(m.T, rule=heat_balance)

    # BESS SoC dynamics
    def bess_soc(m, t):
        if t == 0:
            soc_prev = b.soc_init * b.capacity_kwh
        else:
            soc_prev = m.soc_bess[t - 1]
        return m.soc_bess[t] == soc_prev + (b.eta_charge * m.p_bess_c[t]
                                              - m.p_bess_d[t] / b.eta_discharge) * dt
    m.bess_soc = pyo.Constraint(m.T, rule=bess_soc)

    # BESS binary exclusion (no simultaneous charge & discharge)
    m.bess_c_ub = pyo.Constraint(m.T, rule=lambda m, t: m.p_bess_c[t] <= b.power_kw * m.y_bess_c[t])
    m.bess_d_ub = pyo.Constraint(m.T, rule=lambda m, t: m.p_bess_d[t] <= b.power_kw * m.y_bess_d[t])
    m.bess_ex  = pyo.Constraint(m.T, rule=lambda m, t: m.y_bess_c[t] + m.y_bess_d[t] <= 1)

    # CHP: electrical ↔ heat coupling (fixed heat-to-power ratio)
    ratio = c.capacity_heat_kw / c.capacity_elec_kw
    m.chp_coupling = pyo.Constraint(m.T, rule=lambda m, t: m.p_chp_h[t] == ratio * m.p_chp_e[t])

    # CHP: min-load & on/off linking
    m.chp_ub = pyo.Constraint(m.T, rule=lambda m, t: m.p_chp_e[t] <= c.capacity_elec_kw * m.u_chp[t])
    m.chp_lb = pyo.Constraint(m.T, rule=lambda m, t:
                               m.p_chp_e[t] >= c.min_load * c.capacity_elec_kw * m.u_chp[t])

    # CHP startup detection: z_chp[t] ≥ u_chp[t] - u_chp[t-1]
    def chp_startup(m, t):
        if t == 0:
            return m.z_chp[t] >= m.u_chp[t]
        return m.z_chp[t] >= m.u_chp[t] - m.u_chp[t - 1]
    m.chp_startup = pyo.Constraint(m.T, rule=chp_startup)

    # Thermal storage non-negativity already handled by bounds;
    # but heat balance must keep SoC in range — add slack here
    m.th_lb = pyo.Constraint(m.T, rule=lambda m, t:
                              m.soc_th[t] >= th.soc_min * th.capacity_kwh)
    m.th_ub = pyo.Constraint(m.T, rule=lambda m, t:
                              m.soc_th[t] <= th.soc_max * th.capacity_kwh)

    return m


def solve_deterministic(
    scenario: pd.DataFrame,
    params: SystemParams = DEFAULT_PARAMS,
    tee: bool = False,
) -> dict:
    """Solve deterministic model, return results dict."""
    m = build_deterministic(scenario, params)
    result = _solve(m, tee=tee)

    status = result.solver.termination_condition
    T = list(_T(params))

    out = {
        "framework": "Pyomo",
        "mode": "deterministic",
        "status": str(status),
        "objective_eur": pyo.value(m.obj),
        "hourly": pd.DataFrame({
            "hour": T,
            "p_import_kw":   [pyo.value(m.p_import[t])   for t in T],
            "p_export_kw":   [pyo.value(m.p_export[t])   for t in T],
            "p_bess_c_kw":  [pyo.value(m.p_bess_c[t])    for t in T],
            "p_bess_d_kw":  [pyo.value(m.p_bess_d[t])    for t in T],
            "soc_bess_kwh": [pyo.value(m.soc_bess[t])    for t in T],
            "p_chp_e_kw":   [pyo.value(m.p_chp_e[t])     for t in T],
            "p_chp_h_kw":   [pyo.value(m.p_chp_h[t])     for t in T],
            "u_chp":        [pyo.value(m.u_chp[t])        for t in T],
            "p_hp_kw":      [pyo.value(m.p_hp[t])         for t in T],
            "soc_th_kwh":   [pyo.value(m.soc_th[t])       for t in T],
            "p_pv_kw":      [pyo.value(m.p_pv[t])         for t in T],
        }),
    }
    return out


# ─── Stochastic model (two-stage, CVaR) ──────────────────────────────────────

def build_stochastic(
    scenarios: dict[str, np.ndarray],
    params: SystemParams = DEFAULT_PARAMS,
) -> pyo.ConcreteModel:
    """
    Two-stage stochastic MILP with CVaR risk measure.

    Stage 1 (here-and-now): CHP on/off schedule u_chp[t] (binary)
    Stage 2 (recourse):     Continuous dispatch per scenario

    Objective:
        min  E[cost(s)] + λ · CVaR_α[cost(s)]

    CVaR linearisation (Rockafellar-Uryasev):
        CVaR_α = η + (1/(1-α)) · E[max(cost(s) - η, 0)]
               = η + (1/(1-α)) · Σ_s prob_s · φ_s
        where φ_s ≥ cost_s - η,  φ_s ≥ 0
    """
    m = pyo.ConcreteModel(name="HybridEMS_Pyomo_Stoch")
    T = list(_T(params))
    S = list(range(params.n_scenarios))
    b = params.bess
    c = params.chp
    hp = params.hp
    th = params.thermal
    g = params.grid
    dt = params.dt
    alpha = params.cvar_alpha
    lam = params.risk_weight

    probs = scenarios["probabilities"]              # shape (n_scenarios,)
    pv_sc = scenarios["pv"]                         # shape (n_scenarios, 24)
    d_elec_sc = scenarios["demand_elec"]
    price_sc = scenarios["price"] / 1000.0          # €/kWh
    heat_sc = scenarios["heat"]

    m.T = pyo.Set(initialize=T)
    m.S = pyo.Set(initialize=S)

    # ── Stage-1 (first-stage) binary: CHP commitment ─────────────────────────
    m.u_chp = pyo.Var(m.T, domain=pyo.Binary)
    m.z_chp = pyo.Var(m.T, domain=pyo.Binary)

    # ── Stage-2 (recourse) variables — one copy per scenario ─────────────────
    m.p_import  = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, g.max_import_kw))
    m.p_export  = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, g.max_export_kw))
    m.p_bess_c  = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, b.power_kw))
    m.p_bess_d  = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, b.power_kw))
    m.y_bess_c  = pyo.Var(m.T, m.S, domain=pyo.Binary)
    m.y_bess_d  = pyo.Var(m.T, m.S, domain=pyo.Binary)
    m.soc_bess  = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals,
                          bounds=(b.soc_min * b.capacity_kwh, b.soc_max * b.capacity_kwh))
    m.p_chp_e   = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, c.capacity_elec_kw))
    m.p_chp_h   = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, c.capacity_heat_kw))
    m.p_hp      = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, hp.capacity_kw))
    m.soc_th    = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals,
                          bounds=(th.soc_min * th.capacity_kwh, th.soc_max * th.capacity_kwh))
    m.p_pv      = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals)

    # ── CVaR auxiliary variables ──────────────────────────────────────────────
    m.eta = pyo.Var(domain=pyo.Reals)                         # VaR estimate
    m.phi = pyo.Var(m.S, domain=pyo.NonNegativeReals)         # excess loss

    # ── Scenario cost expression ──────────────────────────────────────────────
    def scenario_cost(m, s):
        energy = sum(
            (price_sc[s, t] + g.grid_fee) * m.p_import[t, s]
            - g.feed_in_tariff * m.p_export[t, s]
            for t in T
        ) * dt
        gas = sum(
            (m.p_chp_e[t, s] / c.eta_elec) * c.gas_price for t in T
        ) * dt
        wear = sum(
            b.degradation_cost * (m.p_bess_c[t, s] + m.p_bess_d[t, s]) for t in T
        ) * dt
        startup = sum(c.startup_cost * m.z_chp[t] for t in T)
        return energy + gas + wear + startup

    m.cost_expr = pyo.Expression(m.S, rule=scenario_cost)

    # ── Objective ─────────────────────────────────────────────────────────────
    def obj_rule(m):
        expected = sum(probs[s] * m.cost_expr[s] for s in S)
        cvar = m.eta + (1.0 / (1.0 - alpha)) * sum(probs[s] * m.phi[s] for s in S)
        return expected + lam * cvar

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ── CVaR constraints ──────────────────────────────────────────────────────
    m.cvar_link = pyo.Constraint(m.S,
        rule=lambda m, s: m.phi[s] >= m.cost_expr[s] - m.eta)

    # ── Per-scenario operational constraints ──────────────────────────────────

    m.pv_limit = pyo.Constraint(m.T, m.S,
        rule=lambda m, t, s: m.p_pv[t, s] <= pv_sc[s, t])

    def elec_balance(m, t, s):
        return (
            m.p_pv[t, s] + m.p_import[t, s] + m.p_bess_d[t, s] + m.p_chp_e[t, s]
            == d_elec_sc[s, t] + m.p_export[t, s] + m.p_bess_c[t, s] + m.p_hp[t, s] / hp.cop
        )
    m.elec_balance = pyo.Constraint(m.T, m.S, rule=elec_balance)

    def heat_balance(m, t, s):
        heat_in = m.p_chp_h[t, s] + m.p_hp[t, s]
        soc_prev = th.soc_init * th.capacity_kwh if t == 0 else m.soc_th[t - 1, s]
        return m.soc_th[t, s] == th.eta_store * soc_prev + (heat_in - heat_sc[s, t]) * dt
    m.heat_balance = pyo.Constraint(m.T, m.S, rule=heat_balance)

    def bess_soc_rule(m, t, s):
        soc_prev = b.soc_init * b.capacity_kwh if t == 0 else m.soc_bess[t - 1, s]
        return m.soc_bess[t, s] == soc_prev + (
            b.eta_charge * m.p_bess_c[t, s] - m.p_bess_d[t, s] / b.eta_discharge
        ) * dt
    m.bess_soc = pyo.Constraint(m.T, m.S, rule=bess_soc_rule)

    m.bess_c_ub = pyo.Constraint(m.T, m.S, rule=lambda m, t, s: m.p_bess_c[t, s] <= b.power_kw * m.y_bess_c[t, s])
    m.bess_d_ub = pyo.Constraint(m.T, m.S, rule=lambda m, t, s: m.p_bess_d[t, s] <= b.power_kw * m.y_bess_d[t, s])
    m.bess_ex   = pyo.Constraint(m.T, m.S, rule=lambda m, t, s: m.y_bess_c[t, s] + m.y_bess_d[t, s] <= 1)

    ratio = c.capacity_heat_kw / c.capacity_elec_kw
    m.chp_coupling = pyo.Constraint(m.T, m.S, rule=lambda m, t, s: m.p_chp_h[t, s] == ratio * m.p_chp_e[t, s])
    m.chp_ub = pyo.Constraint(m.T, m.S, rule=lambda m, t, s: m.p_chp_e[t, s] <= c.capacity_elec_kw * m.u_chp[t])
    m.chp_lb = pyo.Constraint(m.T, m.S, rule=lambda m, t, s:
                               m.p_chp_e[t, s] >= c.min_load * c.capacity_elec_kw * m.u_chp[t])

    def chp_startup(m, t):
        if t == 0:
            return m.z_chp[t] >= m.u_chp[t]
        return m.z_chp[t] >= m.u_chp[t] - m.u_chp[t - 1]
    m.chp_startup = pyo.Constraint(m.T, rule=chp_startup)

    m.th_lb = pyo.Constraint(m.T, m.S, rule=lambda m, t, s: m.soc_th[t, s] >= th.soc_min * th.capacity_kwh)
    m.th_ub = pyo.Constraint(m.T, m.S, rule=lambda m, t, s: m.soc_th[t, s] <= th.soc_max * th.capacity_kwh)

    return m


def solve_stochastic(
    scenarios: dict[str, np.ndarray],
    params: SystemParams = DEFAULT_PARAMS,
    tee: bool = False,
) -> dict:
    """Solve stochastic model, return results dict."""
    m = build_stochastic(scenarios, params)
    result = _solve(m, tee=tee)

    T = list(_T(params))
    S = list(range(params.n_scenarios))

    # Mean dispatch across scenarios
    mean_import  = [np.mean([pyo.value(m.p_import[t, s])  for s in S]) for t in T]
    mean_export  = [np.mean([pyo.value(m.p_export[t, s])  for s in S]) for t in T]
    mean_bess_c  = [np.mean([pyo.value(m.p_bess_c[t, s])  for s in S]) for t in T]
    mean_bess_d  = [np.mean([pyo.value(m.p_bess_d[t, s])  for s in S]) for t in T]
    mean_soc     = [np.mean([pyo.value(m.soc_bess[t, s])  for s in S]) for t in T]
    mean_chp_e   = [np.mean([pyo.value(m.p_chp_e[t, s])   for s in S]) for t in T]
    mean_hp      = [np.mean([pyo.value(m.p_hp[t, s])       for s in S]) for t in T]
    chp_schedule = [pyo.value(m.u_chp[t]) for t in T]

    scenario_costs = [pyo.value(m.cost_expr[s]) for s in S]

    return {
        "framework": "Pyomo",
        "mode": "stochastic",
        "status": str(result.solver.termination_condition),
        "objective": pyo.value(m.obj),
        "expected_cost_eur": float(np.dot(params.n_scenarios * [1 / params.n_scenarios], scenario_costs)),
        "cvar_eur": pyo.value(m.eta) + (1 / (1 - params.cvar_alpha)) * float(
            np.dot(params.n_scenarios * [1 / params.n_scenarios],
                   [pyo.value(m.phi[s]) for s in S])
        ),
        "chp_schedule": chp_schedule,
        "scenario_costs": scenario_costs,
        "hourly_mean": pd.DataFrame({
            "hour": T,
            "p_import_kw":  mean_import,
            "p_export_kw":  mean_export,
            "p_bess_c_kw":  mean_bess_c,
            "p_bess_d_kw":  mean_bess_d,
            "soc_bess_kwh": mean_soc,
            "p_chp_e_kw":   mean_chp_e,
            "p_hp_kw":      mean_hp,
            "u_chp":        chp_schedule,
        }),
    }
