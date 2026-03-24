"""
model_linopy.py
---------------
Hybrid energy system MILP optimizer implemented with **Linopy**.

Linopy wraps HiGHS directly via xarray-indexed variables — no callback
overhead, very fast model instantiation, and native pandas/xarray integration.

Same physical model as model_pyomo.py so results are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import linopy

from system_params import SystemParams, DEFAULT_PARAMS


# ─── helpers ────────────────────────────────────────────────────────────────

def _idx(n: int) -> pd.Index:
    return pd.RangeIndex(n)


# ─── Deterministic model ─────────────────────────────────────────────────────

def build_and_solve_deterministic(
    scenario: pd.DataFrame,
    params: SystemParams = DEFAULT_PARAMS,
) -> dict:
    """
    Build and solve the deterministic hybrid EMS model with Linopy.
    Returns a results dict compatible with the Pyomo counterpart.
    """
    T = pd.Index(range(params.n_hours), name="hour")
    b = params.bess
    c = params.chp
    hp = params.hp
    th = params.thermal
    g = params.grid
    dt = params.dt

    pv_avail = scenario["pv_kw"].values
    d_elec   = (scenario["demand_elec_kw"] + scenario["ev_demand_kw"]).values
    d_heat   = scenario["demand_heat_kw"].values
    price    = scenario["price_eur_mwh"].values / 1000.0   # €/kWh

    m = linopy.Model()

    # ── Variables ──────────────────────────────────────────────────────────────
    p_import  = m.add_variables(lower=0, upper=g.max_import_kw,  coords=[T], name="p_import")
    p_export  = m.add_variables(lower=0, upper=g.max_export_kw,  coords=[T], name="p_export")

    p_bess_c  = m.add_variables(lower=0, upper=b.power_kw, coords=[T], name="p_bess_c")
    p_bess_d  = m.add_variables(lower=0, upper=b.power_kw, coords=[T], name="p_bess_d")
    y_bess_c  = m.add_variables(binary=True, coords=[T], name="y_bess_c")
    y_bess_d  = m.add_variables(binary=True, coords=[T], name="y_bess_d")
    soc_bess  = m.add_variables(
        lower=b.soc_min * b.capacity_kwh,
        upper=b.soc_max * b.capacity_kwh,
        coords=[T], name="soc_bess"
    )

    p_chp_e   = m.add_variables(lower=0, upper=c.capacity_elec_kw, coords=[T], name="p_chp_e")
    p_chp_h   = m.add_variables(lower=0, upper=c.capacity_heat_kw, coords=[T], name="p_chp_h")
    u_chp     = m.add_variables(binary=True, coords=[T], name="u_chp")
    z_chp     = m.add_variables(binary=True, coords=[T], name="z_chp")

    p_hp      = m.add_variables(lower=0, upper=hp.capacity_kw, coords=[T], name="p_hp")
    soc_th    = m.add_variables(
        lower=th.soc_min * th.capacity_kwh,
        upper=th.soc_max * th.capacity_kwh,
        coords=[T], name="soc_th"
    )
    p_pv      = m.add_variables(lower=0, upper=pv_avail, coords=[T], name="p_pv")

    # ── Objective ──────────────────────────────────────────────────────────────
    energy_cost = ((price + g.grid_fee) * p_import - g.feed_in_tariff * p_export).sum() * dt
    gas_cost    = (p_chp_e / c.eta_elec * c.gas_price).sum() * dt
    bess_wear   = (b.degradation_cost * (p_bess_c + p_bess_d)).sum() * dt
    startup_cost = (c.startup_cost * z_chp).sum()

    m.add_objective(energy_cost + gas_cost + bess_wear + startup_cost)

    # ── Constraints ────────────────────────────────────────────────────────────

    # Electrical balance
    m.add_constraints(
        p_pv + p_import + p_bess_d + p_chp_e
        == d_elec + p_export + p_bess_c + p_hp / hp.cop,
        name="elec_balance"
    )

    # BESS SoC dynamics (per time step via shifting)
    # t=0: use initial SoC
    soc_init_arr = np.full(params.n_hours, b.soc_init * b.capacity_kwh)
    soc_prev = linopy.Variable.__new__(linopy.Variable)  # placeholder

    for t in range(params.n_hours):
        soc_p = b.soc_init * b.capacity_kwh if t == 0 else soc_bess.sel({"hour": t - 1})
        m.add_constraints(
            soc_bess.sel({"hour": t})
            == soc_p + (b.eta_charge * p_bess_c.sel({"hour": t})
                        - p_bess_d.sel({"hour": t}) / b.eta_discharge) * dt,
            name=f"bess_soc_{t}"
        )

    # BESS binary exclusion
    m.add_constraints(p_bess_c <= b.power_kw * y_bess_c, name="bess_c_ub")
    m.add_constraints(p_bess_d <= b.power_kw * y_bess_d, name="bess_d_ub")
    m.add_constraints(y_bess_c + y_bess_d <= 1, name="bess_excl")

    # CHP coupling
    ratio = c.capacity_heat_kw / c.capacity_elec_kw
    m.add_constraints(p_chp_h == ratio * p_chp_e, name="chp_coupling")
    m.add_constraints(p_chp_e <= c.capacity_elec_kw * u_chp, name="chp_ub")
    m.add_constraints(p_chp_e >= c.min_load * c.capacity_elec_kw * u_chp, name="chp_lb")

    # CHP startup
    for t in range(params.n_hours):
        u_prev = 0 if t == 0 else u_chp.sel({"hour": t - 1})
        m.add_constraints(
            z_chp.sel({"hour": t}) >= u_chp.sel({"hour": t}) - u_prev,
            name=f"chp_startup_{t}"
        )

    # Thermal storage dynamics
    for t in range(params.n_hours):
        soc_p = th.soc_init * th.capacity_kwh if t == 0 else soc_th.sel({"hour": t - 1})
        heat_in = p_chp_h.sel({"hour": t}) + p_hp.sel({"hour": t})
        m.add_constraints(
            soc_th.sel({"hour": t})
            == th.eta_store * soc_p + (heat_in - d_heat[t]) * dt,
            name=f"heat_bal_{t}"
        )

    # ── Solve ──────────────────────────────────────────────────────────────────
    m.solve(solver_name="highs", io_api="direct")

    sol = m.solution
    hourly = pd.DataFrame({
        "hour":         list(range(params.n_hours)),
        "p_import_kw":  sol["p_import"].values,
        "p_export_kw":  sol["p_export"].values,
        "p_bess_c_kw":  sol["p_bess_c"].values,
        "p_bess_d_kw":  sol["p_bess_d"].values,
        "soc_bess_kwh": sol["soc_bess"].values,
        "p_chp_e_kw":   sol["p_chp_e"].values,
        "p_chp_h_kw":   sol["p_chp_h"].values,
        "u_chp":        sol["u_chp"].values,
        "p_hp_kw":      sol["p_hp"].values,
        "soc_th_kwh":   sol["soc_th"].values,
        "p_pv_kw":      sol["p_pv"].values,
    })

    return {
        "framework": "Linopy",
        "mode": "deterministic",
        "status": str(m.status),
        "objective_eur": float(m.objective.value),
        "hourly": hourly,
    }


# ─── Stochastic model ────────────────────────────────────────────────────────

def build_and_solve_stochastic(
    scenarios: dict[str, np.ndarray],
    params: SystemParams = DEFAULT_PARAMS,
) -> dict:
    """
    Two-stage stochastic MILP with CVaR via Linopy.

    Stage-1:  CHP binary commitment u_chp[t]
    Stage-2:  Continuous recourse per scenario s

    Objective: E[cost_s] + λ · CVaR_α

    KEY: Use named pd.Index coords ("hour", "scenario") so Linopy never
    conflates the two integer axes in an outer join when n_hours == n_scenarios.
    """
    # Named coords — critical to avoid dim-collision outer-join bug
    T  = pd.Index(range(params.n_hours), name="hour")
    n_s = params.n_scenarios
    S  = pd.Index(range(n_s), name="scenario")

    b = params.bess
    c = params.chp
    hp = params.hp
    th = params.thermal
    g = params.grid
    dt = params.dt
    alpha = params.cvar_alpha
    lam = params.risk_weight

    probs   = scenarios["probabilities"]
    pv_sc   = scenarios["pv"]           # (n_s, 24)
    d_elec  = scenarios["demand_elec"]  # (n_s, 24)
    price   = scenarios["price"] / 1000.0
    heat_sc = scenarios["heat"]         # (n_s, 24)

    m = linopy.Model()

    # ── Stage-1: CHP binary commitment (hour only) ────────────────────────────
    u_chp = m.add_variables(binary=True, coords=[T], name="u_chp")
    z_chp = m.add_variables(binary=True, coords=[T], name="z_chp")

    # ── Stage-2: recourse variables indexed by (hour, scenario) ──────────────
    TS = [T, S]
    p_import  = m.add_variables(lower=0, upper=g.max_import_kw,    coords=TS, name="p_import")
    p_export  = m.add_variables(lower=0, upper=g.max_export_kw,    coords=TS, name="p_export")
    p_bess_c  = m.add_variables(lower=0, upper=b.power_kw,         coords=TS, name="p_bess_c")
    p_bess_d  = m.add_variables(lower=0, upper=b.power_kw,         coords=TS, name="p_bess_d")
    y_bess_c  = m.add_variables(binary=True,                       coords=TS, name="y_bess_c")
    y_bess_d  = m.add_variables(binary=True,                       coords=TS, name="y_bess_d")
    soc_bess  = m.add_variables(
        lower=b.soc_min * b.capacity_kwh,
        upper=b.soc_max * b.capacity_kwh,
        coords=TS, name="soc_bess",
    )
    p_chp_e   = m.add_variables(lower=0, upper=c.capacity_elec_kw, coords=TS, name="p_chp_e")
    p_chp_h   = m.add_variables(lower=0, upper=c.capacity_heat_kw, coords=TS, name="p_chp_h")
    p_hp_v    = m.add_variables(lower=0, upper=hp.capacity_kw,     coords=TS, name="p_hp")
    soc_th    = m.add_variables(
        lower=th.soc_min * th.capacity_kwh,
        upper=th.soc_max * th.capacity_kwh,
        coords=TS, name="soc_th",
    )
    p_pv      = m.add_variables(lower=0, coords=TS, name="p_pv")

    # CVaR auxiliary variables
    eta = m.add_variables(coords=None, name="eta")          # scalar VaR estimate
    phi = m.add_variables(lower=0, coords=[S], name="phi")  # excess loss per scenario

    # helper: select a single (hour=t, scenario=s) scalar variable
    def hs(var, t, s):
        return var.sel({"hour": t, "scenario": s})

    # ── Constraints (loop over scenarios for dynamics; vectorise where safe) ──

    for s in range(n_s):
        s_sel = {"scenario": s}

        # PV availability
        m.add_constraints(
            p_pv.sel(s_sel) <= pv_sc[s],
            name=f"pv_lim_{s}",
        )

        # Electrical balance (vectorised over hours for this scenario)
        m.add_constraints(
            p_pv.sel(s_sel) + p_import.sel(s_sel)
            + p_bess_d.sel(s_sel) + p_chp_e.sel(s_sel)
            == d_elec[s] + p_export.sel(s_sel)
            + p_bess_c.sel(s_sel) + p_hp_v.sel(s_sel) / hp.cop,
            name=f"elec_bal_{s}",
        )

        # CHP commitment linking (vectorised over hours)
        m.add_constraints(
            p_chp_e.sel(s_sel) <= c.capacity_elec_kw * u_chp,
            name=f"chp_ub_{s}",
        )
        m.add_constraints(
            p_chp_e.sel(s_sel) >= c.min_load * c.capacity_elec_kw * u_chp,
            name=f"chp_lb_{s}",
        )

        # Time-coupled dynamics (must loop over t)
        for t in range(params.n_hours):
            # BESS SoC
            soc_p = b.soc_init * b.capacity_kwh if t == 0 else hs(soc_bess, t - 1, s)
            m.add_constraints(
                hs(soc_bess, t, s)
                == soc_p + (b.eta_charge * hs(p_bess_c, t, s)
                            - hs(p_bess_d, t, s) / b.eta_discharge) * dt,
                name=f"bess_soc_{t}_{s}",
            )
            # Thermal SoC
            th_sp = th.soc_init * th.capacity_kwh if t == 0 else hs(soc_th, t - 1, s)
            heat_in = hs(p_chp_h, t, s) + hs(p_hp_v, t, s)
            m.add_constraints(
                hs(soc_th, t, s)
                == th.eta_store * th_sp + (heat_in - float(heat_sc[s, t])) * dt,
                name=f"heat_bal_{t}_{s}",
            )

    # BESS binary exclusion (vectorised over both dims)
    m.add_constraints(p_bess_c <= b.power_kw * y_bess_c, name="bess_c_ub")
    m.add_constraints(p_bess_d <= b.power_kw * y_bess_d, name="bess_d_ub")
    m.add_constraints(y_bess_c + y_bess_d <= 1, name="bess_excl")

    # CHP heat–power coupling (vectorised)
    ratio = c.capacity_heat_kw / c.capacity_elec_kw
    m.add_constraints(p_chp_h == ratio * p_chp_e, name="chp_coupling")

    # CHP startup indicator
    for t in range(params.n_hours):
        u_prev = 0 if t == 0 else u_chp.sel({"hour": t - 1})
        m.add_constraints(
            u_chp.sel({"hour": t}) - u_prev <= z_chp.sel({"hour": t}),
            name=f"chp_startup_{t}",
        )

    # ── CVaR constraints + objective ──────────────────────────────────────────
    exp_cost_terms = []
    for s in range(n_s):
        pr_s    = float(probs[s])
        price_s = price[s].tolist()    # list of 24 floats

        energy_terms = sum(
            (price_s[t] + g.grid_fee) * hs(p_import, t, s)
            - g.feed_in_tariff * hs(p_export, t, s)
            for t in range(params.n_hours)
        ) * dt

        gas_terms = sum(
            hs(p_chp_e, t, s) * (c.gas_price / c.eta_elec)
            for t in range(params.n_hours)
        ) * dt

        wear_terms = sum(
            b.degradation_cost * (hs(p_bess_c, t, s) + hs(p_bess_d, t, s))
            for t in range(params.n_hours)
        ) * dt

        startup_terms = sum(
            c.startup_cost * z_chp.sel({"hour": t})
            for t in range(params.n_hours)
        )

        cost_s = energy_terms + gas_terms + wear_terms + startup_terms

        # phi_s >= cost_s - eta  (CVaR Rockafellar-Uryasev)
        m.add_constraints(
            phi.sel({"scenario": s}) >= cost_s - eta,
            name=f"cvar_{s}",
        )

        exp_cost_terms.append(pr_s * cost_s)

    expected_cost = sum(exp_cost_terms)
    cvar_term = eta + (1.0 / (1.0 - alpha)) * sum(
        float(probs[s]) * phi.sel({"scenario": s}) for s in range(n_s)
    )
    m.add_objective(expected_cost + lam * cvar_term)

    # ── Solve ──────────────────────────────────────────────────────────────────
    m.solve(solver_name="highs", io_api="direct")

    sol = m.solution
    T_list = list(range(params.n_hours))
    chp_sched = sol["u_chp"].values   # shape (24,)

    return {
        "framework": "Linopy",
        "mode": "stochastic",
        "status": str(m.status),
        "objective": float(m.objective.value),
        "chp_schedule": chp_sched.tolist(),
        "hourly_mean": pd.DataFrame({
            "hour":         T_list,
            "p_import_kw":  sol["p_import"].mean(dim="scenario").values,
            "p_export_kw":  sol["p_export"].mean(dim="scenario").values,
            "p_bess_c_kw":  sol["p_bess_c"].mean(dim="scenario").values,
            "p_bess_d_kw":  sol["p_bess_d"].mean(dim="scenario").values,
            "soc_bess_kwh": sol["soc_bess"].mean(dim="scenario").values,
            "p_chp_e_kw":   sol["p_chp_e"].mean(dim="scenario").values,
            "p_hp_kw":      sol["p_hp"].mean(dim="scenario").values,
            "u_chp":        chp_sched,
        }),
    }
