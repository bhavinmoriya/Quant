# Hybrid Energy System MILP Optimizer
### Pyomo vs Linopy | Deterministic + Stochastic MILP with CVaR

A portfolio-grade Python project demonstrating optimization of a **hybrid energy system** comprising:

| Component | Symbol | Role |
|-----------|--------|------|
| Photovoltaic (PV) | `pv` | Renewable generation (curtailable) |
| Battery Storage (BESS) | `bess` | Energy arbitrage & peak shaving |
| Combined Heat & Power (CHP) | `chp` | Dispatchable heat + electricity (binary on/off) |
| Heat Pump (HP) | `hp` | Flexible heat demand (converts electricity) |
| Thermal Storage | `th` | Heat buffer decoupling production from demand |
| EV Fleet | `ev` | Smart charging load |
| Grid | `grid` | Import/export with day-ahead prices |

---

## Mathematical Formulation

### Deterministic MILP

**Decision variables** (per hour *t*):

| Variable | Domain | Description |
|----------|--------|-------------|
| `p_import[t]`, `p_export[t]` | ℝ₊ | Grid exchange [kW] |
| `p_bess_c[t]`, `p_bess_d[t]` | ℝ₊ | BESS charge/discharge [kW] |
| `y_bess_c[t]`, `y_bess_d[t]` | {0,1} | Binary: prevent simultaneous C/D |
| `soc_bess[t]` | ℝ₊ | BESS state-of-charge [kWh] |
| `p_chp_e[t]`, `p_chp_h[t]` | ℝ₊ | CHP electric & heat output [kW] |
| `u_chp[t]` | {0,1} | CHP on/off commitment |
| `z_chp[t]` | {0,1} | CHP startup indicator |
| `p_hp[t]` | ℝ₊ | Heat pump thermal output [kW] |
| `soc_th[t]` | ℝ₊ | Thermal storage SoC [kWh] |
| `p_pv[t]` | ℝ₊ | PV used (≤ available) [kW] |

**Objective** — minimise total operating cost:

```
min Σ_t [ (λ_t + c_grid) · p_import_t − FIT · p_export_t
        + (p_chp_e_t / η_e^CHP) · c_gas
        + c_deg · (p_bess_c_t + p_bess_d_t)
        + c_start · z_chp_t ] · Δt
```

**Key constraints:**

- **Electricity balance**: PV + BESS_d + CHP_e + import = demand + BESS_c + HP/COP + export
- **BESS SoC dynamics**: `soc[t] = soc[t-1] + (η_c · P_c - P_d/η_d) · Δt`
- **BESS binary exclusion**: `y_c + y_d ≤ 1`
- **CHP min-load**: `P_min · u_chp ≤ p_chp_e ≤ P_max · u_chp`
- **CHP startup detection**: `z_chp[t] ≥ u_chp[t] − u_chp[t-1]`
- **Thermal balance**: `soc_th[t] = η_th · soc_th[t-1] + (CHP_h + HP − D_heat) · Δt`

### Stochastic MILP with CVaR (Two-Stage)

**Stage 1 (here-and-now)**: CHP commitment `u_chp[t]` — decided before uncertainty reveals

**Stage 2 (recourse)**: Continuous dispatch per scenario *s*

**Scenarios**: Monte Carlo draws for PV, electrical demand, and electricity price
(truncated normal with ±20% / ±10% / ±15% uncertainty respectively)

**Objective with CVaR (Rockafellar-Uryasev linearisation)**:

```
min  E_s[cost_s] + λ · CVaR_α[cost_s]

CVaR_α = η + 1/(1-α) · Σ_s prob_s · φ_s
s.t.  φ_s ≥ cost_s - η,   φ_s ≥ 0
```

where *α* = 0.95, *λ* = 0.3 (configurable in `system_params.py`).

---

## Project Structure

```
hybrid_energy_optimizer/
├── data_generator.py   # Synthetic profiles + Monte Carlo scenarios
├── system_params.py    # All physical & economic parameters (dataclasses)
├── model_pyomo.py      # Pyomo MILP: deterministic + stochastic
├── model_linopy.py     # Linopy MILP: deterministic + stochastic
├── visualizer.py       # Matplotlib publication plots
├── main.py             # Orchestration runner
├── plots/              # Generated PNG figures
└── results/            # CSV result tables
```

---

## Installation

```bash
pip install pyomo linopy highspy pandas numpy matplotlib scipy
# GLPK (fallback solver for Pyomo):
sudo apt install glpk-utils   # Ubuntu/Debian
brew install glpk              # macOS
```

---

## Usage

```bash
# Quick run (30 scenarios)
python main.py

# Full run (100 scenarios, slower but statistically robust)
python main.py --n_scenarios 100

# Custom
python main.py --n_scenarios 50
```

---

## Framework Comparison

| Feature | Pyomo | Linopy |
|---------|-------|--------|
| Syntax style | Constraint rules (Pythonic OOP) | Array-indexed (NumPy-like) |
| Data integration | Manual loops | Native xarray/pandas |
| Solver backends | GLPK, CBC, CPLEX, Gurobi, HiGHS | HiGHS (primary), others via API |
| Model build time | Moderate (Python overhead) | Fast (vectorised) |
| Stochastic scaling | Straightforward indexing | Very fast (xarray broadcast) |
| Best for | Flexibility, custom constraints | Large-scale LP/MILP, speed |

---

## Results Produced

| File | Description |
|------|-------------|
| `plots/pv_scenarios.png` | Fan chart of 200 PV scenarios |
| `plots/dispatch_pyomo_det.png` | Hourly dispatch stack (Pyomo) |
| `plots/dispatch_linopy_det.png` | Hourly dispatch stack (Linopy) |
| `plots/soc_pyomo_det.png` | BESS + thermal storage SoC |
| `plots/framework_comparison.png` | Cost & solve-time bar chart |
| `plots/cost_distribution_pyomo.png` | MC cost histogram with CVaR |
| `results/summary.csv` | Objective values + solve times |
| `results/hourly_*.csv` | Per-hour dispatch tables |

---

## Author

Bhavin Moriya  
[LinkedIn](https://www.linkedin.com/in/bhavin-moriya-b0b88b2/) | [GitHub](https://github.com/bhavinmoriya)

Ph.D. Mathematics | Quantitative Finance & Energy Systems Optimization
