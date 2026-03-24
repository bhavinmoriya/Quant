"""
data_generator.py
-----------------
Generates synthetic but realistic 24-hour profiles for a hybrid energy system:
  - PV generation (solar irradiance curve)
  - Electricity demand (residential + EV charging)
  - Heat demand (space heating + hot water)
  - Electricity spot prices (day-ahead market)
  - Monte Carlo scenario sets for stochastic optimization
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

RNG_SEED = 42


# ── Deterministic (base) profiles ────────────────────────────────────────────

def _hours() -> np.ndarray:
    return np.arange(24)


def pv_profile(capacity_kw: float = 50.0, seed: int = RNG_SEED) -> np.ndarray:
    """Bell-shaped PV generation curve peaking at noon."""
    rng = np.random.default_rng(seed)
    t = _hours()
    base = capacity_kw * np.maximum(0, np.sin(np.pi * (t - 6) / 12)) ** 1.5
    noise = rng.normal(0, 0.03 * capacity_kw, 24)
    return np.maximum(0, base + noise)


def demand_profile(peak_kw: float = 40.0, seed: int = RNG_SEED) -> np.ndarray:
    """Residential electricity demand with morning & evening peaks."""
    rng = np.random.default_rng(seed)
    t = _hours()
    morning = 0.6 * peak_kw * np.exp(-0.5 * ((t - 8) / 1.5) ** 2)
    evening = peak_kw * np.exp(-0.5 * ((t - 19) / 2.0) ** 2)
    base = 0.2 * peak_kw * np.ones(24)
    noise = rng.normal(0, 0.02 * peak_kw, 24)
    return np.maximum(0, morning + evening + base + noise)


def ev_charging_profile(
    n_vehicles: int = 10,
    capacity_kwh: float = 60.0,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """EV charging demand: vehicles arrive home 17-20h, need ~20 kWh each."""
    rng = np.random.default_rng(seed)
    profile = np.zeros(24)
    for _ in range(n_vehicles):
        arrival = int(rng.uniform(17, 21))
        energy_needed = rng.uniform(10, 30)          # kWh
        charge_rate = rng.uniform(3.7, 11.0)         # kW (slow / fast charger)
        hours_needed = int(np.ceil(energy_needed / charge_rate))
        for h in range(arrival, min(arrival + hours_needed, 24)):
            profile[h] += charge_rate
    return profile


def heat_demand_profile(peak_kw: float = 30.0, seed: int = RNG_SEED) -> np.ndarray:
    """Heat demand: high overnight, drops midday (solar gains)."""
    rng = np.random.default_rng(seed)
    t = _hours()
    base = peak_kw * (0.5 + 0.5 * np.cos(np.pi * (t - 3) / 12) ** 2)
    noise = rng.normal(0, 0.03 * peak_kw, 24)
    return np.maximum(0, base + noise)


def spot_price_profile(seed: int = RNG_SEED) -> np.ndarray:
    """Day-ahead electricity spot price [€/MWh] with peak pricing."""
    rng = np.random.default_rng(seed)
    t = _hours()
    base = 60 + 30 * np.sin(np.pi * (t - 6) / 12)
    peak1 = 40 * np.exp(-0.5 * ((t - 9) / 1.0) ** 2)
    peak2 = 50 * np.exp(-0.5 * ((t - 19) / 1.5) ** 2)
    noise = rng.normal(0, 5, 24)
    return np.maximum(10, base + peak1 + peak2 + noise)


def build_base_scenario(seed: int = RNG_SEED) -> pd.DataFrame:
    """Assemble a 24-row DataFrame of the deterministic base scenario."""
    pv = pv_profile(seed=seed)
    demand_elec = demand_profile(seed=seed)
    ev = ev_charging_profile(seed=seed)
    demand_heat = heat_demand_profile(seed=seed)
    price = spot_price_profile(seed=seed)

    return pd.DataFrame(
        {
            "hour": _hours(),
            "pv_kw": pv,
            "demand_elec_kw": demand_elec,
            "ev_demand_kw": ev,
            "demand_heat_kw": demand_heat,
            "price_eur_mwh": price,
        }
    )


# ── Monte Carlo scenario generation ──────────────────────────────────────────

def _truncated_normal(
    mean: np.ndarray,
    std_frac: float,
    n_scenarios: int,
    rng: np.random.Generator,
    lower: float = 0.0,
) -> np.ndarray:
    """Draw n_scenarios rows, each a 24-h profile perturbed around mean."""
    std = std_frac * np.abs(mean) + 1e-6
    scenarios = np.zeros((n_scenarios, 24))
    for h in range(24):
        a = (lower - mean[h]) / std[h]
        scenarios[:, h] = truncnorm.rvs(
            a, np.inf, loc=mean[h], scale=std[h], size=n_scenarios, random_state=rng
        )
    return scenarios


def generate_scenarios(
    n_scenarios: int = 200,
    seed: int = RNG_SEED,
    pv_uncertainty: float = 0.20,
    demand_uncertainty: float = 0.10,
    price_uncertainty: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Generate Monte Carlo scenarios for PV, total electrical demand, and price.

    Returns dict with keys 'pv', 'demand_elec', 'price', each shape (n_scenarios, 24).
    Probabilities are uniform: 1/n_scenarios each.
    """
    rng = np.random.default_rng(seed)
    base = build_base_scenario(seed=seed)

    pv_base = base["pv_kw"].values
    demand_base = (base["demand_elec_kw"] + base["ev_demand_kw"]).values
    price_base = base["price_eur_mwh"].values

    scenarios = {
        "pv": _truncated_normal(pv_base, pv_uncertainty, n_scenarios, rng),
        "demand_elec": _truncated_normal(demand_base, demand_uncertainty, n_scenarios, rng),
        "price": _truncated_normal(price_base, price_uncertainty, n_scenarios, rng, lower=-np.inf),
        "heat": _truncated_normal(base["demand_heat_kw"].values, demand_uncertainty, n_scenarios, rng),
        "probabilities": np.ones(n_scenarios) / n_scenarios,
    }
    return scenarios


# ── Convenience export ────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_base_scenario()
    print(df.to_string(index=False))
    sc = generate_scenarios(n_scenarios=5)
    print("\nPV scenario sample (5 × 24):\n", sc["pv"].round(1))
