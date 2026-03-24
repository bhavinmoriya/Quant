"""
system_params.py
----------------
Central configuration for the hybrid energy system components.
All physical and economic parameters in one place — easy to swap for
real asset data or sensitivity studies.
"""

from dataclasses import dataclass, field


@dataclass
class BESSParams:
    """Battery Energy Storage System (BESS) parameters."""
    capacity_kwh: float = 100.0       # Usable energy capacity [kWh]
    power_kw: float = 50.0            # Max charge / discharge power [kW]
    eta_charge: float = 0.95          # Charging efficiency
    eta_discharge: float = 0.95       # Discharging efficiency
    soc_min: float = 0.10             # Minimum state-of-charge (10%)
    soc_max: float = 0.90             # Maximum state-of-charge (90%)
    soc_init: float = 0.50            # Initial SoC (50%)
    degradation_cost: float = 0.05    # €/kWh cycled (wear cost)


@dataclass
class PVParams:
    """Photovoltaic system parameters."""
    capacity_kw: float = 50.0         # Installed peak capacity [kWp]
    inverter_efficiency: float = 0.97


@dataclass
class CHPParams:
    """Combined Heat and Power unit parameters."""
    capacity_elec_kw: float = 30.0    # Max electrical output [kW]
    capacity_heat_kw: float = 60.0    # Max thermal output [kW]
    eta_elec: float = 0.35            # Electrical efficiency
    eta_heat: float = 0.55            # Thermal (heat recovery) efficiency
    gas_price: float = 0.08           # €/kWh gas
    min_load: float = 0.40            # Min load fraction when ON (MILP)
    startup_cost: float = 5.0         # € per startup


@dataclass
class HeatPumpParams:
    """Heat pump parameters."""
    capacity_kw: float = 20.0         # Max thermal output [kW]
    cop: float = 3.5                  # Coefficient of Performance
    min_load: float = 0.0             # Modulating heat pump


@dataclass
class ThermalStorageParams:
    """Hot water / thermal buffer tank."""
    capacity_kwh: float = 80.0        # [kWh thermal]
    eta_store: float = 0.98           # Storage efficiency per hour (loss)
    soc_min: float = 0.10
    soc_max: float = 0.95
    soc_init: float = 0.50


@dataclass
class EVParams:
    """Aggregated EV fleet (Vehicle-to-Grid capable)."""
    n_vehicles: int = 10
    battery_kwh: float = 60.0         # Per vehicle
    max_charge_kw: float = 11.0       # Per vehicle (AC)
    max_discharge_kw: float = 7.4     # V2G discharge per vehicle
    eta_charge: float = 0.92
    soc_min: float = 0.20             # Must leave with ≥20%
    soc_departure: float = 0.80       # Target SoC at departure (8h)


@dataclass
class GridParams:
    """Grid connection parameters."""
    max_import_kw: float = 100.0
    max_export_kw: float = 80.0
    feed_in_tariff: float = 0.08      # €/kWh for export (FIT)
    grid_fee: float = 0.05            # €/kWh grid usage fee (on import)
    co2_intensity: float = 0.4        # kgCO2/kWh (grid average)


@dataclass
class SystemParams:
    """Aggregate all component parameters."""
    bess: BESSParams = field(default_factory=BESSParams)
    pv: PVParams = field(default_factory=PVParams)
    chp: CHPParams = field(default_factory=CHPParams)
    hp: HeatPumpParams = field(default_factory=HeatPumpParams)
    thermal: ThermalStorageParams = field(default_factory=ThermalStorageParams)
    ev: EVParams = field(default_factory=EVParams)
    grid: GridParams = field(default_factory=GridParams)

    # Optimization horizon
    n_hours: int = 24
    dt: float = 1.0    # Time step [hours]

    # Stochastic
    n_scenarios: int = 100
    cvar_alpha: float = 0.95          # CVaR confidence level
    risk_weight: float = 0.3          # λ in: min E[cost] + λ·CVaR


DEFAULT_PARAMS = SystemParams()
