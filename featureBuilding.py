"""
Phase 2: Data Engineering & Feature Building
--------------------------------------------------
Goal: 
1. Integrate Weather & Load Data (the fundamental drivers of electricity prices).
2. Merge disparately timed datasets using Polars.
3. Build the core Features for our future XGBoost model.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Plotting configuration
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

def simulate_historical_data(days: int = 30) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Simulates a month of PJM Pricing and localized Weather/Load data.
    In a real production environment, you would hit the PyISO API for Load
    and the NOAA/Meteostat API for weather.
    """
    print(f"[*] Simulating {days} days of historical Energy & Weather data...")
    hours = days * 24
    timestamps = [datetime(2023, 7, 1) + timedelta(hours=i) for i in range(hours)]
    
    # --- 1. Weather & Load Data (The Drivers) ---
    # Temperature follows a daily curve (hotter in afternoon) and a slow weekly wave.
    daily_temp = 75 + 15 * np.sin(np.linspace(0, days * 2 * np.pi, hours)) 
    weekly_wave = 10 * np.sin(np.linspace(0, (days/7) * 2 * np.pi, hours))
    temperature = daily_temp + weekly_wave + np.random.normal(0, 3, hours)
    
    # Grid Load is highly correlated with temperature (cooling degrees)
    # If temp > 80, air conditioners turn on, load spikes non-linearly.
    cooling_demand = np.maximum(0, temperature - 75) ** 1.5 
    grid_load = 80000 + (cooling_demand * 500) + np.random.normal(0, 1000, hours)
    
    df_weather = pl.DataFrame({
        "timestamp": timestamps,
        "temperature_f": temperature,
        "grid_load_mw": grid_load
    })
    
    # --- 2. Pricing Data (The Target) ---
    base_price = 30.0
    # DA Market tries to guess the load based on weather forecasts
    da_prices = base_price + (grid_load - 80000) * 0.001 + np.random.normal(0, 5, hours)
    
    # RT Market reacts to actual grid stress. If load > 100,000 MW, prices go exponential.
    rt_prices = da_prices.copy()
    high_stress_mask = grid_load > 105000
    rt_prices[high_stress_mask] += np.random.exponential(50, size=np.sum(high_stress_mask))
    
    df_price = pl.DataFrame({
        "timestamp": timestamps,
        "da_lmp": da_prices,
        "rt_lmp": rt_prices
    })
    
    return df_price, df_weather

def build_features(df_main: pl.DataFrame) -> pl.DataFrame:
    """
    This is the most critical step for an Algorithmic Trader.
    We transform raw data into features that XGBoost can learn from.
    """
    print("\n[*] Engineering Features for XGBoost...")
    
    # Calculate our Target (The Spread)
    df = df_main.with_columns(
        (pl.col("rt_lmp") - pl.col("da_lmp")).alias("target_spread")
    )
    
    # 1. Temporal Features
    # Energy usage is highly dependent on human behavior (working vs sleeping)
    df = df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        (pl.col("timestamp").dt.weekday() >= 6).cast(pl.Int32).alias("is_weekend")
    ])
    
    # 2. Weather Engineering
    # Heating Degree Days (HDD) / Cooling Degree Days (CDD)
    # The grid doesn't care about 60F vs 65F. It cares about 60F vs 95F.
    # We create a non-linear feature for extreme heat.
    df = df.with_columns([
        pl.when(pl.col("temperature_f") > 80)
          .then(pl.col("temperature_f") - 80)
          .otherwise(0).alias("extreme_heat_degrees")
    ])
    
    # 3. Rolling / Momentum Features
    # Was the grid stressed yesterday at this exact hour? 
    # Because prices are hourly, a 24-hour shift looks at "yesterday same hour"
    df = df.with_columns([
        pl.col("target_spread").shift(24).alias("spread_yesterday"),
        pl.col("grid_load_mw").rolling_mean(window_size=24).alias("load_24h_avg")
    ])
    
    # Drop rows with nulls caused by shifting/rolling
    return df.drop_nulls()

def main():
    print("=== Algorithmic Trading Phase 2: Feature Engineering ===")
    
    # 1. Get raw data
    df_price, df_weather = simulate_historical_data(days=30)
    
    # 2. The fundamental Quantitative Join
    # In real life, weather might be every 15 mins, prices every hour. 
    # Polars `join_asof` handles these misalignments perfectly, but here we just do an exact join on the hour.
    print("[*] Merging Market Pricing with Weather Telemetry...")
    df_merged = df_price.join(df_weather, on="timestamp", how="inner")
    
    # 3. Feature Engineering
    df_features = build_features(df_merged)
    
    print("\n[+] Final Feature Matrix (Head):")
    # Display the columns we intend to feed to our ML model
    feature_cols = ["timestamp", "target_spread", "hour", "day_of_week", "extreme_heat_degrees", "load_24h_avg"]
    print(df_features.select(feature_cols).head(5))
    
    # 4. Visualizing the non-linear relationship
    print("\n[*] Plotting the non-linear relationship between Load and Pricing...")
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot showing how RT Price explodes when Load gets too high (The "Hockey Stick" curve)
    sns.scatterplot(
        data=df_features.to_pandas(), 
        x='grid_load_mw', 
        y='rt_lmp', 
        hue='extreme_heat_degrees',
        palette='YlOrRd',
        alpha=0.6
    )
    plt.title("The 'Hockey Stick' - Real-Time Prices Extrapolate at High Grid Stress")
    plt.xlabel("Grid Load (MW)")
    plt.ylabel("Real-Time Price ($/MWh)")
    plt.axvline(105000, color='red', linestyle='--', label="Grid Stress Threshold")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("phase_2_output.png")
    print("[+] Plot saved as 'phase_2_output.png'. You can view it in your Kaggle output directory.")

if __name__ == "__main__":
    main()
