"""
Phase 3: Machine Learning Model (XGBoost)
--------------------------------------------------
Goal:
1. Train an XGBoost Regressor on our engineered features.
2. Predict the DA/RT Spread.
3. Generate actionable Trading Signals (Long/Short/Flat).
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Plotting configuration
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

# --- 1. Recreate the Phase 2 Feature Pipeline (Standalone Script) ---
def get_engineered_data(days: int = 60) -> pl.DataFrame:
    print(f"[*] Simulating {days} days of historical Energy & Weather data...")
    hours = days * 24
    timestamps = [datetime(2023, 6, 1) + timedelta(hours=i) for i in range(hours)]

    # Weather & Load
    temperature = 75 + 15 * np.sin(np.linspace(0, days * 2 * np.pi, hours)) + \
                  10 * np.sin(np.linspace(0, (days/7) * 2 * np.pi, hours)) + np.random.normal(0, 3, hours)
    cooling_demand = np.maximum(0, temperature - 75) ** 1.5
    grid_load = 80000 + (cooling_demand * 500) + np.random.normal(0, 1000, hours)

    # Prices
    da_prices = 30.0 + (grid_load - 80000) * 0.001 + np.random.normal(0, 5, hours)
    rt_prices = da_prices.copy()
    high_stress_mask = grid_load > 105000
    rt_prices[high_stress_mask] += np.random.exponential(50, size=np.sum(high_stress_mask))

    df = pl.DataFrame({
        "timestamp": timestamps,
        "da_lmp": da_prices,
        "rt_lmp": rt_prices,
        "temperature_f": temperature,
        "grid_load_mw": grid_load
    })

    # Feature Engineering
    df = df.with_columns([
        (pl.col("rt_lmp") - pl.col("da_lmp")).alias("target_spread"),
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        (pl.col("timestamp").dt.weekday() >= 6).cast(pl.Int32).alias("is_weekend"),
        pl.when(pl.col("temperature_f") > 80).then(pl.col("temperature_f") - 80).otherwise(0).alias("extreme_heat")
    ])

    # Rolling features
    df = df.with_columns([
        pl.col("target_spread").shift(24).alias("spread_yesterday"),
        pl.col("grid_load_mw").rolling_mean(window_size=24).alias("load_24h_avg")
    ])

    return df.drop_nulls()

# --- 2. Train the XGBoost Model ---
def run_xgboost_strategy(df: pl.DataFrame):
    print("\n[*] Preparing Data for XGBoost...")

    # Split: First 45 days for Training, Last 15 days for Test (Out-of-Sample)
    train_size = int(len(df) * 0.75)
    train_df = df.slice(0,train_size)
    test_df = df.slice(train_size)

    val_size = int(len(test_df) * 0.5)
    val_df = test_df.slice(0,val_size)
    test_df = test_df.slice(val_size)

    # Features (X) and Target (y)
    features = ["hour", "day_of_week", "is_weekend", "extreme_heat", "spread_yesterday", "load_24h_avg", "da_lmp"]

    X_train = train_df.select(features).to_pandas()
    y_train = train_df.select("target_spread").to_pandas().values.ravel()

    X_val, y_val = val_df.select(features).to_pandas(), val_df.select("target_spread").to_pandas().values.ravel()

    X_test = test_df.select(features).to_pandas()
    y_test = test_df.select("target_spread").to_pandas().values.ravel()

    print("[*] Training XGBoost Regressor...")
    # model = xgb.XGBRegressor(
    #     n_estimators=100,
    #     learning_rate=0.05,
    #     max_depth=4,
    #     objective='reg:squarederror',
    #     random_state=42
    # )

    # model.fit(X_train, y_train)

    model = xgb.XGBRegressor(
        base_score=0.5, 
        booster='gbtree', 
        n_estimators=5000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        # n_estimators=1000,
        # early_stopping_rounds=50,
        objective='reg:squarederror',
        max_depth=3,
        # learning_rate=0.01
    )
    
    model.fit(X_train, y_train,
            # eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_set=[(X_train, y_train), (X_val, y_val)],
            # eval_set=[(X_val, y_val)],
            verbose=100)

    # Make Predictions on the Out-of-Sample test set
    predictions = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, predictions)
    print(f"[+] Out-of-Sample Mean Absolute Error (MAE): ${mae:.2f}")

    # --- 3. Generate Trading Signals ---
    # We don't trade on tiny 50 cent spread predictions; we need a conviction threshold.
    # Cost to transact (Collateral fees, crossing the spread) might be ~$1.00/MWh
    # Let's say we only trade if the model expects > $2.00 profit.

    THRESHOLD = 2.0

    print(f"\n[*] Generating Signals (Conviction Threshold = ${THRESHOLD})")
    signals = np.zeros(len(predictions))

    # +1 means submit DEC bid (Buy DA, Sell RT). Profitable if RT > DA
    # -1 means submit INC bid (Sell DA, Buy RT). Profitable if DA > RT
    signals[predictions > THRESHOLD] = 1
    signals[predictions < -THRESHOLD] = -1

    total_trades = np.sum(np.abs(signals) > 0)
    print(f"[+] Generated {total_trades} trade signals out of {len(signals)} test hours.")

    # Attach predictions back to Polars dataframe for plotting
    test_df = test_df.with_columns([
        pl.Series("xgb_forecast", predictions),
        pl.Series("trading_signal", signals)
    ])

    return test_df, model

def main():
    print("=== Algorithmic Trading Phase 3: XGBoost Spread Forecasting ===")

    df = get_engineered_data(days=60)

    # Train model and generate signals on the out-of-sample data
    results_df, model = run_xgboost_strategy(df)

    print("\n[*] Plotting Model Performance and Trading Signals...")

    # We will plot the last 5 days (120 hours) to see it clearly
    plot_df = results_df.tail(120).to_pandas()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # Top Plot: Forecast vs Reality
    ax1.plot(plot_df['timestamp'], plot_df['target_spread'], label='Actual Spread (RT-DA)', color='black', alpha=0.5)
    ax1.plot(plot_df['timestamp'], plot_df['xgb_forecast'], label='XGBoost Forecast', color='orange', linewidth=2)
    ax1.set_title("XGBoost Out-of-Sample Spread Forecast")
    ax1.set_ylabel("Spread ($/MWh)")
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(2.0, color='blue', linestyle=':', alpha=0.5, label="+2 Threshold (DEC)")
    ax1.axhline(-2.0, color='red', linestyle=':', alpha=0.5, label="-2 Threshold (INC)")
    ax1.legend()

    # Bottom Plot: The Actual Prices and our Triggered Trades
    ax2.plot(plot_df['timestamp'], plot_df['da_lmp'], label='Day-Ahead Price', color='blue', linestyle='--', alpha=0.5)
    ax2.plot(plot_df['timestamp'], plot_df['rt_lmp'], label='Real-Time Price', color='red', alpha=0.7)

    # Overlay Long (DEC) and Short (INC) markers
    dec_trades = plot_df[plot_df['trading_signal'] == 1]
    inc_trades = plot_df[plot_df['trading_signal'] == -1]

    ax2.scatter(dec_trades['timestamp'], dec_trades['rt_lmp'], color='green', marker='^', s=100, label="DEC Signal (+1)")
    ax2.scatter(inc_trades['timestamp'], inc_trades['rt_lmp'], color='purple', marker='v', s=100, label="INC Signal (-1)")

    ax2.set_title("Trading Execution based on ML Signals")
    ax2.set_ylabel("Electricity Price ($/MWh)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("phase_3_output.png")
    print("[+] Plot saved as 'phase_3_output.png'. You can view it in your Kaggle output directory.")
    print("\n[+] Phase 3 Complete! You now have a working AI Trading Signal Generator.")

if __name__ == "__main__":
    main()
