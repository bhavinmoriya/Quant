"""
Phase 4: Backtesting & Risk Management
--------------------------------------------------
Goal: 
1. Convert ML signals (+1, -1) into a simulated portfolio.
2. Apply Transaction Costs (Friction).
3. Calculate Risk Metrics: Sharpe Ratio, Maximum Drawdown, Hit Rate.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import xgboost as xgb

# Plotting configuration
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

# --- 1. Generate Fake Data & ML Signals (Condensed from Phase 2 & 3) ---
def get_ml_signals(days: int = 90) -> pl.DataFrame:
    print(f"[*] Simulating {days} days of Energy Data & XGBoost Signals...")
    hours = days * 24
    timestamps = [datetime(2023, 5, 1) + timedelta(hours=i) for i in range(hours)]
    
    # Simulate Prices
    da_prices = 35.0 + 10 * np.sin(np.linspace(0, days * 2 * np.pi, hours)) + np.random.normal(0, 4, hours)
    rt_prices = da_prices + np.random.normal(0, 8, hours)
    
    # Inject some extreme volatility for the ML model to "find"
    spike_indices = np.random.choice(range(hours), size=int(hours * 0.05), replace=False)
    rt_prices[spike_indices] += np.random.exponential(60, size=len(spike_indices))
    
    # Create DataFrame
    df = pl.DataFrame({
        "timestamp": timestamps,
        "da_lmp": da_prices,
        "rt_lmp": rt_prices
    }).with_columns([
        (pl.col("rt_lmp") - pl.col("da_lmp")).alias("actual_spread")
    ])
    
    # Simulate a decent ML model's predictions (correlated with reality, but imperfect)
    # A perfect model would have correlation = 1.0. We simulate ~0.4 correlation.
    predictions = df["actual_spread"].to_numpy() * 0.4 + np.random.normal(0, 10, hours)
    
    # Generate Signals with a $2.00 Conviction Threshold
    signals = np.zeros(hours)
    signals[predictions > 2.0] =  1 # DEC (Buy RT)
    signals[predictions < -2.0] = -1 # INC (Buy DA)
    
    return df.with_columns([
        pl.Series("ml_forecast", predictions),
        pl.Series("signal", signals)
    ])

# --- 2. The Vectorized Backtester ---
def run_backtest(df: pl.DataFrame, trade_size_mw: float = 10.0, fee_per_mwh: float = 0.50):
    """
    Simulates trading the signals over history.
    trade_size_mw: How many Megawatts we trade per signal.
    fee_per_mwh: Clearing fees paid to the ISO and clearing broker.
    """
    print("\n[*] Running Vectorized Backtest...")
    print(f"    -> Trade Size: {trade_size_mw} MW")
    print(f"    -> Transaction Fee: ${fee_per_mwh} per MWh")
    
    # Vectorized PnL Calculation
    # DEC (+1): Profit = (RT - DA) * MW
    # INC (-1): Profit = (DA - RT) * MW  --> which is -(RT - DA) * MW
    # So PnL completely vectorizes as: signal * actual_spread * MW
    
    df = df.with_columns([
        (pl.col("signal") * pl.col("actual_spread") * trade_size_mw).alias("gross_pnl_hour")
    ])
    
    # Apply Transaction Costs (Friction)
    # We only pay fees when we actually trade (signal != 0)
    df = df.with_columns([
        (pl.col("signal").abs() * trade_size_mw * fee_per_mwh).alias("fees_paid")
    ])
    
    # Net PnL = Gross - Fees
    df = df.with_columns([
        (pl.col("gross_pnl_hour") - pl.col("fees_paid")).alias("net_pnl_hour")
    ])
    
    # Calculate Cumulative Returns over the simulation
    df = df.with_columns([
        pl.col("net_pnl_hour").cum_sum().alias("cumulative_pnl")
    ])
    
    return df

# --- 3. Risk & Performance Metrics ---
def calculate_metrics(df: pl.DataFrame):
     # Convert to Pandas for easier math operations
    pdf = df.to_pandas()
    
    # 1. Basic Stats
    total_hours = len(pdf)
    trades_taken = int(pdf['signal'].abs().sum())
    gross_pnl = pdf['gross_pnl_hour'].sum()
    total_fees = pdf['fees_paid'].sum()
    net_pnl = pdf['cumulative_pnl'].iloc[-1]
    
    # 2. Hit Rate (Win/Loss ratio of trades taken)
    winning_trades = len(pdf[pdf['net_pnl_hour'] > 0])
    losing_trades = len(pdf[pdf['net_pnl_hour'] < 0])
    hit_rate = winning_trades / trades_taken if trades_taken > 0 else 0
    
    # 3. Maximum Drawdown (The most you would have lost from a peak)
    # Peak running max
    running_max = np.maximum.accumulate(pdf['cumulative_pnl'])
    drawdowns = pdf['cumulative_pnl'] - running_max
    max_drawdown = drawdowns.min()
    
    # 4. Sharpe Ratio (Annualized)
    # Measures return adjusted for risk (volatility)
    # Assuming risk-free rate is roughly 0 for simplicity
    hourly_returns = pdf[pdf['signal'] != 0]['net_pnl_hour']
    if len(hourly_returns) > 0 and hourly_returns.std() > 0:
        # Annualization factor for hourly energy trading = sqrt(24 * 365)
        sharpe = (hourly_returns.mean() / hourly_returns.std()) * np.sqrt(8760)
    else:
        sharpe = 0.0
        
    print("\n" + "="*40)
    print("      BACKTEST PERFORMANCE TEAR SHEET")
    print("="*40)
    print(f"Total Simulation Time : {total_hours} hours")
    print(f"Trades Executed       : {trades_taken}")
    print(f"Hit Rate (Win %)      : {hit_rate*100:.1f}%")
    print("-" * 40)
    print(f"Gross PnL             : ${gross_pnl:,.2f}")
    print(f"Fees Paid             : ${total_fees:,.2f}")
    print(f"Net PnL               : ${net_pnl:,.2f}")
    print("-" * 40)
    print(f"Max Drawdown          : ${max_drawdown:,.2f}")
    print(f"Sharpe Ratio (Ann.)   : {sharpe:.2f}")
    print("="*40)
    
    return pdf, max_drawdown

def main():
    # 1. Get Signals
    df_signals = get_ml_signals(days=120)
    
    # 2. Run Vectorized Backtest
    df_results = run_backtest(df_signals, trade_size_mw=10.0, fee_per_mwh=0.50)
    
    # 3. Calculate Risk Metrics
    pdf, mdd = calculate_metrics(df_results)
    
    # 4. Visualize the Equity Curve
    print("\n[*] Plotting Equity Curve & Drawdowns...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot 1: Cumulative Net PnL (The Equity Curve)
    ax1.plot(pdf['timestamp'], pdf['cumulative_pnl'], color='green', linewidth=2, label="Cumulative PnL")
    ax1.fill_between(pdf['timestamp'], pdf['cumulative_pnl'], 0, color='green', alpha=0.1)
    
    # Highlight highest peaks to visualize drawdowns
    running_max = np.maximum.accumulate(pdf['cumulative_pnl'])
    ax1.plot(pdf['timestamp'], running_max, color='gray', linestyle='--', alpha=0.6, label="High Water Mark")
    
    ax1.set_title("Algorithmic Trading Portfolio: Equity Curve", fontsize=14)
    ax1.set_ylabel("Net Profit ($)")
    ax1.legend(loc="upper left")
    
    # Plot 2: Drawdown Chart (Underwater Chart)
    drawdowns = pdf['cumulative_pnl'] - running_max
    ax2.fill_between(pdf['timestamp'], drawdowns, 0, color='red', alpha=0.3)
    ax2.plot(pdf['timestamp'], drawdowns, color='red', linewidth=1)
    
    ax2.set_title("Drawdown Depth (Underwater)", fontsize=12)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Date")
    
    plt.tight_layout()
    plt.savefig("phase_4_output.png")
    print("[+] Plot saved as 'phase_4_output.png'. You can view it in your Kaggle output directory.")
    print("\n[+] Phase 4 Complete! You now possess a professional quant backtesting framework.")

if __name__ == "__main__":
    main()
