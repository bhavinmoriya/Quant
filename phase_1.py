"""
Phase 1: Market Fundamentals & The DA/RT Spread
--------------------------------------------------
Goal: 
1. Fetch and process PJM grid data.
2. Understand the Day-Ahead (DA) and Real-Time (RT) markets.
3. Calculate the DA-RT Spread (the core alpha generator for Virtual Bidding).
"""

import polars as pl
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Plotting configuration
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

def fetch_pjm_data(market_type: str, start_date: str, end_date: str, pnode_id: int):
    """
    Fetches historical PJM LMP (Locational Marginal Pricing) data.
    """
    base_url = f"https://api.pjm.com/api/v1/{market_type}"
    
    # We don't use a key for basic public endpoints, though PJM sometimes throttles it.
    params = {
        'rowCount': 50000,
        'startRow': 1,
        'pnode_id': pnode_id,
        'datetime_beginning_ept': f"{start_date} 00:00 to {end_date} 23:59"
    }
    
    print(f"[*] Attempting to fetch {market_type} for node {pnode_id}...")
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json().get('items', [])
            if len(data) > 0:
                print(f"[+] Successfully fetched {len(data)} rows.")
                return pl.DataFrame(data)
            else:
                print("[-] API returned empty data.")
                return None
        else:
            print(f"[-] API Request failed with status: {response.status_code}")
            return None
    except Exception as e:
        print(f"[-] Error during fetch: {e}")
        return None

def generate_synthetic_data() -> pl.DataFrame:
    """
    If the API fails (e.g., due to Kaggle networking restrictions), this function 
    generates a mathematically accurate simulation of a 72-hour period in PJM.
    """
    print("\n[*] Generating synthetic high-volatility PJM pricing data instead...")
    hours = 72
    base_load = 35.0
    
    # Day-Ahead: smooth sine wave representing expected daily load curves
    da_prices = base_load + 15 * np.sin(np.linspace(0, 3 * 2 * np.pi, hours)) + np.random.normal(0, 3, hours)
    
    # Real-Time: spot market with random extreme spikes (generator trips, transmission congestion)
    rt_prices = da_prices + np.random.normal(0, 8, hours)
    rt_prices[15] += 120  # Generator trip (supply shock)
    rt_prices[40] -= 30   # Sudden wind surge (oversupply)
    rt_prices[65] += 200  # Transmission line limit reached (congestion spike)
    
    timestamps = [datetime(2023, 8, 14) + timedelta(hours=i) for i in range(hours)]
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "da_lmp": da_prices,
        "rt_lmp": rt_prices
    })

def main():
    print("=== Algorithmic Trading Phase 1: Market Fundamentals ===")
    
    # 1. Fetching Data
    START_DATE = "8/14/2023"
    END_DATE = "8/16/2023"
    TARGET_NODE = 51217  # PSEG Load Zone (NJ)
    
    df_da = fetch_pjm_data("da_hrl_lmps", START_DATE, END_DATE, TARGET_NODE)
    df_rt = fetch_pjm_data("rt_hrl_lmps", START_DATE, END_DATE, TARGET_NODE)
    
    # Use synthetic data if API fails to guarantee lesson continuation
    if df_da is None or df_rt is None:
        df = generate_synthetic_data()
    else:
        # Assuming successful API join (simplified for exact matching)
        # We drop the API's extra columns and merge cleanly
        df_da = df_da.select(["datetime_beginning_utc", "total_lmp_rt"]).rename({"total_lmp_rt": "da_lmp", "datetime_beginning_utc": "timestamp"})
        df_rt = df_rt.select(["datetime_beginning_utc", "total_lmp_rt"]).rename({"total_lmp_rt": "rt_lmp", "datetime_beginning_utc": "timestamp"})
        df = df_da.join(df_rt, on="timestamp")
    
    print("\n[Data Loaded via Polars]")
    print(df.head(5))
    
    # 2. Calculating the Arbitrage Spread (DA-RT)
    # The spread is exactly what we use Machine Learning to predict!
    print("\n[*] Calculating the DA/RT Spread...")
    df = df.with_columns([
        (pl.col("rt_lmp") - pl.col("da_lmp")).alias("rt_da_spread")
    ])
    
    print(df.select(["timestamp", "da_lmp", "rt_lmp", "rt_da_spread"]).head(5))

    # 3. Visualizing the Volatility
    print("\n[*] Plotting the pricing volatility and spread...")
    
    # Subplot 1: Exact Prices
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    ax1.plot(df['timestamp'], df['da_lmp'], label='Day-Ahead (DA) LMP', color='blue', linestyle='--', alpha=0.7)
    ax1.plot(df['timestamp'], df['rt_lmp'], label='Real-Time (RT) LMP', color='red', linewidth=2)
    ax1.set_title("PJM Pricing: The immense volatility of Real-Time vs Day-Ahead")
    ax1.set_ylabel("Price ($/MWh)")
    ax1.legend()
    
    # Subplot 2: DA/RT Spread
    # Green zones = DEC bids are profitable (RT > DA)
    # Red zones = INC bids are profitable (DA > RT)
    ax2.fill_between(df['timestamp'], df['rt_da_spread'], 0, 
                     where=(df['rt_da_spread'] > 0), color='green', alpha=0.5, label='DEC Profitable (RT > DA)')
    ax2.fill_between(df['timestamp'], df['rt_da_spread'], 0, 
                     where=(df['rt_da_spread'] < 0), color='red', alpha=0.5, label='INC Profitable (DA < RT)')
    ax2.set_title("The DA-RT Spread (The Algorithmic Trader's Alpha)")
    ax2.set_ylabel("Spread ($)")
    ax2.axhline(0, color='black', lw=1)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("phase_1_output.png")
    print("[+] Plot saved as 'phase_1_output.png'. You can view it in your Kaggle output directory.")

if __name__ == "__main__":
    main()
