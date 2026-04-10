import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import time

# --- Parameters for Real Data Analysis ---
start_date = "2020-01-01" # MicroStrategy started accumulating BTC in mid-2020
end_date = pd.to_datetime(pd.Timestamp.now().date()) # Today's date

# MicroStrategy's estimated BTC holdings and debt (simplified fixed values for example)
# For a precise historical analysis, these would ideally be historical values.
btc_holdings_fixed = 200_000 # Example: As of a certain date
debt_fixed = 5e9 # Example: Total debt
shares_fixed = 15e6 # Example: Initial shares outstanding

# --- Load Real Data with Retry Mechanism ---
tickers = ["BTC-USD", "MSTR"]
max_retries = 3
retry_delay = 5  # seconds

data = pd.DataFrame() # Initialize empty DataFrame
for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}/{max_retries} to download data...")
        data = yf.download(tickers, start=start_date, end=end_date)["Close"]
        if not data.empty:
            print("Data downloaded successfully.")
            break # Success, exit loop
        else:
            raise ValueError("Downloaded data is empty.")
    except Exception as e:
        print(f"Download attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Could not download data.")
            # If all attempts fail, `data` will remain empty, which will be handled below.

# Ensure data is aligned and drop NaNs
data = data.dropna()

# Check if data is still empty after retries and drops
if data.empty:
    print("Warning: No valid data available after download and cleanup. Calculations will be affected.")
    # Initialize empty Series to prevent errors in subsequent operations
    btc_prices = pd.Series([], dtype='float64')
    mstr_prices = pd.Series([], dtype='float64')
else:
    # Extract BTC and MSTR prices
    btc_prices = data["BTC-USD"]
    mstr_prices = data["MSTR"]

# --- Calculate NAV and mNAV ---
# NAV = (BTC Price * BTC Holdings - Debt) / Shares
# Only attempt if btc_prices is not empty
if not btc_prices.empty:
    nav_series = (btc_prices * btc_holdings_fixed - debt_fixed) / shares_fixed

    # Filter out days where NAV might be non-positive (e.g., if debt exceeds BTC value)
    nav_series = nav_series[nav_series > 0]

    # Align MSTR prices with calculated NAV
    mstr_prices_aligned = mstr_prices[mstr_prices.index.isin(nav_series.index)]
    nav_series_aligned = nav_series[nav_series.index.isin(mstr_prices_aligned.index)]

    # Calculate mNAV (MSTR Premium/Discount to NAV)
    mnav_series = mstr_prices_aligned / nav_series_aligned
else:
    nav_series = pd.Series([], dtype='float64')
    mstr_prices_aligned = pd.Series([], dtype='float64')
    nav_series_aligned = pd.Series([], dtype='float64')
    mnav_series = pd.Series([], dtype='float64')

# --- Plotting Real Data ---
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1) # Top subplot for prices
if not btc_prices.empty:
    plt.plot(btc_prices.index, btc_prices.values, label="BTC-USD (Real)", color='orange')
if not mstr_prices_aligned.empty:
    plt.plot(mstr_prices_aligned.index, mstr_prices_aligned.values, label="MSTR (Real)", color='blue')
if not nav_series_aligned.empty:
    plt.plot(nav_series_aligned.index, nav_series_aligned.values, label="NAV (Derived)", color='green', linestyle='--')
plt.title("Real BTC, MSTR, and Derived NAV Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2) # Bottom subplot for mNAV
if not mnav_series.empty:
    plt.plot(mnav_series.index, mnav_series.values, label="mNAV (MSTR Price / NAV)", color='purple')
plt.axhline(y=1.0, color='red', linestyle='--', label='Parity (mNAV = 1)')
plt.title("Real MSTR Premium/Discount to NAV (mNAV)")
plt.xlabel("Date")
plt.ylabel("mNAV Ratio")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Analysis of Returns and Risk ---
btc_returns = btc_prices.pct_change().dropna()
mstr_returns = mstr_prices.pct_change().dropna()

# Annualized Sharpe Ratio = (Mean Daily Return / Std Dev Daily Return) * sqrt(252 trading days)
btc_sharpe = btc_returns.mean() / btc_returns.std() * np.sqrt(252) if not btc_returns.empty and btc_returns.std() != 0 else np.nan
mstr_sharpe = mstr_returns.mean() / mstr_returns.std() * np.sqrt(252) if not mstr_returns.empty and mstr_returns.std() != 0 else np.nan
print("BTC Sharpe (Annualized):".ljust(25), f"{btc_sharpe:.4f}")
print("MSTR Sharpe (Annualized):".ljust(25), f"{mstr_sharpe:.4f}")

# Max Drawdown = Minimum value of (current_price / peak_price - 1)
btc_max_drawdown = (btc_prices / btc_prices.expanding().max() - 1).min() if not btc_prices.empty else np.nan
mstr_max_drawdown = (mstr_prices / mstr_prices.expanding().max() - 1).min() if not mstr_prices.empty else np.nan
print("BTC Max Drawdown:".ljust(25), f"{btc_max_drawdown:.4f}")
print("MSTR Max Drawdown:".ljust(25), f"{mstr_max_drawdown:.4f}")
