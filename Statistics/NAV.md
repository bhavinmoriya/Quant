### **Refined Code for Backtesting BTC and MSTR with Real Data**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import time

# --- Parameters for Real Data Analysis ---
start_date = "2020-01-01"  # MicroStrategy started accumulating BTC in mid-2020
end_date = pd.to_datetime(pd.Timestamp.now().date())  # Today's date

# MicroStrategy's estimated BTC holdings and debt (simplified fixed values for example)
btc_holdings_fixed = 200_000  # Example: As of a certain date
debt_fixed = 5e9  # Example: Total debt
shares_fixed = 15e6  # Example: Initial shares outstanding

# --- Load Real Data with Retry Mechanism ---
tickers = ["BTC-USD", "MSTR"]
max_retries = 3
retry_delay = 5  # seconds

data = pd.DataFrame()  # Initialize empty DataFrame
for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}/{max_retries} to download data...")
        data = yf.download(tickers, start=start_date, end=end_date)["Close"]
        if not data.empty:
            print("Data downloaded successfully.")
            break  # Success, exit loop
        else:
            raise ValueError("Downloaded data is empty.")
    except Exception as e:
        print(f"Download attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Could not download data.")
            data = pd.DataFrame()  # Ensure `data` is empty if all retries fail

# --- Data Cleaning and Alignment ---
if not data.empty:
    data = data.dropna()
    btc_prices = data["BTC-USD"]
    mstr_prices = data["MSTR"]

    # Calculate NAV: (BTC Price * BTC Holdings - Debt) / Shares
    nav_series = (btc_prices * btc_holdings_fixed - debt_fixed) / shares_fixed
    nav_series = nav_series[nav_series > 0]  # Filter out non-positive NAV

    # Align MSTR prices with NAV
    mstr_prices_aligned = mstr_prices[mstr_prices.index.isin(nav_series.index)]
    nav_series_aligned = nav_series[nav_series.index.isin(mstr_prices_aligned.index)]

    # Calculate mNAV: MSTR Price / NAV
    mnav_series = mstr_prices_aligned / nav_series_aligned
else:
    print("No valid data available for analysis.")
    btc_prices, mstr_prices, nav_series, mnav_series = pd.Series(), pd.Series(), pd.Series(), pd.Series()

# --- Plotting ---
if not data.empty:
    plt.figure(figsize=(15, 10))

    # Top subplot: Prices
    plt.subplot(2, 1, 1)
    plt.plot(btc_prices.index, btc_prices, label="BTC-USD (Real)", color='orange')
    plt.plot(mstr_prices_aligned.index, mstr_prices_aligned, label="MSTR (Real)", color='blue')
    plt.plot(nav_series_aligned.index, nav_series_aligned, label="NAV (Derived)", color='green', linestyle='--')
    plt.title("Real BTC, MSTR, and Derived NAV Prices")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    # Bottom subplot: mNAV
    plt.subplot(2, 1, 2)
    plt.plot(mnav_series.index, mnav_series, label="mNAV (MSTR Price / NAV)", color='purple')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Parity (mNAV = 1)')
    plt.title("Real MSTR Premium/Discount to NAV (mNAV)")
    plt.xlabel("Date")
    plt.ylabel("mNAV Ratio")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Risk and Return Analysis ---
    btc_returns = btc_prices.pct_change().dropna()
    mstr_returns = mstr_prices.pct_change().dropna()

    # Annualized Sharpe Ratio
    btc_sharpe = (btc_returns.mean() / btc_returns.std()) * np.sqrt(252) if not btc_returns.empty else np.nan
    mstr_sharpe = (mstr_returns.mean() / mstr_returns.std()) * np.sqrt(252) if not mstr_returns.empty else np.nan

    # Max Drawdown
    btc_max_drawdown = (btc_prices / btc_prices.expanding().max() - 1).min() if not btc_prices.empty else np.nan
    mstr_max_drawdown = (mstr_prices / mstr_prices.expanding().max() - 1).min() if not mstr_prices.empty else np.nan

    print("BTC Sharpe (Annualized):".ljust(25), f"{btc_sharpe:.4f}")
    print("MSTR Sharpe (Annualized):".ljust(25), f"{mstr_sharpe:.4f}")
    print("BTC Max Drawdown:".ljust(25), f"{btc_max_drawdown:.4f}")
    print("MSTR Max Drawdown:".ljust(25), f"{mstr_max_drawdown:.4f}")
else:
    print("No data to plot or analyze.")
```

---

### **Key Improvements and Notes**

1. **Error Handling and Retries**:
   - The code now gracefully handles failed data downloads and ensures `data` is empty if all retries fail.

2. **Data Alignment**:
   - Ensures `btc_prices`, `mstr_prices`, and `nav_series` are aligned by date before calculating `mnav_series`.

3. **Visualization**:
   - Two subplots: one for price comparison (BTC, MSTR, NAV) and one for the `mNAV` ratio (MSTR price relative to NAV).

4. **Risk and Return Metrics**:
   - Annualized Sharpe ratios and max drawdowns are calculated for both BTC and MSTR.

5. **Edge Cases**:
   - Handles cases where data is empty or calculations are not possible (e.g., division by zero, empty series).

---

### **Expected Outputs**

- **Plots**:
  - Top: BTC, MSTR, and derived NAV prices over time.
  - Bottom: MSTR’s premium/discount to NAV (`mNAV`), with a parity line at `mNAV = 1`.

- **Printed Metrics**:
  - Annualized Sharpe ratios for BTC and MSTR.
  - Max drawdowns for both assets.

---

### **How to Use This Code**

1. **Install Dependencies**:
   ```bash
   pip install yfinance pandas numpy matplotlib
   ```

2. **Run the Script**:
   - Copy the code into a Python file or Jupyter Notebook and run it.

3. **Interpret Results**:
   - Compare the performance and risk of BTC and MSTR.
   - Analyze how MSTR’s price relates to its NAV over time.

---
