# main.py
import numpy as np
import matplotlib.pyplot as plt

from data import load_data
from risk import rolling_risk
from backtest import var_violations
from strategy import cvar_signal

tickers = ["SPY", "TLT", "GLD"]
data, returns = load_data(tickers)

spy = returns["SPY"]

# --- Risk estimation ---
var_series, cvar_series = rolling_risk(spy)

# --- VaR validation ---
rate, violations = var_violations(spy, var_series)
print("VaR violation rate:", rate)

# --- Strategy ---
signals = cvar_signal(spy.values)

aligned_returns = spy.values[-len(signals):]
strategy_returns = aligned_returns * signals

# --- Equity curves ---
cum_market = (1 + aligned_returns).cumprod()
cum_strategy = (1 + strategy_returns).cumprod()

plt.plot(cum_market, label="Market")
plt.plot(cum_strategy, label="CVaR Strategy")
plt.legend()
plt.show()
