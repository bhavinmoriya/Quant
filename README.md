[**DISCLAIMER**](DISCLAIMER.md)

---

# Algorithmic Trading Training: Energy Sector (PJM)

This repository serves as a progressive, hands-on curriculum for building algorithmic trading strategies specifically designed for wholesale electricity markets (like PJM, ERCOT, CAISO).

The code is optimized for Kaggle/Google Colab environments (8GB RAM limits) by utilizing `polars` instead of `pandas` and vectorizing heavy calculations.

---

## 📈 Curriculum Overview

### [Phase 1: Market Fundamentals (`phase_1.py`)](phase_1.py)
*   **Concepts:** Day-Ahead (DA) vs. Real-Time (RT) markets, Locational Marginal Pricing (LMP).
*   **The Strategy:** Virtual Bidding (INC and DEC Bids) based on DA/RT price spreads.
*   **Code:** Fetching live data from the PJM API using `requests` and processing time-series data efficiently with `polars`.

### [Phase 2: Data Engineering (`phase_2.py`)](phase_2.py)
*   **Concepts:** Grid physics—why RT prices explode exponentially (The "Hockey Stick" curve).
*   **The Strategy:** Engineering non-linear features like `extreme_heat_degrees` to capture grid stress. 
*   **Code:** Using `polars.join` to merge disparate time-series data (Hourly Pricing vs. Weather/Load telemetry).

### Phase 3: Machine Learning (Next Up!)
*   **Concepts:** Formulating the trading problem as a Regression/Classification task.
*   **The Strategy:** Using your previous **XGBoost** and **LSTM** knowledge to forecast the `rt_da_spread`.
*   **Expected Output:** Generating actual -1 (Short), 0 (Flat), and +1 (Long) trading signals.

### [Phase 4: Backtesting & Risk Management (`phase_4.py`)](phase_4.py)
*   **Concepts:** Slippage, transaction fees, and quantitative performance tearsheets.
*   **The Strategy:** Building a vectorized historical simulation engine in Polars to evaluate the ML model's real-world viability.
*   **Expected Output:** A visualization of the Equity Curve and Drawdown depths, alongside metrics like the Sharpe Ratio.

---

## 🛠 Setup for Kaggle/Colab

To run these scripts in your Kaggle environment, simply install the high-performance dependencies:

```bash
!pip install polars requests matplotlib seaborn seaborn
```

Then execute the phase you are working on:

```bash
!python phase_1.py
!python phase_2.py
```
