import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np

def simulate_advanced_confluence(ticker, start_date, end_date):
    # 1. Download Data
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        print("No data found.")
        return

    # Check if columns are MultiIndex and flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        # Assuming the MultiIndex is (Attribute, Ticker), we want Attribute
        df.columns = df.columns.droplevel(1)

    # 2. Calculate Indicators using pandas_ta
    # Bollinger Bands
    bbands = df.ta.bbands(length=20, std=2)
    df = pd.concat([df, bbands], axis=1)

    # RSI
    df['RSI'] = df.ta.rsi(length=14)

    # Volume Average
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()

    # 3. Manual Candlestick Pattern Detection (Logic)
    # Hammer: Small body, long lower wick, little upper wick
    body = abs(df['Close'] - df['Open'])
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)

    df['Hammer'] = (lower_wick > (body * 2)) & (upper_wick < body)

    # Shooting Star: Small body, long upper wick, little lower wick
    df['Shooting_Star'] = (upper_wick > (body * 2)) & (lower_wick < body)

    # 4. THE CONFLUENCE STRATEGY (The "Decision Engine")
    df['Signal'] = 0  # 0 = No Signal, 1 = Strong Buy, -1 = Strong Sell

    # --- STRONG BUY CONDITION ---
    # Price near Lower Band + RSI < 35 + Hammer Pattern + High Volume
    buy_condition = (
        (df['Close'] <= df['BBL_20_2.0_2.0']) &
        (df['RSI'] < 35) &
        (df['Hammer'] == True) &
        (df['Volume'] > df['Vol_Avg'])
    )
    df.loc[buy_condition, 'Signal'] = 1

    # --- STRONG SELL CONDITION ---
    # Price near Upper Band + RSI > 65 + Shooting Star + High Volume
    sell_condition = (
        (df['Close'] >= df['BBU_20_2.0_2.0']) &
        (df['RSI'] > 65) &
        (df['Shooting_Star'] == True) &
        (df['Volume'] > df['Vol_Avg'])
    )
    df.loc[sell_condition, 'Signal'] = -1

    # 5. Advanced Visualization (Multi-Pane Chart)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1, 1]})

    # Top Plot: Price & Bollinger Bands
    ax1.plot(df['Close'], label='Price', color='black', alpha=0.7)
    ax1.plot(df['BBU_20_2.0_2.0'], label='Upper Band', color='red', alpha=0.3)
    ax1.plot(df['BBL_20_2.0_2.0'], label='Lower Band', color='green', alpha=0.3)
    ax1.fill_between(df.index, df['BBL_20_2.0_2.0'], df['BBU_20_2.0_2.0'], color='gray', alpha=0.1)

    # Plot Signals on Price Chart
    ax1.scatter(df.index[df['Signal'] == 1], df['Close'][df['Signal'] == 1],
                marker='^', color='green', s=200, label='STRONG BUY (Confluence)')
    ax1.scatter(df.index[df['Signal'] == -1], df['Close'][df['Signal'] == -1],
                marker='v', color='red', s=200, label='STRONG SELL (Confluence)')
    ax1.set_title(f"Advanced Confluence Strategy: {ticker}", fontsize=16)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Middle Plot: RSI
    ax2.plot(df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5) # Overbought line
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5) # Oversold line
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # Bottom Plot: Volume
    colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
    ax3.bar(df.index, df['Volume'], color=colors, alpha=0.5)
    ax3.plot(df['Vol_Avg'], color='blue', label='Vol Avg')
    ax3.set_ylabel('Volume')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return df

# The function call will be moved to a new cell to allow the backtest to run after the df is returned
simulate_advanced_confluence('TSLA', '2021-01-01', '2026-01-01')
