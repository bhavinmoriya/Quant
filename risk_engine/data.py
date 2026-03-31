# data.py
import yfinance as yf
import pandas as pd

def load_data(tickers, start="2005-01-01"):
    data = yf.download(tickers, start=start)["Close"]
    returns = data.pct_change().dropna()
    return data, returns
