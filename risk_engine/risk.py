# risk.py
import numpy as np

def var(x, alpha=0.95):
    return np.quantile(x, 1 - alpha)

def cvar(x, alpha=0.95):
    v = var(x, alpha)
    tail = x[x <= v]
    return tail.mean()

def rolling_risk(series, window=252, alpha=0.95):
    var_series = series.rolling(window).apply(lambda x: var(x, alpha))
    cvar_series = series.rolling(window).apply(lambda x: cvar(x, alpha))
    return var_series, cvar_series
