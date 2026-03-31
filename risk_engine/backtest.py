# backtest.py
def var_violations(returns, var_series):
    aligned = returns.align(var_series, join="inner")[0]
    violations = aligned < var_series
    rate = violations.mean()
    return rate, violations
