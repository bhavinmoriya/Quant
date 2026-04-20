"""
Task:
1. Simulate GBM under:
   a) real-world drift μ
   b) risk-neutral drift r

2. Show:
   E_real[S_T] != E_risk_neutral[S_T]

3. Explain:
   Why pricing uses risk-neutral expectation
"""

# TODO: implement
import numpy as np
import polars as pl

def simulate_gbm(S0, drift, sigma, T, n=200_000):
    Z = np.random.normal(size=n)

    ST = S0 * np.exp(
        (drift - 0.5 * sigma**2) * T +
        sigma * np.sqrt(T) * Z
    )
    return ST


# Parameters
S0 = 100
mu = 0.10     # real-world drift
r = 0.03      # risk-free rate
sigma = 0.2
T = 1

# Simulations
ST_real = simulate_gbm(S0, mu, sigma, T)
ST_rn   = simulate_gbm(S0, r, sigma, T)

# Put into Polars for clean stats
df = pl.DataFrame({
    "real": ST_real,
    "risk_neutral": ST_rn
})

summary = df.select([
    pl.mean("real").alias("E_real"),
    pl.mean("risk_neutral").alias("E_risk_neutral"),
])

print(summary)
