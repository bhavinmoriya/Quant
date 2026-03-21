"""
Markowitz Mean-Variance Portfolio Optimization
================================================
Implements:
  - Efficient Frontier via Monte Carlo simulation
  - Analytical efficient frontier (via quadratic programming with scipy)
  - Minimum Variance Portfolio
  - Maximum Sharpe Ratio Portfolio
  - Efficient Frontier plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── 1. Synthetic asset returns (replace with real data via yfinance, etc.) ───
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
N = len(TICKERS)
TRADING_DAYS = 252

# Simulate daily log-returns ~ N(mu, Sigma)
mu_daily = np.array([0.0008, 0.0007, 0.0006, 0.0007, 0.0005])
# Random but positive-definite covariance matrix
A = np.random.randn(N, N) * 0.01
cov_daily = A @ A.T + np.diag([0.0002] * N)

# Annualise
mu_annual = mu_daily * TRADING_DAYS
cov_annual = cov_daily * TRADING_DAYS


# ── 2. Portfolio statistics ───────────────────────────────────────────────────
def portfolio_stats(weights: np.ndarray,
                    mu: np.ndarray,
                    cov: np.ndarray,
                    rf: float = 0.04) -> tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe_ratio)."""
    ret = weights @ mu
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe


# ── 3. Monte Carlo simulation of random portfolios ───────────────────────────
def simulate_portfolios(n_portfolios: int = 20_000,
                        mu: np.ndarray = mu_annual,
                        cov: np.ndarray = cov_annual,
                        rf: float = 0.04) -> pd.DataFrame:
    results = np.zeros((n_portfolios, 3 + N))
    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(N))          # uniform on simplex
        ret, vol, sr = portfolio_stats(w, mu, cov, rf)
        results[i, :3] = [ret, vol, sr]
        results[i, 3:] = w

    cols = ["Return", "Volatility", "Sharpe"] + [f"w_{t}" for t in TICKERS]
    return pd.DataFrame(results, columns=cols)


# ── 4. Analytical optimisation (long-only, fully invested) ───────────────────
def _neg_sharpe(w, mu, cov, rf):
    return -portfolio_stats(w, mu, cov, rf)[2]

def _portfolio_vol(w, mu, cov, rf):
    return portfolio_stats(w, mu, cov, rf)[1]

def _base_constraints_bounds(n):
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n          # long-only; change to (None,None) for unconstrained
    w0 = np.ones(n) / n
    return constraints, bounds, w0

def max_sharpe_portfolio(mu=mu_annual, cov=cov_annual, rf=0.04) -> np.ndarray:
    constraints, bounds, w0 = _base_constraints_bounds(len(mu))
    res = minimize(_neg_sharpe, w0, args=(mu, cov, rf),
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x

def min_variance_portfolio(mu=mu_annual, cov=cov_annual, rf=0.04) -> np.ndarray:
    constraints, bounds, w0 = _base_constraints_bounds(len(mu))
    res = minimize(_portfolio_vol, w0, args=(mu, cov, rf),
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x

def efficient_frontier_points(n_points: int = 200,
                               mu=mu_annual, cov=cov_annual, rf=0.04):
    """Trace the efficient frontier by minimising vol for each target return."""
    constraints_base, bounds, w0 = _base_constraints_bounds(len(mu))

    # Return range: from min-variance return to max possible return
    w_mv = min_variance_portfolio(mu, cov, rf)
    r_min = w_mv @ mu
    r_max = mu.max()

    vols, rets = [], []
    for target in np.linspace(r_min, r_max, n_points):
        con = constraints_base + [{
            "type": "eq",
            "fun": lambda w, t=target: w @ mu - t
        }]
        res = minimize(_portfolio_vol, w0, args=(mu, cov, rf),
                       method="SLSQP", bounds=bounds, constraints=con,
                       options={"ftol": 1e-12, "maxiter": 1000})
        if res.success:
            vols.append(res.fun)
            rets.append(target)

    return np.array(vols), np.array(rets)


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main():
    RF = 0.04  # risk-free rate

    print("Simulating random portfolios …")
    sim = simulate_portfolios(rf=RF)

    print("Computing key portfolios …")
    w_sharpe = max_sharpe_portfolio(rf=RF)
    w_minvar = min_variance_portfolio(rf=RF)

    r_sh, v_sh, sr_sh = portfolio_stats(w_sharpe, mu_annual, cov_annual, RF)
    r_mv, v_mv, sr_mv = portfolio_stats(w_minvar, mu_annual, cov_annual, RF)

    print("\n── Maximum Sharpe Portfolio ──────────────────────────")
    for t, w in zip(TICKERS, w_sharpe):
        print(f"  {t:6s}: {w:.4f}")
    print(f"  Return:     {r_sh:.4f}  ({r_sh*100:.2f}%)")
    print(f"  Volatility: {v_sh:.4f}  ({v_sh*100:.2f}%)")
    print(f"  Sharpe:     {sr_sh:.4f}")

    print("\n── Minimum Variance Portfolio ────────────────────────")
    for t, w in zip(TICKERS, w_minvar):
        print(f"  {t:6s}: {w:.4f}")
    print(f"  Return:     {r_mv:.4f}  ({r_mv*100:.2f}%)")
    print(f"  Volatility: {v_mv:.4f}  ({v_mv*100:.2f}%)")
    print(f"  Sharpe:     {sr_mv:.4f}")

    print("\nTracing analytical efficient frontier …")
    ef_vols, ef_rets = efficient_frontier_points(rf=RF)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(sim["Volatility"], sim["Return"],
                    c=sim["Sharpe"], cmap="viridis",
                    alpha=0.4, s=5, label="Random portfolios")
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    ax.plot(ef_vols, ef_rets, "w--", lw=2, label="Efficient Frontier")

    ax.scatter(v_sh, r_sh, marker="*", color="gold", s=300, zorder=5,
               label=f"Max Sharpe  (SR={sr_sh:.2f})")
    ax.scatter(v_mv, r_mv, marker="D", color="red", s=100, zorder=5,
               label=f"Min Variance (SR={sr_mv:.2f})")

    # Capital Market Line
    v_range = np.linspace(0, sim["Volatility"].max(), 200)
    cml = RF + sr_sh * v_range
    ax.plot(v_range, cml, "w:", lw=1.5, label="Capital Market Line")

    ax.set_xlabel("Annualised Volatility (σ)")
    ax.set_ylabel("Annualised Expected Return (μ)")
    ax.set_title("Markowitz Mean-Variance Efficient Frontier")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150)
    plt.show()
    print("\nPlot saved to efficient_frontier.png")


if __name__ == "__main__":
    main()
