# portfolio.py
import cvxpy as cp
import numpy as np

def optimize_cvar(returns, alpha=0.95):
    R = returns.values
    N, d = R.shape

    w = cp.Variable(d)
    eta = cp.Variable()
    u = cp.Variable(N)

    losses = -R @ w

    objective = eta + (1/(1-alpha)/N) * cp.sum(u)

    constraints = [
        u >= losses - eta,
        u >= 0,
        cp.sum(w) == 1
    ]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    return w.value
