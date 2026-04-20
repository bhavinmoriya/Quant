import numpy as np

def var_calc(x, alpha):
    return np.quantile(x, 1 - alpha)

def cvar_calc(x, alpha):
    v = var_calc(x, alpha)
    return x[x <= v].mean()
