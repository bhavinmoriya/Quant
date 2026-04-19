import numpy as np

def mc_option(S0, K, r, sigma, T, n=100_000, option_type='call'):
    Z = np.random.normal(size=n)
    ST = S0 * np.exp(
        (r - 0.5 * sigma**2) * T +
        sigma * np.sqrt(T) * Z
    )
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:  # put
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r*T) * payoff.mean()
