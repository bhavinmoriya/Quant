import numpy as np
from scipy.stats import norm


def d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)


def call_price(S, K, r, sigma, T):
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)


def delta(S, K, r, sigma, T):
    return norm.cdf(d1(S, K, r, sigma, T))


def gamma(S, K, r, sigma, T):
    return norm.pdf(d1(S, K, r, sigma, T)) / (S * sigma * np.sqrt(T))


def vega(S, K, r, sigma, T):
    return S * norm.pdf(d1(S, K, r, sigma, T)) * np.sqrt(T)
