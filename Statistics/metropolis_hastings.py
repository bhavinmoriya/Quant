import numpy as np
import matplotlib.pyplot as plt

# target distribution (unnormalized)
def pi(x):
    return np.exp(-x**4 + x**2)

def metropolis_hastings(n_samples=10000, proposal_std=1.0):
    samples = []
    x = 0  # start point
    
    for _ in range(n_samples):
        x_proposed = x + np.random.normal(0, proposal_std)
        
        acceptance_ratio = pi(x_proposed) / pi(x)
        alpha = min(1, acceptance_ratio)
        
        if np.random.rand() < alpha:
            x = x_proposed
        
        samples.append(x)
    
    return np.array(samples)

samples = metropolis_hastings()

# plot
x = np.linspace(-2, 2, 500)
plt.hist(samples, bins=100, density=True, alpha=0.5)
plt.plot(x, pi(x)/np.trapz(pi(x), x), 'r')
plt.title("MCMC Sampling")
plt.show()
