import numpy as np

def metropolis_hastings(n_samples=2000, sigma=0.5):
    samples = []
    x = np.random.randn()
    for _ in range(n_samples):
        x_new = np.random.normal(x, sigma)
        log_alpha = log_pi(x_new) - log_pi(x)
        if np.log(np.random.rand()) < log_alpha:
            x = x_new
        samples.append(x)
    return np.array(samples)

# Target: standard normal
def log_pi(x):
    return -0.5 * x**2

def grad_log_pi(x):
    return -x

def hmc(n_samples=2000, eps=0.1, L=10):
    samples = []
    x = np.random.randn()

    for _ in range(n_samples):
        p = np.random.randn()
        x_new = x
        p_new = p

        # Leapfrog
        p_new += 0.5 * eps * grad_log_pi(x_new)
        for _ in range(L):
            x_new += eps * p_new
            if _ != L - 1:
                p_new += eps * grad_log_pi(x_new)
        p_new += 0.5 * eps * grad_log_pi(x_new)

        # Metropolis correction
        current_H = -log_pi(x) + 0.5 * p**2
        proposed_H = -log_pi(x_new) + 0.5 * p_new**2

        if np.random.rand() < np.exp(current_H - proposed_H):
            x = x_new

        samples.append(x)

    return np.array(samples)

import matplotlib.pyplot as plt

# Generate samples
mh_samples = metropolis_hastings(n_samples=5000, sigma=0.5)
hmc_samples = hmc(n_samples=5000, eps=0.1, L=10)

# Plot traces
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(mh_samples[:1000], label="MH", alpha=0.6)
plt.plot(hmc_samples[:1000], label="HMC", alpha=0.6)
plt.title("Trace Plot (First 1000 Samples)")
plt.legend()

# Plot histograms
plt.subplot(1, 2, 2)
plt.hist(mh_samples, bins=30, density=True, alpha=0.5, label="MH")
plt.hist(hmc_samples, bins=30, density=True, alpha=0.5, label="HMC")
plt.title("Histogram of Samples")
plt.legend()
plt.show()
