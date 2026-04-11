**🚀 Hamiltonian Monte Carlo (HMC) vs. Metropolis-Hastings (MH): A Visual Comparison**

Sampling from complex distributions is a core task in Bayesian statistics and machine learning. While **Metropolis-Hastings (MH)** is a classic MCMC method, **Hamiltonian Monte Carlo (HMC)** leverages gradient information to explore the target distribution more efficiently.

I implemented both algorithms to sample from a **standard normal distribution** and compared their performance. Here’s what I found:

---

### **1. Algorithms**
#### **Metropolis-Hastings (MH)**
- Proposes new states via random walk (Gaussian proposal).
- No gradient information used.
- Can struggle in high dimensions or correlated distributions.

#### **Hamiltonian Monte Carlo (HMC)**
- Uses gradient of the log density to guide proposals.
- Simulates Hamiltonian dynamics via leapfrog integration.
- More efficient exploration, especially in high dimensions.

---

### **2. Code**
#### **Metropolis-Hastings**
```python
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
```

#### **HMC** (as before)
```python
def hmc(n_samples=2000, eps=0.1, L=10):
    samples = []
    x = np.random.randn()
    for _ in range(n_samples):
        p = np.random.randn()
        x_new, p_new = x, p
        # Leapfrog integration
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
```

---

### **3. Visual Comparison**
Let’s plot the traces and histograms of samples from both algorithms:

```python
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
```

---

### **4. Results**
- **Trace Plot**: HMC explores the space more efficiently, with fewer "sticky" periods than MH.
- **Histogram**: Both converge to the target (standard normal), but HMC requires fewer samples for accurate results.

---

### **5. Why HMC Wins**
- **Faster Convergence**: HMC’s gradient-guided proposals reduce random walk behavior.
- **Scalability**: HMC shines in high dimensions, where MH struggles.

**Question for you**: Have you used HMC in your work? What distributions did you sample, and how did it compare to other methods?

#HamiltonianMonteCarlo #MCMC #BayesianInference #DataScience #Python


