Here’s a **Python simulation** demonstrating the **Central Limit Theorem (CLT)** in action. It shows how the **sampling distribution of the mean** becomes approximately normal as the sample size increases, even if the original population is **non-normal**. We’ll compare sample sizes **less than 30** and **greater than or equal to 30**.

---

### **📌 Simulation Setup**
1. **Population**: We’ll use a **highly non-normal distribution** (exponential distribution, which is right-skewed).
2. **Sample Sizes**: We’ll test **n = 10 (small)** and **n = 50 (large)**.
3. **Metrics**: We’ll plot the **sampling distribution of the mean** for both sample sizes and compare it to a normal distribution.

---

### **🐍 Python Code for the Simulation**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
population_size = 10000  # Size of the population
num_samples = 10000      # Number of samples to draw
sample_sizes = [10, 50]  # Small (n < 30) and large (n >= 30) sample sizes

# Generate a non-normal population (exponential distribution)
population = expon.rvs(scale=2, size=population_size)

# Plot the population distribution
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(population, bins=50, density=True, alpha=0.7, color='blue')
plt.title('Population Distribution (Exponential)')
plt.xlabel('Value')
plt.ylabel('Density')

# Simulate sampling distributions for n=10 and n=50
for i, n in enumerate(sample_sizes):
    # Draw samples and calculate their means
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=n, replace=True)
        sample_means.append(np.mean(sample))

    # Plot the sampling distribution of the mean
    plt.subplot(1, 3, i + 2)
    plt.hist(sample_means, bins=50, density=True, alpha=0.7, color='green', label=f'Sample Means (n={n})')

    # Overlay the normal distribution with the same mean and std
    mu, std = np.mean(sample_means), np.std(sample_means)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
    plt.title(f'Sampling Distribution (n={n})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()
```

---

### **📊 What the Simulation Shows**
1. **Population Distribution (Exponential)**:
   - The population is **highly right-skewed** (not normal at all).

2. **Sampling Distribution for n = 10 (Small Sample)**:
   - The distribution of sample means is **still skewed** and **not normal**.
   - The **Central Limit Theorem (CLT) hasn’t fully kicked in** yet.

3. **Sampling Distribution for n = 50 (Large Sample)**:
   - The distribution of sample means is **approximately normal** (bell-shaped).
   - The **CLT is working**: Even though the population is non-normal, the sampling distribution of the mean is **close to normal** for **n ≥ 30**.

---

### **💡 Key Observations**
- For **n < 30**, the sampling distribution of the mean **retains the shape of the population distribution** (skewed in this case).
- For **n ≥ 30**, the sampling distribution of the mean **becomes approximately normal**, regardless of the population distribution.
- This is the **power of the Central Limit Theorem**: It allows us to use **normal-based statistical methods** (e.g., t-tests, confidence intervals) even for non-normal populations, **as long as the sample size is large enough (n ≥ 30)**.

---
Would you like me to:
1. **Run the simulation** and show you the actual plots?
2. **Explain the mathematical reasoning** behind why n ≥ 30 works for CLT?
3. **Test other non-normal distributions** (e.g., uniform, bimodal) to see how CLT behaves?
