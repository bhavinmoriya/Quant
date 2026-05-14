import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
from scipy.stats import pareto

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
population_size = 10000  # Size of the population
num_samples = 10000      # Number of samples to draw
sample_sizes = [10, 50]  # Small (n < 30) and large (n >= 30) sample sizes

# Generate a non-normal population (exponential distribution)
population = expon.rvs(scale=2, size=population_size)
population = pareto.rvs(b=2.65, size=population_size)

# Plot the population distribution
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(population, bins=50, density=True, alpha=0.7, color='blue')
# plt.title('Population Distribution (Exponential)')
plt.title('Population Distribution (Pareto)')
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
