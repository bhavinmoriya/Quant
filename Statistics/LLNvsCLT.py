import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Define a non-normal distribution (e.g., exponential)
population = np.random.exponential(scale=1.0, size=100000)

# Sample sizes to illustrate CLT
sample_sizes = [2, 10, 30, 100]

# Plot the population distribution and sampling distributions
plt.figure(figsize=(15, 10))

# Plot the population distribution
plt.subplot(2, 2, 1)
sns.histplot(population, bins=50, kde=True, color='skyblue')
plt.title('Population Distribution\n(Exponential)')

# Plot sampling distributions for different sample sizes
for i, n in enumerate(sample_sizes, start=2):
    # Generate sample means
    sample_means = [np.mean(np.random.choice(population, n, replace=True)) for _ in range(1000)]
    
    plt.subplot(2, 2, i-1)
    sns.histplot(sample_means, bins=30, kde=True, color='salmon')
    plt.title(f'Sample Means (n={n})')

plt.tight_layout()
plt.show()
