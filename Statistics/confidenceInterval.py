import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

def get_confidence_interval_z_scores(confidence_level):
    """
    Calculates the critical Z-scores for a given confidence level.

    Args:
        confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).

    Returns:
        tuple: A tuple containing the lower and upper Z-scores.
    """
    alpha = 1 - confidence_level
    lower_tail_prob = alpha / 2
    upper_tail_prob = 1 - (alpha / 2)

    z_lower = norm.ppf(lower_tail_prob)
    z_upper = norm.ppf(upper_tail_prob)

    return z_lower, z_upper

# Example usage for a 95% confidence interval
z_lower_95, z_upper_95 = get_confidence_interval_z_scores(0.95)
print(f"For 95% CI: Lower Z-score = {z_lower_95:.3f}, Upper Z-score = {z_upper_95:.3f}")

# Example usage for a 99% confidence interval
z_lower_99, z_upper_99 = get_confidence_interval_z_scores(0.99)
print(f"For 99% CI: Lower Z-score = {z_lower_99:.3f}, Upper Z-score = {z_upper_99:.3f}")import numpy as np
from scipy.stats import norm

def get_confidence_interval_z_scores(confidence_level):
    """
    Calculates the critical Z-scores for a given confidence level.

    Args:
        confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).

    Returns:
        tuple: A tuple containing the lower and upper Z-scores.
    """
    alpha = 1 - confidence_level
    lower_tail_prob = alpha / 2
    upper_tail_prob = 1 - (alpha / 2)

    z_lower = norm.ppf(lower_tail_prob)
    z_upper = norm.ppf(upper_tail_prob)

    return z_lower, z_upper

# Example usage for a 95% confidence interval
z_lower_95, z_upper_95 = get_confidence_interval_z_scores(0.95)
print(f"For 95% CI: Lower Z-score = {z_lower_95:.3f}, Upper Z-score = {z_upper_95:.3f}")

# Example usage for a 99% confidence interval
z_lower_99, z_upper_99 = get_confidence_interval_z_scores(0.99)
print(f"For 99% CI: Lower Z-score = {z_lower_99:.3f}, Upper Z-score = {z_upper_99:.3f}")

# Population parameters
population_mean = 170  # True average height (cm)
population_std = 10    # True standard deviation
n_samples = 100        # Number of samples per experiment
n_experiments = 50     # Number of experiments (to show variability)

# Generate sample means and confidence intervals
sample_means = []
confidence_intervals = []
for _ in range(n_experiments):
    sample = np.random.normal(population_mean, population_std, n_samples)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)  # Sample standard deviation
    margin_of_error = 1.96 * (sample_std / np.sqrt(n_samples))  # 95% CI
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    sample_means.append(sample_mean)
    confidence_intervals.append((ci_lower, ci_upper))

# Plot the results
plt.figure(figsize=(12, 6))

# Plot each confidence interval
for i, (mean, (lower, upper)) in enumerate(zip(sample_means, confidence_intervals)):
    plt.plot([i + 1, i + 1], [lower, upper], color='blue', linewidth=2)
    plt.scatter(i + 1, mean, color='red', s=50, zorder=3)

# Highlight the true population mean
plt.axhline(y=population_mean, color='green', linestyle='--', label='True Population Mean')

# Annotate intervals that do not contain the true mean
for i, (lower, upper) in enumerate(confidence_intervals):
    if lower > population_mean or upper < population_mean:
        plt.scatter(i + 1, lower, color='orange', s=100, marker='x', zorder=4)

# Labels and title
plt.title('95% Confidence Intervals for Sample Means (n=100)', fontsize=14)
plt.xlabel('Experiment', fontsize=12)
plt.ylabel('Height (cm)', fontsize=12)
plt.ylim(160, 180)
plt.xlim(0, n_experiments + 1)
plt.xticks(np.arange(1, n_experiments + 1, 5))
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
