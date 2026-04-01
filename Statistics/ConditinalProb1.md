Let's visualize the **Bayesian update** and **Monte Carlo simulation** results. I'll use `matplotlib` and `seaborn` for clear, informative plots.

---

## **1. Bayesian Update Visualization**
We'll plot the **prior** and **posterior** distributions of the market crash probability.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

# Prior: Beta(2, 8)
prior_alpha, prior_beta = 2, 8
x = np.linspace(0, 1, 1000)
prior = beta.pdf(x, prior_alpha, prior_beta)

# Posterior: Beta(3, 17) after observing 1 crash in 10 trials
posterior_alpha, posterior_beta = 3, 17
posterior = beta.pdf(x, posterior_alpha, posterior_beta)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, prior, label=f'Prior (α={prior_alpha}, β={prior_beta})', color='blue')
plt.plot(x, posterior, label=f'Posterior (α={posterior_alpha}, β={posterior_beta})', color='red')
plt.title('Bayesian Update: Market Crash Probability')
plt.xlabel('Probability of Market Crash')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
```

**Output**:
A plot showing how the probability distribution updates after observing new data.

---

## **2. Monte Carlo Simulation Visualization**
We'll visualize the relationship between **market drops** and **liquidity crises** using a scatter plot and a bar plot for conditional probabilities.

```python
import seaborn as sns

np.random.seed(42)

# Simulate 10,000 scenarios
market_drop = np.random.binomial(1, 0.2, 10000)
liquidity_crisis = np.random.binomial(1, 0.4 * market_drop + 0.05 * (1 - market_drop), 10000)

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(10000), y=liquidity_crisis, hue=market_drop, palette={0: 'blue', 1: 'red'})
plt.title('Monte Carlo Simulation: Market Drop vs. Liquidity Crisis')
plt.xlabel('Simulation Trial')
plt.ylabel('Liquidity Crisis (1 = Yes, 0 = No)')
plt.yticks([0, 1], ['No Crisis', 'Crisis'])
plt.legend(title='Market Drop', labels=['No', 'Yes'])
plt.show()

# Bar plot: P(Liquidity Crisis | Market Drop)
p_crisis_given_drop = liquidity_crisis[market_drop == 1].mean()
p_crisis_given_no_drop = liquidity_crisis[market_drop == 0].mean()

plt.figure(figsize=(6, 4))
sns.barplot(x=['Market Drop', 'No Market Drop'], y=[p_crisis_given_drop, p_crisis_given_no_drop])
plt.title('Conditional Probability of Liquidity Crisis')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.show()
```

**Output**:
1. A scatter plot showing trials where a market drop led to a liquidity crisis.
2. A bar plot comparing the probability of a liquidity crisis with and without a market drop.

---

### **Key Insights from Visualizations**
- **Bayesian Update**: Shows how your belief about the market crash probability changes with new data.
- **Monte Carlo Simulation**: Highlights the increased risk of a liquidity crisis when a market drop occurs.

---

