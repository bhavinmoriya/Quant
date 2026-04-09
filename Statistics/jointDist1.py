import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parameters for the bivariate normal distribution
mu = [170, 70]  # Mean height (cm) and weight (kg)
cov = [[10, 20], [20, 100]]  # Covariance matrix (correlation: cov[0,1]/sqrt(cov[0,0]*cov[1,1]))

# Create a grid of (x, y) points
x = np.linspace(150, 190, 100)
y = np.linspace(50, 90, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Evaluate the joint pdf at each point
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)

# Plot the joint distribution
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
ax.set_xlabel('Height (cm)')
ax.set_ylabel('Weight (kg)')
ax.set_title('Joint Distribution of Height and Weight')
plt.colorbar(contour, label='Probability Density')
plt.show()
