import numpy as np

# Transition matrix: Bull (0) and Bear (1) markets
P = np.array([
    [0.9, 0.1],  # Bull → Bull, Bull → Bear
    [0.2, 0.8]   # Bear → Bull, Bear → Bear
])

# Simulate 1000 steps
current_state = 0  # Start in Bull
states = [current_state]
for _ in range(999):
    current_state = np.random.choice([0, 1], p=P[current_state])
    states.append(current_state)

# Count time spent in each state
bull_days = states.count(0)
bear_days = states.count(1)
print(f"Days in Bull Market: {bull_days}, Bear Market: {bear_days}")

# Calculate stationary distribution (πP = π)
eigenvalues, eigenvectors = np.linalg.eig(P.T)
stationary = eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0]
stationary = stationary / stationary.sum()  # Normalize
print("Stationary Distribution:", stationary)
