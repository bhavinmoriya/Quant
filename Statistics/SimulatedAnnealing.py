import numpy as np
import matplotlib.pyplot as plt

# Define the energy function
def E(x):
    return x**4 - x**2

# Simulated annealing function (modified to track history)
def simulated_annealing(E, n_steps=10000):
    x = np.random.randn()
    T = 1.0
    cooling = 0.999

    best_x = x
    best_E = E(x)

    # Track the history of x and E(x)
    history_x = [x]
    history_E = [E(x)]

    for _ in range(n_steps):
        x_new = x + np.random.normal(0, 0.5)  # Smaller step for smoother visualization
        dE = E(x_new) - E(x)

        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            x = x_new

        if E(x) < best_E:
            best_x = x
            best_E = E(x)

        # Record history
        history_x.append(x)
        history_E.append(E(x))

        T *= cooling

    return best_x, best_E, history_x, history_E

# Run simulated annealing
best_x, best_E, history_x, history_E = simulated_annealing(E)

# Plot the function and convergence path
x_vals = np.linspace(-1.5, 1.5, 500)
y_vals = E(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=r'$E(x) = x^4 - x^2$', color='blue')
plt.scatter(history_x, history_E, color='red', alpha=0.3, label='Exploration path')
plt.scatter(best_x, best_E, color='green', s=100, label=f'Global minimum\nx={best_x:.3f}, E={best_E:.3f}')
plt.title('Simulated Annealing: Convergence to Global Minimum')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.legend()
plt.grid(True)
plt.show()
