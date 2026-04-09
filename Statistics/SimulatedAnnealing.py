import numpy as np
def simulated_annealing(E, n_steps=10000):
    x = np.random.randn()
    T = 1.0
    cooling = 0.999
    
    best_x = x
    best_E = E(x)
    
    for _ in range(n_steps):
        x_new = x + np.random.normal(0, 1)
        
        dE = E(x_new) - E(x)
        
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            x = x_new
        
        if E(x) < best_E:
            best_x = x
            best_E = E(x)
        
        T *= cooling
    
    return best_x, best_E

# Example energy
def E(x):
    return x**4 - x**2

best = simulated_annealing(E)
print(best)
