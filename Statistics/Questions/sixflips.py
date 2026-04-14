import random

# Function to simulate the number of flips to get two consecutive heads
def simulate_two_consecutive_heads():
    flips = 0
    consecutive_heads = 0
    while True:
        flip = random.choice(['H', 'T'])
        flips += 1
        if flip == 'H':
            consecutive_heads += 1
            if consecutive_heads == 2:
                return flips
        else:
            consecutive_heads = 0

# Run the simulation for a large number of trials
num_trials = 10000
trial_results = [simulate_two_consecutive_heads() for _ in range(num_trials)]

# Calculate the average number of flips
average_flips = sum(trial_results) / num_trials
average_flips

import sympy

# Define symbolic variables
E0, E1 = sympy.symbols('E0 E1')

# Define the equations
eq1 = sympy.Eq(E0, 1 + 0.5 * E1 + 0.5 * E0)
eq2 = sympy.Eq(E1, 1 + 0.5 * E0)

# Solve the system of equations
solution = sympy.solve([eq1, eq2], (E0, E1))

print(f"E0 (Expected flips from start): {solution[E0]}")
print(f"E1 (Expected flips after one H): {solution[E1]}")
