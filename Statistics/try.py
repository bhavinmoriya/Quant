import numpy as np
import matplotlib.pyplot as plt

def kelly_fraction(p, b):
    """Calculate the Kelly fraction to bet."""
    q = 1 - p
    return (b * p - q) / b

def simulate_betting(initial_bankroll, p, b, n_iterations=100):
    """Simulate betting using the Kelly Criterion."""
    bankroll = initial_bankroll
    bankroll_history = [bankroll]

    for _ in range(n_iterations):
        # Calculate the Kelly fraction
        fraction = kelly_fraction(p, b)
        bet_size = fraction * bankroll

        # Simulate the bet outcome (win or lose)
        if np.random.random() < p:  # Win
            bankroll += bet_size * b  # Profit = bet_size * b
        else:  # Lose
            bankroll -= bet_size  # Lose the bet amount

        bankroll_history.append(bankroll)

        # Stop if bankroll is depleted
        if bankroll <= 0:
            break

    return bankroll_history

# Parameters
initial_bankroll = 10000  # Starting bankroll: $10,000
p = 0.6                  # Probability of winning: 60%
b = 1.5                  # Profit per unit bet: $1.50 profit per $1 bet
n_iterations = 100       # Number of bets

# Run the simulation
bankroll_history = simulate_betting(initial_bankroll, p, b, n_iterations)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(bankroll_history, label='Bankroll Over Time', color='blue')
plt.title('Bankroll Growth Using Kelly Criterion (100 Bets)', fontsize=14)
plt.xlabel('Bet Number', fontsize=12)
plt.ylabel('Bankroll ($)', fontsize=12)
plt.axhline(y=initial_bankroll, color='red', linestyle='--', label='Initial Bankroll')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
