import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import uuid

def kelly_fraction(p, b):
    """Calculate the Kelly fraction to bet."""
    q = 1 - p
    return max(0, (b * p - q) / b)  # Ensure non-negative bet

def simulate_betting(initial_bankroll, p, b, n_iterations=100):
    """Simulate betting using the Kelly Criterion."""
    bankroll = initial_bankroll
    bankroll_history = [bankroll]

    for _ in range(n_iterations):
        fraction = kelly_fraction(p, b)
        bet_size = fraction * bankroll

        if np.random.random() < p:  # Win
            bankroll += bet_size * b
        else:  # Lose
            bankroll -= bet_size

        bankroll_history.append(bankroll)

        if bankroll <= 0:
            break

    return bankroll_history

def plot_bankroll_history(bankroll_history, initial_bankroll, p, b, n_iterations):
    """Plot the bankroll history and save it to a file."""
    plt.figure(figsize=(12, 6))
    plt.plot(bankroll_history, label='Bankroll Over Time', color='blue')
    plt.title(f'Bankroll Growth Using Kelly Criterion ({n_iterations} Bets)', fontsize=14)
    plt.xlabel('Bet Number', fontsize=12)
    plt.ylabel('Bankroll ($)', fontsize=12)
    plt.axhline(y=initial_bankroll, color='red', linestyle='--', label='Initial Bankroll')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Create static directory if it doesn't exist
    Path("static").mkdir(exist_ok=True)

    # Save the plot to a file
    plot_path = Path(f"static/plot_{uuid.uuid4().hex}.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path
