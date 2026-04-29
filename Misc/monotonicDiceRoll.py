import random
from itertools import combinations

def simulate_strictly_increasing_probability(num_simulations: int = 100000) -> float:
    """
    Simulate rolling three dice and estimate the probability of strictly increasing numbers.
    Args:
        num_simulations: Number of trials to run (default: 100,000).
    Returns:
        Estimated probability of strictly increasing numbers.
    """
    count = 0
    for _ in range(num_simulations):
        # Roll three dice
        dice = [random.randint(1, 6) for _ in range(3)]
        # Check if strictly increasing
        if dice[0] < dice[1] < dice[2]:
            count += 1
    return count / num_simulations

def simulate_strictly_decreasing_probability(num_simulations: int = 100000) -> float:
    """
    Simulate rolling three dice and estimate the probability of strictly increasing numbers.
    Args:
        num_simulations: Number of trials to run (default: 100,000).
    Returns:
        Estimated probability of strictly increasing numbers.
    """
    count = 0
    for _ in range(num_simulations):
        # Roll three dice
        dice = [random.randint(1, 6) for _ in range(3)]
        # Check if strictly increasing
        if dice[0] > dice[1] > dice[2]:
            count += 1
    return count / num_simulations

def analytical_probability() -> float:
    """
    Calculate the exact probability of rolling three strictly increasing numbers.
    Returns:
        Exact probability (5/54 ≈ 0.09259).
    """
    # Total possible outcomes: 6^3 = 216
    total_outcomes = 6 ** 3
    # Favorable outcomes: C(6, 3) = 20
    favorable_outcomes = len(list(combinations(range(1, 7), 3)))
    return favorable_outcomes / total_outcomes

if __name__ == "__main__":
    # Simulation
    num_simulations = 100000
    simulated_prob = simulate_strictly_increasing_probability(num_simulations)
    print(f"Simulated (for increasing) probability (after {num_simulations:,} trials): {simulated_prob:.6f} ({simulated_prob * 100:.2f}%)")

    simulated_prob = simulate_strictly_decreasing_probability(num_simulations)
    print(f"Simulated (for decreasing) probability (after {num_simulations:,} trials): {simulated_prob:.6f} ({simulated_prob * 100:.2f}%)")

    # Analytical
    exact_prob = analytical_probability()
    print(f"Exact probability: {exact_prob:.6f} ({exact_prob * 100:.2f}%)")

    # Compare
    print(f"\nDifference: {abs(simulated_prob - exact_prob):.6f}")
    
