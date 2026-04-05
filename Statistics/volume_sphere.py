import numpy as np

def probability_sum_squares_at_most_1(n, num_simulations=1000_000):
    """
    Simulates choosing n numbers uniformly from -1 to 1,
    and calculates the probability that the sum of their squares is at most 1.

    Args:
        n (int): Number of random numbers to choose.
        num_simulations (int): Number of simulations to run (default: 1000).

    Returns:
        float: Probability that the sum of squares is at most 1.
    """
    # Simulate choosing n numbers uniformly from -1 to 1
    random_numbers = np.random.uniform(-1, 1, (num_simulations, n))

    # Calculate the sum of squares for each simulation
    sum_squares = np.sum(random_numbers**2, axis=1)

    # Check if the sum of squares is at most 1
    at_most_1 = sum_squares <= 1

    # Calculate the probability
    probability = np.mean(at_most_1)

    return probability

# Example usage
n = 5
probability = probability_sum_squares_at_most_1(n)
print(f"The probability that the sum of squares is at most 1 for n={n} is: {probability:.3f}")

def odd_factorial(n):
  half_factorial = np.sqrt(np.pi)/2
  if n == 1:
    return half_factorial
  else:
    for i in range(3,n+1,2):
      half_factorial *= i/2
    return half_factorial

from math import factorial
import numpy as np
for n in range(1,32,2):
  probability = probability_sum_squares_at_most_1(n, 1000_000)*2**n
  print(f"The volume of a unit sphere of dimension n={n} is: {probability:.3f}",np.pi**(n/2)/factorial(int(n/2)) if n %2 ==0 else np.pi**(n/2)/odd_factorial(int(n)))
