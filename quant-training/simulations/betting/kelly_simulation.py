import numpy as np
from core.probability_utils import kelly_fraction


def simulate(p=0.55, b=1, steps=1000):
    f = kelly_fraction(p, b)
    wealth = 1

    history = []

    for _ in range(steps):
        if np.random.rand() < p:
            wealth *= (1 + f * b)
        else:
            wealth *= (1 - f)

        history.append(wealth)

    return history


if __name__ == "__main__":
    runs = [simulate()[-1] for _ in range(1000)]
    print("Median:", np.median(runs))
