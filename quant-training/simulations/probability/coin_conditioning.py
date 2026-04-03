import numpy as np

def simulate(n=1_000_000):
    success = 0
    total = 0

    for _ in range(n):
        c1 = np.random.choice([0, 1])  # 1 = H
        c2 = np.random.choice([0, 1])

        if c1 + c2 >= 1:  # conditioning
            total += 1
            if c1 == 1 and c2 == 1:
                success += 1

    return success / total


if __name__ == "__main__":
    print("Simulated:", simulate())
