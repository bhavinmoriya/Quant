import numpy as np

def expected_value(threshold, trials=100000):
    total = 0

    for _ in range(trials):
        while True:
            roll = np.random.randint(1, 7)
            if roll >= threshold:
                total += roll
                break

    return total / trials


def find_optimal():
    results = {}
    for t in range(1, 7):
        results[t] = expected_value(t)

    return results


if __name__ == "__main__":
    res = find_optimal()
    for k, v in res.items():
        print(f"Threshold {k}: {v:.4f}")
