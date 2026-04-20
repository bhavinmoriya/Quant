def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule.")
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    integral = (h / 3) * (y[0] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]) + y[-1])
    return integral

# Example usage:
f = lambda x: x ** 2
a, b, n = 0, 1, 4
print(simpsons_rule(f, a, b, n))  # Output: 0.3333333333333333
