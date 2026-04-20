def trapezium_rule(f, a, b, n):
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    integral = (h / 2) * (y[0] + 2 * sum(y[1:-1]) + y[-1])
    return integral

# Example usage:
f = lambda x: x ** 2
a, b, n = 0, 1, 4
print(trapezium_rule(f, a, b, n))  # Output: 0.34375
