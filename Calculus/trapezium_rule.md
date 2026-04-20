The **trapezium rule** (also known as the **trapezoidal rule**) is a numerical method for approximating the definite integral of a function. It works by dividing the area under the curve into trapezoids (rather than rectangles, as in the Riemann sum) and summing their areas.

---

## **Trapezium Rule Formula**

Given a function \( f(x) \) over the interval \([a, b]\), the trapezium rule approximates the integral as:

\[
\int_{a}^{b} f(x) \, dx \approx \frac{h}{2} \left[ f(a) + 2 \sum_{i=1}^{n-1} f(x_i) + f(b) \right]
\]

where:
- \( h = \frac{b - a}{n} \) (width of each subinterval),
- \( n \) = number of subintervals (trapezoids),
- \( x_i = a + i \cdot h \) for \( i = 0, 1, 2, \dots, n \).

---

## **Steps to Apply the Trapezium Rule**

1. **Divide the interval \([a, b]\) into \( n \) equal subintervals** of width \( h \).
2. **Evaluate the function \( f(x) \) at each point** \( x_0, x_1, \dots, x_n \).
3. **Apply the formula** to sum the areas of the trapezoids.

---

## **Example**

Let’s approximate \( \int_{0}^{1} x^2 \, dx \) using \( n = 4 \) subintervals.

- \( a = 0 \), \( b = 1 \), \( n = 4 \)
- \( h = \frac{1 - 0}{4} = 0.25 \)
- Points: \( x_0 = 0 \), \( x_1 = 0.25 \), \( x_2 = 0.5 \), \( x_3 = 0.75 \), \( x_4 = 1 \)
- Function values: \( f(0) = 0 \), \( f(0.25) = 0.0625 \), \( f(0.5) = 0.25 \), \( f(0.75) = 0.5625 \), \( f(1) = 1 \)

Now, apply the formula:

\[
\int_{0}^{1} x^2 \, dx \approx \frac{0.25}{2} \left[ 0 + 2(0.0625 + 0.25 + 0.5625) + 1 \right] = 0.34375
\]

The exact value is \( \frac{1}{3} \approx 0.3333 \), so the approximation is close.

---

## **Python Implementation**

Here’s how you can implement the trapezium rule in Python:

```python
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
```

---

## **When to Use the Trapezium Rule**
- When the function is **smooth** and **well-behaved**.
- When you need a **simple, fast approximation** (though Simpson’s rule is often more accurate for the same number of points).

---
