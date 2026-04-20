### **Simpson’s Rule for Numerical Integration**

Simpson’s rule is a **numerical method** for approximating the definite integral of a function. It improves upon the trapezium rule by using **parabolic arcs** (quadratic polynomials) instead of straight lines to approximate the function over subintervals. This makes it more accurate for smooth functions, especially when the number of subintervals is even.

---

---

## **Simpson’s Rule Formula**

For a function \( f(x) \) over the interval \([a, b]\), Simpson’s rule approximates the integral as:

\[
\int_{a}^{b} f(x) \, dx \approx \frac{h}{3} \left[ f(x_0) + 4 \sum_{\text{odd } i} f(x_i) + 2 \sum_{\text{even } i} f(x_i) + f(x_n) \right]
\]

where:
- \( h = \frac{b - a}{n} \) (width of each subinterval),
- \( n \) = **even** number of subintervals,
- \( x_i = a + i \cdot h \) for \( i = 0, 1, 2, \dots, n \).

---

---

## **Key Points**
- **Requires an even number of subintervals** (\( n \) must be even).
- **More accurate** than the trapezium rule for the same number of points, especially for smooth functions.
- **Error term**: The error is proportional to \( h^4 \), making it more precise for well-behaved functions.

---

---

## **Steps to Apply Simpson’s Rule**
1. **Divide \([a, b]\) into an even number of subintervals** \( n \).
2. **Calculate \( h = \frac{b - a}{n} \)**.
3. **Evaluate \( f(x) \) at each \( x_i \)**.
4. **Apply the formula**:
   - Sum the function values at the **odd indices** (1, 3, 5, ...) and multiply by 4.
   - Sum the function values at the **even indices** (2, 4, 6, ...) and multiply by 2.
   - Add the endpoints \( f(x_0) \) and \( f(x_n) \).
   - Multiply the total by \( \frac{h}{3} \).

---

---

## **Example**

Approximate \( \int_{0}^{1} x^2 \, dx \) using Simpson’s rule with \( n = 4 \) subintervals.

- \( a = 0 \), \( b = 1 \), \( n = 4 \) (even)
- \( h = \frac{1 - 0}{4} = 0.25 \)
- Points: \( x_0 = 0 \), \( x_1 = 0.25 \), \( x_2 = 0.5 \), \( x_3 = 0.75 \), \( x_4 = 1 \)
- Function values:
  \( f(0) = 0 \),
  \( f(0.25) = 0.0625 \),
  \( f(0.5) = 0.25 \),
  \( f(0.75) = 0.5625 \),
  \( f(1) = 1 \)

Now, apply the formula:

\[
\int_{0}^{1} x^2 \, dx \approx \frac{0.25}{3} \left[ 0 + 4(0.0625 + 0.5625) + 2(0.25) + 1 \right]
\]

\[
= \frac{0.25}{3} \left[ 0 + 4(0.625) + 0.5 + 1 \right] = \frac{0.25}{3} \left[ 2.5 + 0.5 + 1 \right] = \frac{0.25}{3} \times 4 = \frac{1}{3} \approx 0.3333
\]

**Result:** The approximation matches the exact value \( \frac{1}{3} \)!

---

---

## **Python Implementation**

```python
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
```

---

---
---
## **When to Use Simpson’s Rule**
- When the function is **smooth** and **twice differentiable**.
- When you need **higher accuracy** than the trapezium rule for the same number of points.
- When the number of subintervals can be **even**.

---
