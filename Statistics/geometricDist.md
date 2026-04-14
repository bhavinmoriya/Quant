The **geometric distribution** is a discrete probability distribution that models the number of trials needed to get the **first success** in repeated, independent Bernoulli trials (e.g., coin flips, where "success" could be flipping heads).

---

---

## **Key Properties of the Geometric Distribution**
1. **Definition**:
   - Let \( p \) be the probability of success on a single trial (e.g., \( p = 0.5 \) for a fair coin flip).
   - The geometric distribution gives the probability that the **first success** occurs on the \( k \)-th trial.

2. **Probability Mass Function (PMF)**:
   \[
   P(X = k) = (1 - p)^{k-1} \cdot p
   \]
   - \( k \) is the number of trials until the first success (\( k = 1, 2, 3, \dots \)).
   - Example: For a fair coin (\( p = 0.5 \)):
     - \( P(X = 1) = 0.5 \) (first flip is heads).
     - \( P(X = 2) = 0.25 \) (tails then heads).
     - \( P(X = 3) = 0.125 \) (tails, tails, heads), and so on.

3. **Expected Value (Mean)**:
   \[
   E[X] = \frac{1}{p}
   \]
   - For a fair coin (\( p = 0.5 \)), the expected number of flips to get the first head is **2**.

4. **Variance**:
   \[
   \text{Var}(X) = \frac{1 - p}{p^2}
   \]
   - For \( p = 0.5 \), the variance is **2**.

5. **Memoryless Property**:
   - The geometric distribution is **memoryless**, meaning the probability of success on the next trial is independent of how many failures have already occurred.
   - Mathematically: \( P(X > s + t \mid X > s) = P(X > t) \).

---

---

## **How It Relates to the Coin Flip Problem**
- The geometric distribution applies to **waiting for the first success** (e.g., first heads).
- For **two consecutive heads**, the process is **not geometric** because the trials are no longer independent. The outcome depends on the previous flip, so the memoryless property does not hold.

---

---
## **Example: Geometric Distribution for First Heads**
If you flip a fair coin until you get the first heads:
- The probability of getting the first heads on the 3rd flip is:
  \[
  P(X = 3) = (1 - 0.5)^{2} \cdot 0.5 = 0.125
  \]
- On average, you’d expect to need **2 flips** to get the first heads.

---
