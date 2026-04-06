The **Central Limit Theorem (CLT)** and the **Law of Large Numbers (LLN)** are both fundamental concepts in probability and statistics, but they address different aspects of randomness and convergence.

---

### **1. Law of Large Numbers (LLN)**
**What it says:**
- As the number of trials or observations (`n`) increases, the **sample mean** (average) of those observations will converge to the **expected value (population mean)**.

**Key Points:**
- Focuses on the **convergence of the sample mean** to the true mean.
- Does **not** say anything about the shape of the distribution of the sample mean.
- Example: If you roll a fair die many times, the average of the rolls will approach 3.5.

---

### **2. Central Limit Theorem (CLT)**
**What it says:**
- For **any** population distribution (with finite mean and variance), the **sampling distribution of the sample mean** will approximate a **normal distribution** as the sample size (`n`) becomes large (typically `n > 30`).

**Key Points:**
- Focuses on the **distribution of the sample mean**, not just its value.
- Explains why many natural phenomena follow a normal (bell-shaped) distribution.
- Example: Even if individual data points (e.g., heights, test scores) are not normally distributed, the average of many samples will be.

---

### **Key Differences**
| Feature                | Law of Large Numbers (LLN)               | Central Limit Theorem (CLT)               |
|------------------------|------------------------------------------|-------------------------------------------|
| **Focus**              | Convergence of sample mean to true mean  | Shape of the sampling distribution        |
| **Distribution**       | Does not specify distribution shape      | Sample mean becomes normal                |
| **Sample Size**        | Works for any `n` (as `n → ∞`)           | Requires "large enough" `n` (e.g., `n > 30`) |

---

Here’s a ![visualization](https://github.com/bhavinmoriya/Quant/blob/main/Statistics/Pictures/__emitted_0.png) demonstrating the **Central Limit Theorem (CLT)** in action:

- **Top-left**: The original population distribution is **exponential** (non-normal).
- **Other plots**: As the sample size (`n`) increases from 2 to 100, the distribution of the **sample means** becomes more and more **normal (bell-shaped)**, even though the original data is not normal.

This illustrates the power of the CLT: **no matter the shape of the original distribution, the sampling distribution of the mean will tend toward normality as the sample size grows**.

