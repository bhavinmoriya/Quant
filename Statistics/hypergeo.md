# **Hypergeometric Distribution: Intuition, Definition, and Applications**

The **hypergeometric distribution** is a discrete probability distribution that models the number of **successes in a sequence of draws without replacement** from a finite population. It’s commonly used in scenarios like quality control, lottery systems, and ecological sampling, where sampling is done **without replacement**.

---

---

## **1. Intuition**
Imagine you have a **finite population** (e.g., a deck of cards, a batch of products, or a pond of fish) with:
- **\( N \)**: Total number of items in the population.
- **\( K \)**: Number of "successes" in the population (e.g., defective items, aces in a deck, tagged fish).
- **\( n \)**: Number of items drawn from the population **without replacement**.
- **\( k \)**: Number of successes in the \( n \) drawn items.

The hypergeometric distribution answers the question:
> *What is the probability of drawing exactly \( k \) successes in \( n \) draws from a population of size \( N \) containing \( K \) successes?*

---

---

## **2. Definition**
Let \( X \) be a random variable representing the **number of successes in \( n \) draws without replacement** from a population of size \( N \) containing \( K \) successes. The probability mass function (PMF) of \( X \) is:

\[
P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}} \quad \text{for } k = \max(0, n - (N - K)), \dots, \min(n, K)
\]

### **Interpretation of the PMF**
- **Numerator**:
  - \( \binom{K}{k} \): Number of ways to choose \( k \) successes from the \( K \) available.
  - \( \binom{N-K}{n-k} \): Number of ways to choose the remaining \( n-k \) failures from the \( N-K \) available.
- **Denominator**:
  - \( \binom{N}{n} \): Total number of ways to choose \( n \) items from the population of size \( N \).

---

---

## **3. Key Properties**


Hypergeometric Distribution Properties


| Property               | Formula                                                                 |
|------------------------|-------------------------------------------------------------------------|
| **Support**            | \( k = \max(0, n - (N - K)), \dots, \min(n, K) \)                     |
| **PMF**                | \( P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}} \)     |
| **Mean (Expected Value)** | \( \mathbb{E}[X] = n \cdot \frac{K}{N} \)                           |
| **Variance**           | \( \text{Var}(X) = n \cdot \frac{K}{N} \cdot \left(1 - \frac{K}{N}\right) \cdot \frac{N-n}{N-1} \) |
| **Mode**               | \( \text{Mode}(X) = \left\lfloor (n+1) \cdot \frac{K+1}{N+2} \right\rfloor \) |

---

---

## **4. Relationship to Binomial Distribution**
The hypergeometric distribution is similar to the **binomial distribution**, but with a critical difference:

| Feature               | Hypergeometric Distribution                     | Binomial Distribution                     |
|-----------------------|-------------------------------------------------|------------------------------------------|
| **Sampling**          | Without replacement                             | With replacement                         |
| **Population Size**   | Finite (\( N \))                                | Infinite (or large enough to approximate) |
| **Probability of Success** | Changes with each draw (not independent)    | Constant (\( p \)) for each trial        |
| **Variance**          | Smaller variance (due to finite population)     | \( n p (1-p) \)                           |

**When to Use Which?**
- Use **hypergeometric** for small populations or sampling without replacement.
- Use **binomial** for large populations or sampling with replacement (or if the population is so large that the difference is negligible).

---

---

## **5. Examples**

### **Example 1: Lottery**
- **Scenario**: In a lottery, 5 winning numbers are drawn from a pool of 50 numbers. If you buy 10 tickets, what is the probability of winning with exactly 2 of your numbers?
- **Parameters**:
  - \( N = 50 \) (total numbers)
  - \( K = 5 \) (winning numbers)
  - \( n = 10 \) (your tickets)
  - \( k = 2 \) (matches)
- **Solution**:
  \[
  P(X = 2) = \frac{\binom{5}{2} \binom{45}{8}}{\binom{50}{10}}
  \]
  (Calculate the binomial coefficients to get the exact probability.)

---

### **Example 2: Quality Control**
- **Scenario**: A factory produces 100 light bulbs, 10 of which are defective. If you randomly test 20 bulbs, what is the probability that exactly 3 are defective?
- **Parameters**:
  - \( N = 100 \)
  - \( K = 10 \)
  - \( n = 20 \)
  - \( k = 3 \)
- **Solution**:
  \[
  P(X = 3) = \frac{\binom{10}{3} \binom{90}{17}}{\binom{100}{20}}
  \]

---
### **Example 3: Ecology**
- **Scenario**: A pond has 500 fish, 50 of which are tagged. If you catch 20 fish, what is the probability that exactly 4 are tagged?
- **Parameters**:
  - \( N = 500 \)
  - \( K = 50 \)
  - \( n = 20 \)
  - \( k = 4 \)
- **Solution**:
  \[
  P(X = 4) = \frac{\binom{50}{4} \binom{450}{16}}{\binom{500}{20}}
  \]

---

---
## **6. Mean and Variance Intuition**
- **Mean**: \( \mathbb{E}[X] = n \cdot \frac{K}{N} \)
  This is the same as the expected value for the binomial distribution, where \( p = \frac{K}{N} \). Intuitively, if you draw \( n \) items, the expected number of successes is proportional to the fraction of successes in the population.

- **Variance**: \( \text{Var}(X) = n \cdot \frac{K}{N} \cdot \left(1 - \frac{K}{N}\right) \cdot \frac{N-n}{N-1} \)
  The variance is smaller than the binomial variance because sampling without replacement reduces randomness (the term \( \frac{N-n}{N-1} \) is the **finite population correction factor**).

---

---
## **7. Applications**
1. **Quality Control**: Testing a sample of products for defects without replacement.
2. **Lotteries and Gambling**: Calculating the probability of winning with a certain number of matches.
3. **Ecology**: Estimating the size of animal populations using capture-recapture methods.
4. **Finance**: Modeling the probability of default in a portfolio of loans (without replacement).
5. **Epidemiology**: Estimating the spread of a disease in a finite population.

---
---
## **8. Visualizing the Hypergeometric Distribution**
The PMF of the hypergeometric distribution is **unimodal** (has a single peak) and is skewed depending on the parameters \( N \), \( K \), and \( n \). For example:
- If \( \frac{K}{N} \) is small, the distribution is right-skewed.
- If \( \frac{K}{N} \) is close to 0.5, the distribution is symmetric.

---

---
## **9. Approximations**
For large populations (\( N \to \infty \)), the hypergeometric distribution can be approximated by the **binomial distribution** with \( p = \frac{K}{N} \). This is because sampling without replacement becomes similar to sampling with replacement when the population is very large.

**Rule of Thumb**:
If \( \frac{n}{N} < 0.05 \) (i.e., the sample size is less than 5% of the population), the binomial approximation is reasonable.

---
---
## **10. Summary**
- The hypergeometric distribution models the **number of successes in \( n \) draws without replacement** from a finite population of size \( N \) with \( K \) successes.
- **PMF**: \( P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}} \).
- **Mean**: \( \mathbb{E}[X] = n \cdot \frac{K}{N} \).
- **Variance**: \( \text{Var}(X) = n \cdot \frac{K}{N} \cdot \left(1 - \frac{K}{N}\right) \cdot \frac{N-n}{N-1} \).
- **Applications**: Quality control, lotteries, ecology, finance, and epidemiology.
- **Relationship to Binomial**: Use hypergeometric for small populations or sampling without replacement; use binomial for large populations or sampling with replacement.
---

### **Hypergeometric Distribution in Finance: Applications and Use Cases**

Yes! The **hypergeometric distribution** is not just for lotteries and ecology—it has **important applications in finance**, particularly in scenarios involving **sampling without replacement** or **finite populations**. Here’s how it’s used in the financial world:

---

---

## **1. Credit Risk and Loan Portfolios**
### **Scenario: Probability of Defaults in a Loan Portfolio**
- **Problem**: A bank has issued \( N \) loans, of which \( K \) are expected to default (based on credit ratings or historical data). If the bank randomly audits \( n \) loans, what is the probability that exactly \( k \) of them are in default?
- **Hypergeometric Model**:
  - \( N \): Total number of loans in the portfolio.
  - \( K \): Number of loans expected to default.
  - \( n \): Number of loans audited.
  - \( k \): Number of defaults found in the audit.
- **PMF**:
  \[
  P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}
  \]
- **Use Case**:
  - Helps banks estimate the **risk exposure** in their loan portfolios.
  - Used in **stress testing** to model worst-case scenarios for defaults.

---

### **Example: Loan Defaults**
- **Parameters**:
  - \( N = 1000 \) (total loans)
  - \( K = 50 \) (expected defaults)
  - \( n = 100 \) (loans audited)
  - \( k = 5 \) (defaults found in audit)
- **Question**: What is the probability of finding exactly 5 defaults in the audit?
- **Solution**:
  \[
  P(X = 5) = \frac{\binom{50}{5} \binom{950}{95}}{\binom{1000}{100}}
  \]
  This probability helps the bank assess whether the observed defaults are consistent with expectations or if there’s an anomaly.

---

---

## **2. Fraud Detection**
### **Scenario: Detecting Fraudulent Transactions**
- **Problem**: A financial institution processes \( N \) transactions per day, of which \( K \) are fraudulent. If a random sample of \( n \) transactions is selected for review, what is the probability that exactly \( k \) of them are fraudulent?
- **Hypergeometric Model**:
  - \( N \): Total transactions.
  - \( K \): Number of fraudulent transactions.
  - \( n \): Sample size for review.
  - \( k \): Number of fraudulent transactions in the sample.
- **Use Case**:
  - Helps institutions **estimate the effectiveness** of their fraud detection systems.
  - Used to **calibrate sampling methods** for audits.

---
### **Example: Fraudulent Transactions**
- **Parameters**:
  - \( N = 10,000 \) (daily transactions)
  - \( K = 100 \) (fraudulent transactions)
  - \( n = 500 \) (sample size)
  - \( k = 5 \) (fraudulent transactions in sample)
- **Question**: What is the probability of finding exactly 5 fraudulent transactions in the sample?
- **Solution**:
  \[
  P(X = 5) = \frac{\binom{100}{5} \binom{9900}{495}}{\binom{10000}{500}}
  \]

---

---
## **3. Portfolio Diversification**
### **Scenario: Selecting Stocks for a Portfolio**
- **Problem**: An investor wants to build a portfolio by selecting \( n \) stocks from a universe of \( N \) stocks, of which \( K \) are considered "high-growth" (e.g., tech stocks). What is the probability that the portfolio contains exactly \( k \) high-growth stocks?
- **Hypergeometric Model**:
  - \( N \): Total stocks in the universe.
  - \( K \): Number of high-growth stocks.
  - \( n \): Number of stocks in the portfolio.
  - \( k \): Number of high-growth stocks in the portfolio.
- **Use Case**:
  - Helps investors **assess the diversification** of their portfolios.
  - Used to **model the probability** of achieving a desired exposure to specific sectors or asset classes.

---
### **Example: High-Growth Stocks**
- **Parameters**:
  - \( N = 500 \) (total stocks)
  - \( K = 100 \) (high-growth stocks)
  - \( n = 50 \) (portfolio size)
  - \( k = 10 \) (high-growth stocks in portfolio)
- **Question**: What is the probability that the portfolio contains exactly 10 high-growth stocks?
- **Solution**:
  \[
  P(X = 10) = \frac{\binom{100}{10} \binom{400}{40}}{\binom{500}{50}}
  \]

---

---
## **4. Risk Management: Value at Risk (VaR)**
### **Scenario: Estimating VaR for a Finite Portfolio**
- **Problem**: A fund holds \( N \) assets, of which \( K \) are expected to experience a significant loss (e.g., a drop of more than 10%) in a market downturn. If the fund randomly selects \( n \) assets to stress-test, what is the probability that exactly \( k \) of them will experience a significant loss?
- **Hypergeometric Model**:
  - \( N \): Total assets in the fund.
  - \( K \): Number of assets expected to experience significant losses.
  - \( n \): Number of assets stress-tested.
  - \( k \): Number of assets with significant losses in the test.
- **Use Case**:
  - Helps funds **estimate their Value at Risk (VaR)** by modeling the probability of extreme losses in a finite portfolio.
  - Used in **regulatory capital calculations** to ensure sufficient reserves for potential losses.

---
### **Example: Stress-Testing Assets**
- **Parameters**:
  - \( N = 200 \) (total assets)
  - \( K = 20 \) (assets expected to experience significant losses)
  - \( n = 50 \) (assets stress-tested)
  - \( k = 5 \) (assets with significant losses in the test)
- **Question**: What is the probability that exactly 5 assets in the stress test will experience significant losses?
- **Solution**:
  \[
  P(X = 5) = \frac{\binom{20}{5} \binom{180}{45}}{\binom{200}{50}}
  \]

---

---
## **5. Bond Portfolios and Default Correlation**
### **Scenario: Modeling Defaults in a Bond Portfolio**
- **Problem**: A bond portfolio consists of \( N \) bonds, of which \( K \) are issued by companies at risk of default. If an investor randomly selects \( n \) bonds from the portfolio, what is the probability that exactly \( k \) of them will default?
- **Hypergeometric Model**:
  - \( N \): Total bonds in the portfolio.
  - \( K \): Number of bonds at risk of default.
  - \( n \): Number of bonds selected.
  - \( k \): Number of defaults in the selection.
- **Use Case**:
  - Helps investors **model default correlation** in bond portfolios.
  - Used in **credit risk modeling** to estimate the probability of multiple defaults.

---
### **Example: Bond Defaults**
- **Parameters**:
  - \( N = 100 \) (total bonds)
  - \( K = 10 \) (bonds at risk of default)
  - \( n = 20 \) (bonds selected)
  - \( k = 2 \) (defaults in the selection)
- **Question**: What is the probability that exactly 2 of the selected bonds will default?
- **Solution**:
  \[
  P(X = 2) = \frac{\binom{10}{2} \binom{90}{18}}{\binom{100}{20}}
  \]

---

---
## **6. Algorithmic Trading: Sampling Without Replacement**
### **Scenario: Backtesting Trading Strategies**
- **Problem**: A trader wants to backtest a strategy using a sample of \( n \) historical data points from a dataset of \( N \) total data points, of which \( K \) are "profitable" (based on past performance). What is the probability that exactly \( k \) of the sampled data points are profitable?
- **Hypergeometric Model**:
  - \( N \): Total historical data points.
  - \( K \): Number of profitable data points.
  - \( n \): Sample size for backtesting.
  - \( k \): Number of profitable data points in the sample.
- **Use Case**:
  - Helps traders **assess the robustness** of their strategies by modeling the probability of achieving a certain number of profitable trades in a sample.
  - Used to **avoid overfitting** by ensuring the sample is representative of the broader dataset.

---
### **Example: Backtesting a Strategy**
- **Parameters**:
  - \( N = 1000 \) (total data points)
  - \( K = 300 \) (profitable data points)
  - \( n = 100 \) (sample size)
  - \( k = 30 \) (profitable data points in the sample)
- **Question**: What is the probability that exactly 30 of the sampled data points are profitable?
- **Solution**:
  \[
  P(X = 30) = \frac{\binom{300}{30} \binom{700}{70}}{\binom{1000}{100}}
  \]

---

---
## **7. Limitations and Approximations in Finance**
While the hypergeometric distribution is powerful, it has some limitations in financial applications:
1. **Computational Complexity**: Calculating binomial coefficients for large \( N \), \( K \), and \( n \) can be computationally intensive. In such cases, approximations (e.g., binomial or Poisson) are often used.
2. **Assumption of Known \( K \)**: The hypergeometric distribution assumes that the number of successes \( K \) in the population is known. In practice, \( K \) may be uncertain or estimated, which introduces additional complexity.
3. **Finite Population**: The hypergeometric distribution is only applicable to **finite populations**. For very large populations (e.g., global stock markets), the binomial distribution is often a reasonable approximation.

---
---
## **8. Summary of Financial Applications**


Hypergeometric Distribution in Finance


| **Application**               | **Scenario**                                      | **Use Case**                                  |
|-------------------------------|---------------------------------------------------|---------------------------------------------|
| Credit Risk                   | Loan defaults in a portfolio                     | Estimate risk exposure and stress test      |
| Fraud Detection               | Fraudulent transactions in a dataset             | Assess fraud detection effectiveness       |
| Portfolio Diversification    | Selecting high-growth stocks for a portfolio     | Model sector exposure and diversification   |
| Risk Management (VaR)        | Stress-testing assets for significant losses     | Estimate Value at Risk (VaR)                |
| Bond Portfolios               | Defaults in a bond portfolio                      | Model default correlation and credit risk   |
| Algorithmic Trading           | Backtesting trading strategies                     | Assess strategy robustness and avoid overfitting |

---
---
## **9. Key Takeaways**
- The hypergeometric distribution is **used in finance** to model scenarios involving **sampling without replacement** from a finite population.
- It is particularly useful for **credit risk, fraud detection, portfolio diversification, risk management, and algorithmic trading**.
- The distribution helps quantify probabilities in **finite populations**, where the binomial distribution (sampling with replacement) may not be appropriate.
- While powerful, it can be **computationally intensive** for large datasets, and approximations (e.g., binomial) are often used in practice.
