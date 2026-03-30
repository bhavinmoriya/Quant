Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall) are **still widely used**, but their role has **shifted** in modern risk management and quantitative finance.

---

# 1. Where VaR stands today

VaR answers: *“what is the worst loss at confidence level α?”*

\mathrm{VaR}_{\alpha}(X)=\inf{x\in\mathbb{R}:P(X\le x)\ge \alpha}

### Why it was dominant

* Easy to communicate (single number)
* Regulatory adoption (Basel II/III historically)
* Fast to compute (parametric / historical)

### The problem (you should care as a mathematician)

* **Not coherent** (violates subadditivity in general)
* Blind to **tail severity**
* Highly sensitive to distributional assumptions

👉 Example insight:
Two portfolios can have same VaR but wildly different catastrophic losses.

---

# 2. CVaR (Expected Shortfall) — the upgrade

CVaR answers: *“if things go bad, how bad on average?”*

\mathrm{CVaR}*{\alpha}(X)=\mathbb{E}[X \mid X \ge \mathrm{VaR}*{\alpha}(X)]

### Why it is state-of-the-art (more or less)

* **Coherent risk measure** (important theoretically)
* Captures **tail risk**
* Convex → **optimization-friendly**

👉 This is why **Basel III shifted from VaR to Expected Shortfall**.

---

# 3. What’s actually “state of the art” in 2026?

Think of VaR/CVaR as **baseline primitives**, not the frontier.

### Still used in:

* Banking regulation (CVaR mainly)
* Portfolio optimization
* Risk reporting dashboards

---

### But modern quant risk uses:

#### (A) Tail-focused models

* Extreme Value Theory (EVT)
* Peaks-over-threshold models
  → Better modeling of rare events

---

#### (B) Distribution-aware approaches

* Full predictive distributions (not just quantiles)
* Bayesian risk estimation

---

#### (C) Scenario & stress testing

* Macro shocks
* Liquidity crises
* Regime shifts

---

#### (D) Machine learning risk models

* Quantile regression (direct VaR estimation)
* Deep distribution models
* Copula-based dependence learning

---

#### (E) Dynamic risk measures

* Time-consistent CVaR
* Conditional risk processes

---

# 4. A deeper mathematical lens (you’ll like this)

VaR is essentially a **quantile functional**:
[
\mathrm{VaR}_\alpha = F^{-1}(\alpha)
]

CVaR is a **tail integral functional**:
[
\mathrm{CVaR}*\alpha = \frac{1}{1-\alpha}\int*\alpha^1 F^{-1}(u),du
]

👉 This connects directly to:

* Functional analysis
* Measure theory
* Convex duality (CVaR has a dual representation!)

---

# 5. Python experiment (to sharpen intuition)

Simulate heavy tails (Student-t) vs Gaussian:

```python
import numpy as np
import scipy.stats as stats

np.random.seed(42)

n = 100000

# Gaussian vs heavy-tailed
gaussian = np.random.normal(0, 1, n)
student_t = stats.t(df=3).rvs(n)

alpha = 0.95

def var(x, alpha):
    return np.quantile(x, alpha)

def cvar(x, alpha):
    v = var(x, alpha)
    return x[x >= v].mean()

for name, data in [("Gaussian", gaussian), ("Student-t", student_t)]:
    print(name)
    print("VaR:", var(data, alpha))
    print("CVaR:", cvar(data, alpha))
    print()
```

### What you’ll observe

* VaR might look similar
* CVaR explodes for heavy tails

👉 This is *exactly* why finance moved beyond VaR.

---

# 6. Bottom line (clear, no fluff)

* VaR → **legacy but still used**
* CVaR → **current standard in many areas**
* True “state of the art” → **distributional + tail + dynamic models**

---


