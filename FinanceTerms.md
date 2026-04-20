### **Put-Call Parity: The Fundamental Link Between Call and Put Options**

**Put-Call Parity** is a **key principle in options trading** that defines the relationship between the prices of **European call and put options** with the **same strike price (K) and expiration (T)**. It ensures that the market remains **arbitrage-free**—meaning no trader can make a risk-free profit by exploiting price differences.

---

---

### **📌 The Put-Call Parity Formula**
For **European options** (which can only be exercised at expiration), the relationship is:

**Call Price (C) + Present Value of Strike (K) = Put Price (P) + Stock Price (S)**

Or, in mathematical terms:
**C + K·e^(-rT) = P + S**

Where:
- **C** = Price of the call option
- **P** = Price of the put option
- **S** = Current stock price
- **K** = Strike price
- **r** = Risk-free interest rate
- **T** = Time to expiration
- **e^(-rT)** = Discount factor (present value of 1 unit of currency at time T)

---

---

### **🔍 Intuition Behind Put-Call Parity**
The formula ensures that **two equivalent portfolios** must have the **same value** to prevent arbitrage:

| Portfolio A (Long Call + Cash) | Portfolio B (Long Put + Stock) |
|--------------------------------|--------------------------------|
| 1 Call Option (C)              | 1 Put Option (P)               |
| Cash = K·e^(-rT)               | 1 Share of Stock (S)           |

At expiration, **both portfolios will have the same payoff**, so their **current values must be equal**.

---

---

### **📊 Example**
Let’s say:
- Stock price (**S**) = $100
- Strike price (**K**) = $105
- Risk-free rate (**r**) = 5% (0.05)
- Time to expiration (**T**) = 1 year
- Call price (**C**) = $8
- Put price (**P**) = ?

Using Put-Call Parity:
**C + K·e^(-rT) = P + S**
**8 + 105·e^(-0.05·1) = P + 100**
**8 + 105·0.9512 ≈ P + 100**
**8 + 100 ≈ P + 100**
**P ≈ $8**

So, the put price should be **$8** to maintain parity.

---
---

### **⚠️ Why Put-Call Parity Matters**
1. **Arbitrage-Free Pricing**: If the equation doesn’t hold, traders can exploit the imbalance for **risk-free profit** (arbitrage).
   - Example: If **C + K·e^(-rT) > P + S**, traders can **sell the call, buy the put, short the stock, and invest K·e^(-rT)** to lock in a profit.

2. **Synthetic Positions**: Allows traders to **replicate** the payoff of one option using a combination of the other option and the underlying asset.
   - Example: A **synthetic long call** = Long put + Long stock - Cash (K·e^(-rT)).

3. **Pricing Consistency**: Ensures that call and put prices are **consistent** with each other and the underlying stock.

---
---
### **📉 Put-Call Parity for American Options**
Put-Call Parity **does not strictly hold** for American options (which can be exercised early) because:
- Early exercise introduces **additional complexity**.
- However, **bounds** can still be derived (e.g., **C ≥ P + S - K** for American options).

---
---
### **💡 Real-World Applications**
- **Options Trading Strategies**: Used to create **synthetic positions** (e.g., synthetic long/short stock).
- **Volatility Arbitrage**: Traders use put-call parity to check for mispricing between calls and puts.
- **Portfolio Hedging**: Helps in constructing **delta-neutral** or **market-neutral** portfolios.

---
---
### **📌 Summary Table**
| Concept | Formula | Interpretation |
|---------|---------|----------------|
| **Put-Call Parity** | C + K·e^(-rT) = P + S | Ensures no arbitrage between calls and puts. |
| **Synthetic Call** | C = P + S - K·e^(-rT) | Replicate a call using a put, stock, and cash. |
| **Synthetic Put** | P = C - S + K·e^(-rT) | Replicate a put using a call, stock, and cash. |
| **Synthetic Stock** | S = C - P + K·e^(-rT) | Replicate stock using options and cash. |
| **Synthetic Cash** | K·e^(-rT) = C - P + S | Replicate cash using options and stock. |

---
