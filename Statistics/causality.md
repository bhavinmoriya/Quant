Here’s a **Python program** that demonstrates the concept of **causality** using a simple example: **Does studying more hours lead to higher exam scores?**

This program:
1. Generates synthetic data for **study hours** and **exam scores**.
2. Uses **statistical correlation** to check if there’s a relationship.
3. Introduces a **confounding variable** (e.g., prior knowledge) to show why correlation ≠ causation.
4. Uses **linear regression** to model the causal relationship.

---

### **Python Code: Causality Example**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 100
study_hours = np.random.uniform(1, 10, n_samples)  # Independent variable (cause)
prior_knowledge = np.random.uniform(1, 10, n_samples)  # Confounding variable
exam_scores = 50 + 3 * study_hours + 2 * prior_knowledge + np.random.normal(0, 5, n_samples)  # Dependent variable (effect)

# Create a DataFrame
data = pd.DataFrame({
    'StudyHours': study_hours,
    'PriorKnowledge': prior_knowledge,
    'ExamScores': exam_scores
})

# 1. Correlation between StudyHours and ExamScores
correlation = data['StudyHours'].corr(data['ExamScores'])
print(f"Correlation between Study Hours and Exam Scores: {correlation:.2f}")

# 2. Plot the relationship
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='StudyHours', y='ExamScores', data=data)
plt.title("Study Hours vs. Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")

# 3. Linear Regression: Does studying cause higher scores?
X = data[['StudyHours']]
y = data['ExamScores']
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Regression: ExamScores = {slope:.2f} * StudyHours + {intercept:.2f}")

# Plot regression line
plt.plot(X, model.predict(X), color='red', label=f'Regression: y = {slope:.2f}x + {intercept:.2f}')
plt.legend()

# 4. Introduce confounding: Prior Knowledge
plt.subplot(1, 2, 2)
sns.scatterplot(x='PriorKnowledge', y='ExamScores', data=data)
plt.title("Prior Knowledge vs. Exam Scores")
plt.xlabel("Prior Knowledge")
plt.ylabel("Exam Scores")
plt.tight_layout()
plt.show()

# 5. Multiple Regression: Control for confounding
X_confounded = data[['StudyHours', 'PriorKnowledge']]
model_confounded = LinearRegression()
model_confounded.fit(X_confounded, y)
print("\nMultiple Regression Coefficients:")
print(f"StudyHours: {model_confounded.coef_[0]:.2f}")
print(f"PriorKnowledge: {model_confounded.coef_[1]:.2f}")
```

---

---

### **Key Takeaways from the Code**
1. **Correlation ≠ Causation**:
   - The correlation between `StudyHours` and `ExamScores` is high, but this alone doesn’t prove causation.
   - The confounding variable (`PriorKnowledge`) also affects `ExamScores`.

2. **Linear Regression**:
   - Simple regression shows a positive relationship between study hours and exam scores.
   - Multiple regression reveals the **true causal effect** of study hours by controlling for prior knowledge.

3. **Visualization**:
   - The scatter plots show the relationships between variables.

---

### **How to Extend This Example**
- Use **causal inference libraries** like [`DoWhy`](https://github.com/py-why/dowhy) for more rigorous causal analysis.
- Simulate **randomized controlled trials (RCTs)** to isolate causal effects.
- Explore **Granger causality** for time-series data.

---
Would you like me to:
1. Explain any part of the code in more detail?
2. Show how to use `DoWhy` for causal inference?
3. Adapt the example to a different causality scenario?
