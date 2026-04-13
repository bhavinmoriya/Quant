import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# 2. Linear Regression: Does studying cause higher scores?
X = data[['StudyHours']]
y = data['ExamScores']
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Regression: ExamScores = {slope:.2f} * StudyHours + {intercept:.2f}")

# 3. Multiple Regression: Control for confounding
X_confounded = data[['StudyHours', 'PriorKnowledge']]
model_confounded = LinearRegression()
model_confounded.fit(X_confounded, y)
print("\nMultiple Regression Coefficients:")
print(f"StudyHours: {model_confounded.coef_[0]:.2f}")
print(f"PriorKnowledge: {model_confounded.coef_[1]:.2f}")

# 4. Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(
    data['StudyHours'],
    data['PriorKnowledge'],
    data['ExamScores'],
    c='blue',
    marker='o',
    label='Data Points'
)

# Create a meshgrid for the regression plane
study_hours_grid, prior_knowledge_grid = np.meshgrid(
    np.linspace(data['StudyHours'].min(), data['StudyHours'].max(), 10),
    np.linspace(data['PriorKnowledge'].min(), data['PriorKnowledge'].max(), 10)
)

# Predict ExamScores for the grid
exam_scores_grid = (
    model_confounded.intercept_ +
    model_confounded.coef_[0] * study_hours_grid +
    model_confounded.coef_[1] * prior_knowledge_grid
)

# Plot the regression plane
ax.plot_surface(
    study_hours_grid,
    prior_knowledge_grid,
    exam_scores_grid,
    alpha=0.5,
    color='red',
    label='Regression Plane'
)

# Labels and title
ax.set_xlabel('Study Hours')
ax.set_ylabel('Prior Knowledge')
ax.set_zlabel('Exam Scores')
ax.set_title('3D Visualization: Study Hours, Prior Knowledge, and Exam Scores')

plt.legend()
plt.tight_layout()
plt.show()

# 2. Plot the relationship
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='StudyHours', y='ExamScores', data=data)
plt.title("Study Hours vs. Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")
# Plot regression line
plt.plot(X, model.predict(X), color='red', label=f'Regression: y = {slope:.2f}x + {intercept:.2f}')
plt.legend()
