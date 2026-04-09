import numpy as np
import pandas as pd

# Define the joint probability distribution for two discrete random variables X and Y
joint_dist = {
    (0, 0): 0.0,
    (0, 1): 0.0,
    (0, 2): 0.25,
    (1, 0): 0.0,
    (1, 1): 0.5,
    (1, 2): 0.0,
    (2, 0): 0.25,
    (2, 1): 0.0,
    (2, 2): 0.0,
}

# Create a DataFrame for better visualization
index = sorted(set(x for x, _ in joint_dist.keys()))
columns = sorted(set(y for _, y in joint_dist.keys()))
df = pd.DataFrame(index=index, columns=columns)

for (x, y), prob in joint_dist.items():
    df.loc[x, y] = prob

print("Joint Distribution Table:")
print(df)

# Calculate marginal distributions
marginal_x = df.sum(axis=1)
marginal_y = df.sum(axis=0)

print("\nMarginal Distribution of X:")
print(marginal_x)
print("\nMarginal Distribution of Y:")
print(marginal_y)

# Calculate conditional distribution P(Y|X=1)
conditional_y_given_x1 = df.loc[1] / marginal_x[1]
print("\nConditional Distribution P(Y|X=1):")
print(conditional_y_given_x1)
