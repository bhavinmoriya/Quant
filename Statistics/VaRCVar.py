import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Retrieve previously calculated values (assuming the previous cell was executed)
n = 100000
np.random.seed(42)
gaus_data = np.random.normal(0, 1, n)
student_data = stats.t(df=3).rvs(n)
alpha = 0.95

def var_calc(x, alpha):
    return np.quantile(x, 1 - alpha)

def cvar_calc(x, alpha):
    v = var_calc(x, alpha)
    return x[x <= v].mean()

var_gaussian = var_calc(gaus_data, alpha)
cvar_gaussian = cvar_calc(gaus_data, alpha)

var_student = var_calc(student_data, alpha)
cvar_student = cvar_calc(student_data, alpha)

# Plotting for Gaussian Distribution
plt.figure(figsize=(14, 6))

# Subplot 1: Gaussian
plt.subplot(1, 2, 1)
x = np.linspace(min(gaus_data), max(gaus_data), 1000)
plt.plot(x, stats.norm.pdf(x, 0, 1), label='PDF (Normal Distribution)')
plt.axvline(var_gaussian, color='red', linestyle='--', label=f'VaR ({alpha*100:.0f}%): {var_gaussian:.2f}')

# Shade the CVaR region
x_cvar_gaussian = x[x <= var_gaussian]
plt.fill_between(x_cvar_gaussian, 0, stats.norm.pdf(x_cvar_gaussian, 0, 1), color='orange', alpha=0.3, label='CVaR Region')

plt.title('Gaussian Distribution: VaR vs. CVaR')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Plotting for Student-t Distribution
plt.subplot(1, 2, 2)
x = np.linspace(min(student_data), max(student_data), 1000)
plt.plot(x, stats.t.pdf(x, df=3), label='PDF (Student-t Distribution, df=3)')
plt.axvline(var_student, color='red', linestyle='--', label=f'VaR ({alpha*100:.0f}%): {var_student:.2f}')

# Shade the CVaR region
x_cvar_student = x[x <= var_student]
plt.fill_between(x_cvar_student, 0, stats.t.pdf(x_cvar_student, df=3), color='orange', alpha=0.3, label='CVaR Region')

plt.title('Student-t Distribution: VaR vs. CVaR')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
