import numpy as np
from scipy.stats import norm

class BayesianGaussianAD:
    def __init__(self, mu0=0, tau2=1, sigma2=1):
        self.mu0 = mu0        # prior mean
        self.tau2 = tau2      # prior variance
        self.sigma2 = sigma2  # known data variance
        self.n = 0
        self.sum_x = 0

    def update(self, x):
        """Update posterior with new data"""
        self.n += len(x)
        self.sum_x += np.sum(x)

    def posterior_params(self):
        """Compute posterior parameters"""
        if self.n == 0:
            return self.mu0, self.tau2
        
        sigma_n2 = 1 / (self.n / self.sigma2 + 1 / self.tau2)
        mu_n = sigma_n2 * (self.mu0 / self.tau2 + self.sum_x / self.sigma2)
        
        return mu_n, sigma_n2

    def predictive_prob(self, x_new):
        """Posterior predictive probability density"""
        mu_n, sigma_n2 = self.posterior_params()
        pred_var = self.sigma2 + sigma_n2
        
        return norm.pdf(x_new, mu_n, np.sqrt(pred_var))

    def is_anomaly(self, x_new, threshold=0.01):
        prob = self.predictive_prob(x_new)
        return prob < threshold, prob


# --- Example usage ---
np.random.seed(42)

data = np.random.normal(0, 1, 100)
model = BayesianGaussianAD()

model.update(data)

test_points = [0.2, 1.5, 5.0]

for x in test_points:
    anomaly, prob = model.is_anomaly(x)
    print(f"x={x:.2f}, prob={prob:.5f}, anomaly={anomaly}")
