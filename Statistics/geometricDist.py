import matplotlib.pyplot as plt
import numpy as np

def k_geo_plot(p=.5, k_geo=np.arange(1, 11)):
  # Geometric distribution for p = 0.5 (first heads)
  k_geo = np.arange(1, 11)
  pmf_geo = p * ((1 - p) ** (k_geo -1))


  # Plotting
  plt.figure(figsize=(10, 6))
  plt.bar(k_geo, pmf_geo, width=0.4, label="Geometric (First Heads)", align='center', alpha=0.7)
  # plt.bar(k_two_heads + 0.4, pmf_two_heads, width=0.4, label="Two Consecutive Heads", align='center', alpha=0.7)

  plt.xlabel("Number of Flips (k)")
  plt.ylabel("Probability P(X = k)")
  # plt.title("Geometric Distribution vs. Two Consecutive Heads")
  plt.title("Geometric Distribution vs. Two Consecutive Heads")
  plt.xticks(np.arange(1, 12))
  plt.legend()
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()
  plt.savefig('geometric_vs_two_heads.png', dpi=200, bbox_inches='tight')

k_geo_plot(.5)
