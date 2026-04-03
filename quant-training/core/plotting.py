import matplotlib.pyplot as plt

def plot_wealth(histories):
    for h in histories[:20]:  # plot few paths
        plt.plot(h)

    plt.title("Wealth Trajectories")
    plt.xlabel("Steps")
    plt.ylabel("Wealth")
    plt.show()
