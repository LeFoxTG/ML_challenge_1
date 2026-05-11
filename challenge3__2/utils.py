import matplotlib.pyplot as plt
import numpy as np

def plot_results(rewards, path="results/training.png"):
    plt.figure(figsize=(10,5))
    plt.plot(rewards, alpha=0.3)

    if len(rewards) > 20:
        ma = np.convolve(rewards, np.ones(20)/20, mode="valid")
        plt.plot(ma)

    plt.title("Training PPO Pitfall")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.text(0.02, 0.95, f"Mean: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}", transform=plt.gca().transAxes)

    plt.savefig(path)
    plt.close()
