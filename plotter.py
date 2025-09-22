import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def main(csv_file, window_size=25, aura_factor=0.5, aura_alpha=0.1):
    """
    Plots the moving average of rewards with a subtle aura around it.

    Parameters:
        csv_file: str - path to the CSV log
        window_size: int - size of the moving average window
        aura_factor: float - factor to scale the std deviation for the aura
        aura_alpha: float - transparency of the aura
    """
    # Create plots folder if it does not exist
    os.makedirs("plots", exist_ok=True)

    # --- Read CSV ---
    df = pd.read_csv(csv_file)

    # Group by episode and compute mean and std of rewards across workers
    grouped = df.groupby("episode")["total_reward"]
    mean_rewards = grouped.mean()
    std_rewards = grouped.std()

    # Compute moving average of the mean rewards
    mean_rewards_moving_avg = mean_rewards.rolling(window=window_size).mean()

    # --- Plot ---
    plt.figure(figsize=(12, 6))

    # Moving average line
    plt.plot(
        mean_rewards_moving_avg.index,
        mean_rewards_moving_avg.values,
        color="orange",
        label=f"Moving Average ({window_size} episodes)",
        linewidth=2,
    )

    # Aura / shadow around moving average (± std scaled)
    plt.fill_between(
        mean_rewards_moving_avg.index,
        mean_rewards_moving_avg.values - aura_factor * std_rewards.values,
        mean_rewards_moving_avg.values + aura_factor * std_rewards.values,
        color="orange",
        alpha=aura_alpha,
        label="Aura (Std Dev)",
    )

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Moving Average of Rewards with Subtle Aura")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_file = os.path.join("plots", "reward_moving_avg_aura_final.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved as '{output_file}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_rewards_moving_avg_aura_final.py csv_file [window_size]")
        sys.exit(1)

    csv_file = sys.argv[1]
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    main(csv_file, window_size)
