import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def main(csv_file, window_size=25, aura_factor=0.5, aura_alpha=0.1, title=None):
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv(csv_file)

    grouped = df.groupby("episode")["total_reward"]
    mean_rewards = grouped.mean()
    std_rewards = grouped.std()

    mean_rewards_moving_avg = mean_rewards.rolling(window=window_size).mean()

    plt.figure(figsize=(12, 8))

    plt.plot(
        mean_rewards_moving_avg.index,
        mean_rewards_moving_avg.values,
        color="blue",
        label=f"Average rewards",
        linewidth=2,
    )

    plt.fill_between(
        mean_rewards_moving_avg.index,
        mean_rewards_moving_avg.values - aura_factor * std_rewards.values,
        mean_rewards_moving_avg.values + aura_factor * std_rewards.values,
        color="blue",
        alpha=aura_alpha,
    )

    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)

    plt.xlabel("Episodes", fontsize=32)
    plt.ylabel("Total Reward", fontsize=32)
    if title:
        plt.title(title, fontsize=32)
    plt.grid(False)
    plt.legend(fontsize=24)
    plt.tight_layout()

    output_file = os.path.join("plots", "reward_moving_avg_aura_final.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved as '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPO reward moving average with aura.")
    parser.add_argument("csv_file", help="Path to the CSV file containing rewards.")
    parser.add_argument("--window_size", type=int, default=25, help="Window size for moving average.")
    parser.add_argument("--aura_factor", type=float, default=0.5, help="Scaling factor for the reward variance aura.")
    parser.add_argument("--aura_alpha", type=float, default=0.1, help="Transparency for the aura fill.")
    parser.add_argument("--title", type=str, default=None, help="Optional title for the plot.")

    args = parser.parse_args()

    main(
        args.csv_file,
        window_size=args.window_size,
        aura_factor=args.aura_factor,
        aura_alpha=args.aura_alpha,
        title=args.title,
    )
