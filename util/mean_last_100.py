import pandas as pd

csv_file = "tests/zaxxon/logger_20250926_120415.csv"

df = pd.read_csv(csv_file)

last_100 = df.tail(100)
mean_reward = last_100["total_reward"].mean()

print(f"Média dos últimos 100 episódios: {mean_reward:.2f}")
