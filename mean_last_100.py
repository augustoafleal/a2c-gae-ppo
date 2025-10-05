import pandas as pd

# Caminho para o CSV
csv_file = "tests/zaxxon/logger_20250926_120415.csv"

# Carregar CSV
df = pd.read_csv(csv_file)

# Agrupar por episódio (soma das recompensas de todos os workers)
# episode_rewards = df.groupby("episode")["total_reward"].mean().reset_index()

# Selecionar os últimos 100 episódios
last_100 = df.tail(100)

# Calcular a média
mean_reward = last_100["total_reward"].mean()

print(f"Média dos últimos 100 episódios: {mean_reward:.2f}")
