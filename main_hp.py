import argparse
import json
import torch
import copy
import itertools
import train_grpo_replay_buffer
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
args = parser.parse_args()

with open(args.config, "r") as f:
    base_hp = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_grid = {
    "use_kl": [True, False],
    "ppo_epochs": [1, 4],
    "grpo_mc": [True, False],
    "replay_frac": [0.0, 0.1],
}

keys, values = zip(*hyper_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for i, combo in enumerate(combinations):
    combo["run_id"] = i

csv_filename = "hyperparameter_combinations.csv"
with open(csv_filename, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run_id"] + list(hyper_grid.keys()))
    writer.writeheader()
    for combo in combinations:
        writer.writerow(combo)

for run_id, combo in enumerate(combinations):
    print(f"\n[INFO] Starting run {run_id + 1}/{len(combinations)} with combo: {combo}")

    hp = copy.deepcopy(base_hp)
    hp.update(combo)
    print(f"[HP] Hyperparameters for this run: {hp}")

    hp["seed"] = hp.get("seed", 123) + run_id
    hp["run_id"] = run_id
    if hp["agent_type"] in ("grpo", "grpo_batch"):
        train_grpo_replay_buffer.run(hp, device)
    else:
        raise ValueError(f"Agent type {hp['agent_type']} not supported.")

    print(f"[INFO] Finished run {run_id + 1}/{len(combinations)}")
