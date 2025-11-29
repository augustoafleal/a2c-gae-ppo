import argparse
import json
import torch

import train, evaluate, train_grpo_replay_buffer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
args = parser.parse_args()

with open(args.config, "r") as f:
    hp = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if hp.get("load_model"):
    print(f"[INFO] Evaluation mode. Using model from {hp['load_model']}")
    evaluate.run(hp, device)
else:
    print(f"[INFO] Training mode with agent type: {hp["agent_type"]}.")
    if hp["agent_type"] in ("grpo", "grpo_batch"):
        train_grpo_replay_buffer.run(hp, device)
    else:
        train.run(hp, device)
