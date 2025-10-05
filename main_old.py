import argparse
import json
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np
import time
import ale_py
from A2C import A2C
from Logger import Logger
from RenderRecorder import RenderRecorder
from AtariUtils import FireResetEnv


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
args = parser.parse_args()

with open(args.config, "r") as f:
    hp = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_render_env(env_id, atari_mode, stack_size=4, seed=0):
    if atari_mode:
        env = gym.make(env_id, frameskip=1, render_mode="rgb_array", obs_type="rgb")
        env = AtariPreprocessing(
            env,
            frame_skip=hp.get("frame_skip", 4),
            grayscale_obs=True,
            scale_obs=True,
            terminal_on_life_loss=False,
            noop_max=30,
            screen_size=84,
        )
        env = FireResetEnv(env)
        env = FrameStackObservation(env, stack_size=stack_size)
        # env.reset(seed=seed)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
        # env.reset(seed=seed)
    return env


if hp.get("load_model"):
    print(f"[INFO] Loading saved model from {hp['load_model']}...")
    agent: A2C = torch.load(hp["load_model"], map_location=device, weights_only=False)
    agent.device = device

    rewards = []
    """
    for ep in range(10):
        test_env = make_render_env(hp["env_name"], hp["atari_mode"], hp.get("stack_size", 4), seed=hp["seed"] + ep)
        state, _ = test_env.reset()
        done, total_reward = False, 0
        while not done:
            with torch.no_grad():
                if hp["atari_mode"]:
                    state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
                    action, _, _, _ = agent.select_action(state_tensor)
                else:
                    action, _, _, _ = agent.select_action(np.expand_dims(state, axis=0))
                action = action.item()
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"[INFO] Episode {ep + 1}/100 - Reward: {total_reward:.2f}")
        test_env.close()

    mean_reward = np.mean(rewards)
    print(f"[RESULT] Average reward over 100 episodes: {mean_reward:.2f}")
    """

    for i in range(5):
        recorder = RenderRecorder(fps=30)
        test_env = make_render_env(hp["env_name"], hp["atari_mode"], hp.get("stack_size", 4), seed=hp["seed"])
        state, _ = test_env.reset()
        done, total_reward = False, 0
        while not done:
            with torch.no_grad():
                if hp["atari_mode"]:
                    state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
                    action, _, _, _ = agent.select_action(state_tensor)
                else:
                    action, _, _, _ = agent.select_action(np.expand_dims(state, axis=0))
                action = action.item()

            next_state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward

            frame = test_env.render()
            recorder.capture(frame)

            state = next_state
            done = terminated or truncated

        recorder.save()
        print(f"[INFO] Video saved to {recorder.filename} | Reward for this run: {total_reward:.2f}")
    exit(0)

print("[INFO] No model loaded, starting training...")


critic_losses, actor_losses, entropies = [], [], []


def make_gym_atari_env(env_id, num_envs, seed=0, frame_skip=4, stack_size=4, render_mode=None):
    def make_single_env():
        def _init():
            env = gym.make(
                env_id,
                frameskip=1,
                render_mode=render_mode,
                obs_type="rgb",
                full_action_space=False,
                repeat_action_probability=0,
            )
            env = AtariPreprocessing(
                env,
                frame_skip=frame_skip,
                grayscale_obs=True,
                scale_obs=True,
                terminal_on_life_loss=False,
                noop_max=30,
                screen_size=84,
            )
            env = FireResetEnv(env)
            env = FrameStackObservation(env, stack_size=stack_size)
            return env

        return _init

    return SyncVectorEnv([make_single_env() for _ in range(num_envs)])


def make_classic_env(env_name, num_envs):
    def make_single_env():
        def _init():
            return gym.make(env_name)

        return _init

    return SyncVectorEnv([make_single_env() for _ in range(num_envs)])


# === Inicializa ambientes ===
if hp["atari_mode"]:
    gym.register_envs(ale_py)
    envs = make_gym_atari_env(
        hp["env_name"],
        num_envs=hp["n_envs"],
        seed=hp["seed"],
        frame_skip=hp["frame_skip"],
        stack_size=hp["stack_size"],
        render_mode=hp.get("render_mode", None),
    )
    obs_space = 84 * 84 * hp["stack_size"]  # flatten Atari frames
else:
    envs = make_classic_env(hp["env_name"], hp["n_envs"])
    obs_space = envs.single_observation_space.shape[0]

act_space = envs.single_action_space.n

# === Inicializa agente ===
agent = A2C(
    agent_type=hp["agent_type"],
    n_features=obs_space,
    n_actions=act_space,
    device=device,
    critic_lr=hp["critic_lr"],
    actor_lr=hp["actor_lr"],
    n_envs=hp["n_envs"],
    atari_mode=hp["atari_mode"],
    ppo_epochs=hp["ppo_epochs"],
    ppo_batch_size=hp["ppo_batch_size"],
    clip_coef=hp.get("clip_coef", 0.1),
)

logger = Logger()

states, _ = envs.reset(seed=hp["seed"])
episode_rewards = np.zeros(hp["n_envs"], dtype=np.float32)
last_episode_rewards = np.zeros(hp["n_envs"], dtype=np.float32)
worker_episodes = np.zeros(hp["n_envs"], dtype=int)


total_time_steps = 0
max_time_steps = 10_000_000
update = 0
start_time = time.time()

while total_time_steps < max_time_steps:

    update_start_time = time.time()
    update += 1
    # Buffers do rollout
    rollouts = {
        "actions": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], dtype=torch.long, device=device),
        "value_preds": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], device=device),
        "rewards": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], device=device),
        "old_log_probs": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], device=device),
        "masks": torch.ones(hp["n_steps_per_update"], hp["n_envs"], device=device),
        "entropies": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], device=device),  # NOVA LINHA
    }
    if hp["atari_mode"]:
        rollouts["states"] = torch.zeros(
            hp["n_steps_per_update"], hp["n_envs"], hp["stack_size"], 84, 84, device=device
        )
    else:
        rollouts["states"] = torch.zeros(hp["n_steps_per_update"], hp["n_envs"], obs_space, device=device)

    for step in range(hp["n_steps_per_update"]):
        actions, action_log_probs, state_value_preds, entropy = agent.select_action(states)

        next_states, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())

        total_time_steps += hp["n_envs"]

        episode_rewards += rewards
        if hp["atari_mode"]:
            rewards = np.clip(rewards, -1.0, 1.0)
        for i, done in enumerate(np.logical_or(terminated, truncated)):
            if done:
                last_episode_rewards[i] = episode_rewards[i]  # guarda o retorno final desse env
                logger.log(
                    worker=i,
                    episode=worker_episodes[i],  # contador por worker
                    total_steps=update * hp["n_steps_per_update"] + step,
                    total_reward=last_episode_rewards[i],
                    terminated=done,
                )
                worker_episodes[i] += 1
                episode_rewards[i] = 0.0  # reseta o acumulador

        rollouts["states"][step] = torch.as_tensor(states, device=device)
        rollouts["actions"][step] = actions
        rollouts["value_preds"][step] = state_value_preds.squeeze(-1)
        rollouts["rewards"][step] = torch.as_tensor(rewards, device=device)
        rollouts["old_log_probs"][step] = action_log_probs
        rollouts["masks"][step] = torch.as_tensor(1.0 - np.logical_or(terminated, truncated), device=device)
        rollouts["entropies"][step] = entropy  # NOVA LINHA

        states = next_states

    critic_loss, actor_loss, entropy = agent.update_agent(rollouts, hp)

    critic_losses.append(critic_loss)
    actor_losses.append(actor_loss)
    entropies.append(entropy)

    mean_reward = rollouts["rewards"].sum(dim=0).mean().cpu().item()

    update_time = time.time() - update_start_time
    elapsed_time = time.time() - start_time
    steps_remaining = max_time_steps - total_time_steps
    steps_per_update = hp["n_steps_per_update"] * hp["n_envs"]
    updates_remaining = steps_remaining / steps_per_update
    eta_seconds = updates_remaining * update_time

    if update % 5 == 0:
        print(
            f"Update {update:4d} | "
            f"CriticLoss: {critic_loss:.3f} | "
            f"ActorLoss: {actor_loss:.3f} | "
            f"Entropy: {entropy:.3f} | "
            f"Mean last rewards: {last_episode_rewards.mean():.2f} | "
            f"LastRewards per env: {last_episode_rewards}"
        )
        print(
            f"Update {update} | Total steps: {total_time_steps} | "
            f"Update time: {update_time:.2f}s | Elapsed: {elapsed_time/60:.2f} min | "
            f"ETA: {eta_seconds/60:.2f} min"
        )

torch.save(agent, "ppo_atari_agent_full.pth")

recorder = RenderRecorder(fps=30)

if hp["atari_mode"]:

    def make_render_env(env_id, seed=0):
        env = gym.make(env_id, frameskip=1, render_mode="rgb_array", obs_type="rgb")
        env = AtariPreprocessing(
            env,
            frame_skip=4,
            grayscale_obs=True,
            scale_obs=True,
            terminal_on_life_loss=False,
            noop_max=30,
            screen_size=84,
        )
        env = FireResetEnv(env)
        env = FrameStackObservation(env, stack_size=hp["stack_size"])
        env.reset(seed=seed)
        return env

    test_env = make_render_env(hp["env_name"], seed=hp["seed"])
else:
    test_env = gym.make(hp["env_name"], render_mode="rgb_array")
    test_env.reset(seed=hp["seed"])

state, _ = test_env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        if hp["atari_mode"]:
            state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
            action, _, _, _ = agent.select_action(state_tensor)
            action = action.item()
        else:
            action, _, _, _ = agent.select_action(np.expand_dims(state, axis=0))
            action = action.item()

    next_state, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward

    frame = test_env.render()
    recorder.capture(frame)

    state = next_state
    done = terminated or truncated

recorder.save()
print(f"Recompensa total no teste: {total_reward}")
