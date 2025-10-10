import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from util.RenderRecorder import RenderRecorder
from util.AtariUtils import FireResetEnv


def make_render_env(env_id, atari_mode, stack_size=4, seed=0):
    if atari_mode:
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
        env = FrameStackObservation(env, stack_size=stack_size)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def run(hp, device):
    print(f"[INFO] Loading model from {hp['load_model']}...")
    agent = torch.load(hp["load_model"], map_location=device, weights_only=False)
    agent.device = device
    run_episodes = 1
    rewards = []
    for ep in range(run_episodes):
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
        test_env.close()
        print(f"Episode: {ep + 1} | Rewards: {total_reward}")
    mean_reward = np.mean(rewards)
    print(f"[RESULT] Average reward over {run_episodes} episodes: {mean_reward:.2f}")

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
    print(f"[INFO] Video saved to {recorder.filename} | Reward: {total_reward:.2f}")
