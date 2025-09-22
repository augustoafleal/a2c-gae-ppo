"""
import gymnasium as gym
import torch
from tqdm import tqdm
from gymnasium.vector import SyncVectorEnv
import numpy as np
from A2C import A2C
from RenderRecorder import RenderRecorder
from Logger import Logger

# --- Hiperparâmetros ---
hp = {
    "env_name": "LunarLander-v3",
    "atari_mode": False,
    "seed": 123,
    "n_envs": 8,
    "n_updates": 1500,
    "n_steps_per_update": 256,
    "gamma": 0.99,
    "lam": 0.95,
    "ent_coef": 0.01,
    "actor_lr": 0.0007,
    "critic_lr": 0.0007,
    "max_episode_steps": 1000,
    "use_ppo": False,
    "ppo_epochs": 10,
    "ppo_batch_size": 32,
    "agent_type": "a2c",  # "a2c" ou "ppo"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

critic_losses, actor_losses, entropies = [], [], []


# --- Ambientes paralelos ---
def make_env():
    return gym.make(hp["env_name"])


envs = SyncVectorEnv([make_env for _ in range(hp["n_envs"])])

# --- Inicializa agente ---
obs_space = envs.single_observation_space.shape[0]

"""

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing
import torch
import numpy as np
from A2C import A2C
from Logger import Logger
from RenderRecorder import RenderRecorder
from AtariUtils import FireResetEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

# --- Hiperparâmetros ---
hp = {
    # "env_name": "ALE/Breakout-v5",
    "env_name": "LunarLander-v3",
    "atari_mode": False,
    "seed": 123,
    "n_envs": 8,
    "n_updates": 1,
    "n_steps_per_update": 128,
    "gamma": 0.99,
    "lam": 0.95,
    "ent_coef": 0.01,
    "actor_lr": 0.001,
    "critic_lr": 0.005,
    "max_episode_steps": 1000,
    "use_ppo": False,
    "ppo_epochs": 3,
    "ppo_batch_size": 256,
    "agent_type": "ppo",
    "frame_skip": 4,
    "stack_size": 4,
    "clip_coef": 0.1,
    "render_mode": None,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_losses, actor_losses, entropies = [], [], []


# --- Inicialização de ambientes ---
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


# --- Escolhe tipo de ambiente ---
if hp["atari_mode"]:
    gym.register_envs(ale_py)
    envs = make_gym_atari_env(
        hp["env_name"],
        num_envs=hp["n_envs"],
        seed=hp["seed"],
        frame_skip=hp["frame_skip"],
        stack_size=hp["stack_size"],
        render_mode=hp["render_mode"],
    )
    obs_space = 84 * 84 * hp["stack_size"]  # flatten Atari frames
else:
    envs = make_classic_env(hp["env_name"], hp["n_envs"])
    obs_space = envs.single_observation_space.shape[0]


act_space = envs.single_action_space.n
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
    clip_coef=hp["clip_coef"],
)
logger = Logger()

# --- Estado inicial ---
states, _ = envs.reset(seed=hp["seed"])
# --- Loop de updates ---
# Antes do loop, inicializa lista para acumular recompensas por env
episode_rewards = np.zeros(hp["n_envs"], dtype=np.float32)
last_episode_rewards = np.zeros(hp["n_envs"], dtype=np.float32)
worker_episodes = np.zeros(hp["n_envs"], dtype=int)

# for update in tqdm(range(hp["n_updates"])):
for update in range(hp["n_updates"]):

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

        # acumula recompensas por env
        episode_rewards += rewards
        if hp["atari_mode"]:
            rewards = np.clip(rewards, -1.0, 1.0)
        # verifica se algum episódio terminou
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

    # --- Atualiza agente ---
    critic_loss, actor_loss, entropy = agent.update_agent(rollouts, hp)

    critic_losses.append(critic_loss)
    actor_losses.append(actor_loss)
    entropies.append(entropy)

    # --- Métrica de recompensa média ---
    mean_reward = rollouts["rewards"].sum(dim=0).mean().cpu().item()

    if update % 1 == 0:
        print(
            f"Update {update:4d} | "
            f"CriticLoss: {critic_loss:.3f} | "
            f"ActorLoss: {actor_loss:.3f} | "
            f"Entropy: {entropy:.3f} | "
            # f"MeanReward(rollout): {mean_reward:.2f} | "
            f"Mean last rewards: {last_episode_rewards.mean():.2f} | "
            f"LastRewards per env: {last_episode_rewards}"
        )


########################################################
## TESTES
########################################################
"""
# --- Depois do treino ---
recorder = RenderRecorder(fps=30)

# Criar um ambiente único em modo renderizável
test_env = gym.make(hp["env_name"], render_mode="rgb_array")
state, _ = test_env.reset(seed=hp["seed"])

done = False
total_reward = 0

while not done:
    with torch.no_grad():
        # Adapta a entrada para parecer um batch de 1 ambiente
        action, _, _, _ = agent.select_action(np.expand_dims(state, axis=0))

    # Converte para int (tirar do tensor e batch)
    action = action.item()

    # Executa a ação no ambiente
    next_state, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward

    # Captura frame do ambiente
    frame = test_env.render()
    recorder.capture(frame)

    state = next_state
    done = terminated or truncated

# Salva o vídeo
recorder.save()
print(f"Recompensa total no teste: {total_reward}")
"""

recorder = RenderRecorder(fps=30)

if hp["atari_mode"]:
    # === Ambiente Atari com preprocessamento + render ===
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
    # === Ambiente normal ===
    test_env = gym.make(hp["env_name"], render_mode="rgb_array")
    test_env.reset(seed=hp["seed"])

state, _ = test_env.reset()
done = False
total_reward = 0

while not done:
    with torch.no_grad():
        if hp["atari_mode"]:
            # adiciona batch dimension e manda no formato correto (1, 4, 84, 84)
            state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
            action, _, _, _ = agent.select_action(state_tensor)
            action = action.item()
        else:
            # ambiente clássico (Box2D, Mujoco, etc)
            action, _, _, _ = agent.select_action(np.expand_dims(state, axis=0))
            action = action.item()

    next_state, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward

    # capturar frame sempre em rgb_array
    frame = test_env.render()
    recorder.capture(frame)

    state = next_state
    done = terminated or truncated

recorder.save()
print(f"Recompensa total no teste: {total_reward}")
