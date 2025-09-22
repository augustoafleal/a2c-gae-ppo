# A2C & PPO (PyTorch)

PyTorch implementation of **Advantage Actor-Critic (A2C)** and **Proximal Policy Optimization (PPO)** algorithms, supporting both tabular/continuous environments and Atari (convolutional networks).

## Installation
    pip install -r requirements.txt

## Quick Usage
    import torch
    from a2c_gae_ppo import A2C

    hp = {
        "gamma": 0.99,
        "lam": 0.95,
        "ent_coef": 0.01,
        "stack_size": 4
    }

    agent = A2C(
        agent_type="ppo",        # "a2c" or "ppo"
        n_features=4,            # state dimension (e.g., CartPole)
        n_actions=2,             # number of actions
        device=torch.device("cpu"),
        critic_lr=1e-3,
        actor_lr=1e-3,
        n_envs=1,
        atari_mode=False,
        ppo_epochs=4,
        ppo_batch_size=64,
        clip_coef=0.2
    )

    state = torch.rand(1, 4)  # example state
    action, log_prob, value, entropy = agent.select_action(state)
    print("Action:", action.item())

## Expected `rollouts` format
The `update_agent` method expects a dictionary `rollouts` with the following keys:

- `states`: (T, N, obs_dim) or (T, N, C, H, W) for Atari  
- `actions`: (T, N)  
- `rewards`: (T, N)  
- `value_preds`: (T, N)  
- `masks`: (T, N)  
- `old_log_probs`: (T, N)  
- `entropies`: (T, N)

Where:
- T = number of steps per rollout  
- N = number of environments (`n_envs`)

## Quick Notes
- `atari_mode=True` uses convolutional layers and expects `stack_size` channels (default 4).  
- The base class uses a combined optimizer (`optim`) for feature extractor + actor/critic. Separate `actor_optim`/`critic_optim` exist but are optional.  
- `A2C(agent_type="a2c", ...)` returns `A2CSimple`; `A2C(agent_type="ppo", ...)` returns `PPO`.