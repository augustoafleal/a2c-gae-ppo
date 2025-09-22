import numpy as np
import torch
import torch.nn as nn
from torch import optim


class A2CBase(nn.Module):
    """
    Base class with common A2C logic (GAE, advantage computation, update).
    """

    def __init__(
        self,
        n_features,
        n_actions,
        device,
        critic_lr,
        actor_lr,
        n_envs,
        atari_mode,
        ppo_epochs=None,
        ppo_batch_size=None,
        clip_coef=None,
        stack_size=4,  # NOVO
    ):
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.atari_mode = atari_mode

        if atari_mode:
            self.conv1 = nn.Conv2d(in_channels=stack_size, out_channels=32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            self.flattened_size = 64 * 7 * 7  # depois de convs e input 84x84
            self.flatten = nn.Flatten()

            self.critic = nn.Sequential(nn.Linear(self.flattened_size, 512), nn.ReLU(), nn.Linear(512, 1)).to(device)

            self.actor = nn.Sequential(nn.Linear(self.flattened_size, 512), nn.ReLU(), nn.Linear(512, n_actions)).to(
                device
            )
            # --- Optimizer único ---
            self.feature_extractor_params = (
                list(self.conv1.parameters()) + list(self.conv2.parameters()) + list(self.conv3.parameters())
            )

            self.critic_optim = optim.Adam(list(self.critic.parameters()) + self.feature_extractor_params, lr=critic_lr)
            self.actor_optim = optim.Adam(list(self.actor.parameters()) + self.feature_extractor_params, lr=actor_lr)

            trunk_params = list(self.conv1.parameters()) + list(self.conv2.parameters()) + list(self.conv3.parameters())
            self.optim = optim.Adam(
                [
                    {"params": trunk_params, "lr": actor_lr},  # feature extractor
                    {"params": self.actor.parameters(), "lr": actor_lr},  # política
                    {"params": self.critic.parameters(), "lr": critic_lr},  # crítico
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
            )

        else:

            self.hidden_size = (64, 64)

            self.critic = nn.Sequential(
                nn.Linear(n_features, self.hidden_size[1]),
                nn.Tanh(),
                nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                nn.Tanh(),
                nn.Linear(self.hidden_size[0], 1),
            ).to(device)

            self.actor = nn.Sequential(
                nn.Linear(n_features, self.hidden_size[1]),
                nn.Tanh(),
                nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                nn.Tanh(),
                nn.Linear(self.hidden_size[0], n_actions),
            ).to(device)

            self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.optim = optim.Adam(
                [
                    {"params": self.actor.parameters(), "lr": actor_lr},
                    {"params": self.critic.parameters(), "lr": critic_lr},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
            )
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_coef = clip_coef

    """
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return self.critic(x), self.actor(x)
    """

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        if self.atari_mode:
            # x esperado: (batch, C, H, W)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.flatten(x)  # flatten mantendo batch
            # x agora shape: (batch, flattened_size)
            critic_out = self.critic(x)
            actor_out = self.actor(x)
        else:
            # x esperado: (batch, n_features)
            critic_out = self.critic(x)
            actor_out = self.actor(x)

        return critic_out, actor_out

    def select_action(self, x):
        value, logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, value, entropy

    def update_parameters(self, critic_loss, actor_loss):
        # self.critic_optim.zero_grad()
        # critic_loss.backward()
        # self.critic_optim.step()

        # self.actor_optim.zero_grad()
        # actor_loss.backward()
        # self.actor_optim.step()
        # self.optim.zero_grad()
        self.optim.zero_grad()

        total_loss = critic_loss + actor_loss
        total_loss.backward()

        # opcional: clipping global
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

        self.optim.step()


class A2CSimple(A2CBase):
    """Simple A2C update (no multiple epochs, no PPO clipping)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_agent(self, rollouts, hp):
        if self.atari_mode:
            T, N, C, H, W = rollouts["states"].shape
            obs_dim = C * H * W
        else:
            T, N, obs_dim = rollouts["states"].shape
        # T, N, _ = rollouts["states"].shape
        device = rollouts["states"].device

        advantages = torch.zeros(T, N, device=device)
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rollouts["rewards"][t]
                + hp["gamma"] * rollouts["value_preds"][t + 1] * rollouts["masks"][t]
                - rollouts["value_preds"][t]
            )
            gae = td_error + hp["gamma"] * hp["lam"] * rollouts["masks"][t] * gae
            advantages[t] = gae

        returns = advantages + rollouts["value_preds"]
        advantages_flat = advantages.reshape(-1)
        old_log_probs_flat = rollouts["old_log_probs"].reshape(-1)
        entropies_flat = rollouts["entropies"].reshape(-1)

        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages_flat.detach() * old_log_probs_flat).mean() - hp["ent_coef"] * entropies_flat.mean()

        self.update_parameters(critic_loss, actor_loss)
        return critic_loss.item(), actor_loss.item(), entropies_flat.mean().item()


class PPO(A2CBase):
    """PPO implementation with multiple epochs and clipping"""

    def __init__(self, *args, ppo_epochs=None, ppo_batch_size=None, clip_coef=None, **kwargs):
        super().__init__(*args, **kwargs)

        # validação obrigatória
        if ppo_epochs is None:
            raise ValueError("ppo_epochs must be provided for PPO")
        if ppo_batch_size is None:
            raise ValueError("ppo_batch_size must be provided for PPO")
        if clip_coef is None:
            raise ValueError("clip_coef must be provided for PPO")

        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_coef = clip_coef

    def update_agent(self, rollouts, hp):
        if self.atari_mode:
            T, N, C, H, W = rollouts["states"].shape
            obs_dim = C * H * W
        else:
            T, N, obs_dim = rollouts["states"].shape
        # T, N, _ = rollouts["states"].shape
        device = rollouts["states"].device

        # Compute advantages
        advantages = torch.zeros(T, N, device=device)
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rollouts["rewards"][t]
                + hp["gamma"] * rollouts["value_preds"][t + 1] * rollouts["masks"][t]
                - rollouts["value_preds"][t]
            )
            gae = td_error + hp["gamma"] * hp["lam"] * rollouts["masks"][t] * gae
            advantages[t] = gae

        returns = advantages + rollouts["value_preds"]

        if self.atari_mode:
            # T, N, C, H, W -> (T*N, C, H, W)
            states_flat = rollouts["states"].reshape(-1, hp["stack_size"], 84, 84)
        else:
            states_flat = rollouts["states"].reshape(-1, obs_dim)
        # states_flat = rollouts["states"].reshape(-1, obs_dim)
        actions_flat = rollouts["actions"].reshape(-1)
        returns_flat = returns.reshape(-1).detach()
        old_log_probs_flat = rollouts["old_log_probs"].reshape(-1).detach()
        advantages_flat = advantages.reshape(-1).detach()

        total_size = states_flat.size(0)
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_epoch = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(total_size)
            for start in range(0, total_size, self.ppo_batch_size):
                idx = perm[start : start + self.ppo_batch_size]
                batch_states = states_flat[idx]
                batch_actions = actions_flat[idx]
                batch_returns = returns_flat[idx]
                batch_adv = advantages_flat[idx]
                batch_old_log_probs = old_log_probs_flat[idx]

                # Forward
                new_values, new_logits = self.forward(batch_states)
                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean() - hp["ent_coef"] * entropy
                critic_loss = (batch_returns - new_values.squeeze(-1)).pow(2).mean()

                ## Critic
                # self.critic_optim.zero_grad()
                # critic_loss.backward()
                # self.critic_optim.step()

                ## Actor
                # self.actor_optim.zero_grad()
                # actor_loss.backward()
                # self.actor_optim.step()

                self.update_parameters(critic_loss, actor_loss)

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += entropy.item()

        num_updates = (total_size // self.ppo_batch_size) * self.ppo_epochs
        return critic_loss_epoch / num_updates, actor_loss_epoch / num_updates, entropy_epoch / num_updates


def A2C(agent_type, **kwargs):
    if agent_type == "a2c":
        return A2CSimple(**kwargs)
    elif agent_type == "ppo":
        return PPO(**kwargs)
    else:
        raise ValueError(f"Unknown agent_type {agent_type}")
