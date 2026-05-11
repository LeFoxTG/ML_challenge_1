import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from model import AtariActorCritic


class PPOAgent:
    def __init__(self, act_dim, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hiperparámetros Grupo 2
        self.horizon = 1024
        self.n_epochs = 6
        self.batch_size = 128
        self.gamma = 0.995
        self.gae_lambda = 0.97
        self.clip_eps = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5

        self.network = AtariActorCritic(act_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)

        self.reset_buffer()

    def reset_buffer(self):
        self.obs, self.actions, self.logp, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def preprocess(self, obs):
        obs = np.array(obs, copy=False)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if obs.ndim != 3:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")

        # Caso 1: (H, W, C)
        if obs.shape[-1] == 4:
            obs = obs.permute(2, 0, 1)

        # Caso 2: (C, H, W) → ya está bien
        elif obs.shape[0] == 4:
            pass

        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")

        return obs

    def select_action(self, obs):
        obs = self.preprocess(obs).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.network(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()

        return action.item(), dist.log_prob(action).item(), value.item()

    def store(self, obs, action, logp, reward, done, value):
        self.obs.append(self.preprocess(obs))
        self.actions.append(action)
        self.logp.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, next_value):
        adv = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            adv.insert(0, gae)
            next_value = self.values[t]

        returns = [a + v for a, v in zip(adv, self.values)]
        return torch.tensor(adv), torch.tensor(returns)

    def update(self, next_obs):
        next_obs = self.preprocess(next_obs).unsqueeze(0)

        with torch.no_grad():
            _, next_value = self.network(next_obs)

        adv, returns = self.compute_gae(next_value.item())

        obs = torch.stack(self.obs)
        actions = torch.tensor(self.actions, device=self.device)
        old_logp = torch.tensor(self.logp, device=self.device)
        adv = adv.to(self.device)
        returns = returns.to(self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.n_epochs):
            idx = torch.randperm(len(obs))

            for start in range(0, len(obs), self.batch_size):
                mb = idx[start:start+self.batch_size]

                logits, values = self.network(obs[mb])
                dist = Categorical(logits=logits)

                new_logp = dist.log_prob(actions[mb])
                ratio = (new_logp - old_logp[mb]).exp()

                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv[mb]

                loss_pi = -torch.min(surr1, surr2).mean()
                loss_v = ((values - returns[mb])**2).mean()
                entropy = dist.entropy().mean()

                loss = loss_pi + self.vf_coef*loss_v - self.ent_coef*entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

        self.reset_buffer()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, obs):
        obs = self.preprocess(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.network(obs)
        return logits.argmax(dim=-1).item()