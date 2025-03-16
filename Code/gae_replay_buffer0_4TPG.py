import numpy as np
import torch

class GAEReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, gamma=0.99, lam=0.95, device="cpu"):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.device = device

        # allocate space
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0  # current index
        self.size = 0  # current data in buffer

    def store(self, state, action, reward, value, log_prob, done):
        """儲存一筆經驗"""
        if self.ptr >= self.capacity:
            return
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(self, last_value=0):
        """compute Generalized Advantage Estimation (GAE)"""
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.rewards)

        gae = 0
        for t in reversed(range(self.size)):
            delta = self.rewards[t] + self.gamma * (1 - self.dones[t]) * last_value - self.values[t]
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            last_value = self.values[t]

        # compute advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns

    def sample(self, batch_size):
        """sample data"""
        idxs = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            torch.tensor(self.states[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.returns[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.advantages[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.log_probs[idxs], dtype=torch.float32, device=self.device),
        )

    def clear(self):
        """clear buffer"""
        self.ptr = 0
        self.size = 0
