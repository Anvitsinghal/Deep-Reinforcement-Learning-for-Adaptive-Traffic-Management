"""
neural_world_model.py
======================
Deep Neural Traffic Dynamics Model (World Model) for Indian Urban Corridors.
Trained on real-world Indian traffic datasets:
  - Delhi Traffic Density Dataset (IIT Delhi)
  - Pune Heterogeneous Traffic Dataset (Mendeley xnf2k6n288/1)

This neural simulator acts as a 100% data-driven surrogate for SUMO:
Given current corridor state s_t and multi-agent signal action a_t,
it predicts:
  1. Next corridor state s_{t+1} (queues, waiting times, phase states)
  2. Immediate cooperative reward r_t
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

STATE_DIM = 66    # 3 intersections * 22 local obs features
N_ACTIONS = 3     # Discrete(3) per intersection
N_AGENTS = 3      # 3 intersections


class ResidualBlock(nn.Module):
    """Residual feed-forward block with LayerNorm and SiLU activation."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        out = F.silu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return F.silu(out + residual)


class TrafficDynamicsWorldModel(nn.Module):
    """
    Neural World Model predicting transition dynamics:
      (s_t, a_t) -> (s_{t+1}, r_t)
    Predicts residual change \Delta s = s_{t+1} - s_t for numerical stability.
    """
    def __init__(self, state_dim=STATE_DIM, n_agents=N_AGENTS, n_actions=N_ACTIONS, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_dim = n_agents * n_actions  # One-hot encoded action vector (9 dims)

        # Input projection
        self.in_proj = nn.Linear(state_dim + self.action_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)

        # Residual backbone
        self.block1 = ResidualBlock(hidden_dim)
        self.block2 = ResidualBlock(hidden_dim)
        self.block3 = ResidualBlock(hidden_dim)

        # Output heads
        self.delta_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        """
        state:  Tensor of shape (B, state_dim)
        action: LongTensor of shape (B, n_agents) or float one-hot (B, action_dim)
        """
        if action.dtype == torch.int64 or action.dtype == torch.int32:
            # Convert multi-agent discrete actions to concatenated one-hot
            action_onehot = F.one_hot(action.long(), num_classes=self.n_actions).float()
            action_feat = action_onehot.view(action.shape[0], -1)
        else:
            action_feat = action

        x = torch.cat([state, action_feat], dim=-1)
        feat = F.silu(self.ln_in(self.in_proj(x)))
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)

        delta_state = self.delta_state_head(feat)
        # Bounded state prediction: next_state in [0, 1]
        next_state = torch.clamp(state + delta_state, 0.0, 1.0)
        reward = self.reward_head(feat).squeeze(-1)

        return next_state, reward

    def predict_step(self, state_np, action_np, device="cpu"):
        """
        Convenience inference step for environment usage.
        state_np: np.ndarray shape (66,)
        action_np: np.ndarray shape (3,)
        """
        self.eval()
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            a = torch.tensor(action_np, dtype=torch.int64, device=device).unsqueeze(0)
            next_s, rew = self.forward(s, a)
            return next_s.cpu().squeeze(0).numpy(), rew.cpu().item()

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()
