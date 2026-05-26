"""
ppo_model.py

PPO Actor-Critic neural network for F1Tenth autonomous racing.

Architecture:
  - Shared LiDAR encoder:  1080 -> 256 -> 128
  - Actor head:             128  ->  64 -> 2   (steering, speed) as Gaussian
  - Critic head:            128  ->  64 -> 1   (state value V(s))

Actions are continuous, parameterised as a diagonal Gaussian.
Outputs are squashed into valid ranges via tanh + affine scaling.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

# ── Action bounds ────────────────────────────────────────────────────────────
STEERING_MAX = 0.4   # radians  ±
SPEED_MIN    = 0.5   # m/s  (keep car always moving forward)
SPEED_MAX    = 3.5   # m/s

LOG_STD_MIN  = -4.0
LOG_STD_MAX  =  0.5


class LidarEncoder(nn.Module):
    """Compress raw 1080-D LiDAR scan to a compact latent vector."""

    def __init__(self, lidar_dim: int = 1080, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lidar_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Shared-encoder PPO Actor-Critic network.

    Actor  → Gaussian mean + log_std for [steering, speed]
    Critic → scalar state value V(s)
    """

    def __init__(self, lidar_dim: int = 1080, latent_dim: int = 128):
        super().__init__()

        self.encoder = LidarEncoder(lidar_dim, latent_dim)

        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
        )
        self.mean_layer    = nn.Linear(64, 2)
        self.log_std_layer = nn.Linear(64, 2)

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init — standard practice for PPO."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Policy output layer uses smaller gain
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)

    # ── Forward helpers ──────────────────────────────────────────────────────

    def encode(self, lidar: torch.Tensor) -> torch.Tensor:
        """Normalise scan values and encode."""
        lidar = torch.clamp(lidar, 0.0, 30.0) / 30.0   # range-normalise
        return self.encoder(lidar)

    def get_value(self, lidar: torch.Tensor) -> torch.Tensor:
        latent = self.encode(lidar)
        return self.critic_head(latent)

    def get_action_and_value(
        self,
        lidar: torch.Tensor,
        action: torch.Tensor = None,
    ):
        """
        Returns:
            action      – sampled or supplied action, shape (B, 2)
            log_prob    – sum log-prob of action, shape (B,)
            entropy     – policy entropy, shape (B,)
            value       – critic estimate, shape (B, 1)
            raw_action  – un-squashed Gaussian sample (for storage)
        """
        latent     = self.encode(lidar)
        actor_feat = self.actor_head(latent)

        mean    = self.mean_layer(actor_feat)
        log_std = self.log_std_layer(actor_feat)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        dist = Normal(mean, std)

        if action is None:
            raw_action = dist.rsample()          # reparameterised sample
        else:
            raw_action = action                  # use stored raw action

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic_head(latent)

        # Squash to valid ranges
        scaled_action = self._squash(raw_action)

        return scaled_action, log_prob, entropy, value, raw_action

    def _squash(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Map unbounded Gaussian sample → valid [steering, speed].
          steering: tanh(raw[0]) * STEERING_MAX
          speed:    sigmoid(raw[1]) * (SPEED_MAX - SPEED_MIN) + SPEED_MIN
        """
        steering = torch.tanh(raw[..., 0:1]) * STEERING_MAX
        speed    = torch.sigmoid(raw[..., 1:2]) * (SPEED_MAX - SPEED_MIN) + SPEED_MIN
        return torch.cat([steering, speed], dim=-1)

    # ── Convenience method for inference ────────────────────────────────────

    @torch.no_grad()
    def predict(self, lidar_np: np.ndarray, deterministic: bool = False):
        """
        

        Returns:
            steering (float), speed (float)
        """
        device = next(self.parameters()).device
        lidar_t = torch.FloatTensor(lidar_np).unsqueeze(0).to(device)

        latent     = self.encode(lidar_t)
        actor_feat = self.actor_head(latent)
        mean       = self.mean_layer(actor_feat)
        log_std    = self.log_std_layer(actor_feat)
        log_std    = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        if deterministic:
            raw = mean
        else:
            raw = Normal(mean, log_std.exp()).rsample()

        # action = self._squash(raw).squeeze(0).numpy()
        action = self._squash(raw).squeeze(0).cpu().numpy()

        return float(action[0]), float(action[1])

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"[ActorCritic] Model saved → {path}")

    def load(self, path: str, device: str = "cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"[ActorCritic] Model loaded ← {path}")
