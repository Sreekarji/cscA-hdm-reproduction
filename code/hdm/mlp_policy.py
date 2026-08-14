"""
MLP Actor-Critic policy using HAN graph embeddings.
Replaces DDPM with a simple 3-layer MLP for stable training.
This isolates HAN's contribution from DDPM training complexity.

Architecture:
- State: HAN graph embedding GL_t [256-dim] + per-task message embeddings [n_tasks x 256-dim]
- Actor: MLP(GL_t concat mean(message_embs)) -> action [action_dim]
- Critic: MLP(GL_t concat action) -> value [1-dim]

action_dim is supplied by the caller (train_han_mlp.compute_action_dim) and is
n_tasks*(1+n_relays+n_mcs) with the relay head, n_tasks*(1+n_mcs) without.
The global head width is derived as action_dim - n_tasks either way.
"""

import torch
import torch.nn as nn
import numpy as np


class MLPActor(nn.Module):
    """
    Simple MLP actor replacing DDPM.
    Takes graph embedding + task embeddings -> communication policy.
    """
    def __init__(
        self,
        graph_emb_dim: int = 256,
        task_emb_dim: int = 256,
        action_dim: int = 45,
        hidden_dim: int = 256,
        n_tasks: int = 5,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.action_dim = action_dim

        # Task-specific BW head
        # Input: per-task embedding [256] -> BW fraction [1]
        self.task_bw_head = nn.Sequential(
            nn.Linear(task_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Global relay + MCS head
        # Input: graph embedding [256] -> relay+MCS actions
        relay_mcs_dim = action_dim - n_tasks
        self.global_head = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, relay_mcs_dim),
        )

        # Softmax temperature for BW allocation
        self.bw_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, graph_emb: torch.Tensor,
                message_embs: torch.Tensor = None):
        """
        Args:
            graph_emb: [batch, 256] graph embedding from HAN
            message_embs: [n_tasks, 256] per-task embeddings from HAN
        Returns:
            action: [batch, action_dim] communication policy
        """
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)

        # Task-specific BW allocation using per-task embeddings
        if message_embs is not None:
            # [n_tasks, 1] -> softmax -> [1, n_tasks]
            bw_logits = self.task_bw_head(message_embs)  # [n_tasks, 1]
            bw_alloc = torch.softmax(
                bw_logits.squeeze(-1) / self.bw_temperature.abs(), dim=0
            ).unsqueeze(0)  # [1, n_tasks]
        else:
            # Uniform if no message embeddings
            bw_alloc = torch.ones(
                graph_emb.shape[0], self.n_tasks, device=graph_emb.device
            ) / self.n_tasks

        # Global relay + MCS
        relay_mcs = torch.sigmoid(self.global_head(graph_emb))

        # Combine
        action = torch.cat([bw_alloc, relay_mcs], dim=-1)
        return action

    def parse_action(self, action, n_tasks=5, n_relays=5, n_mcs=3):
        # Layout is inferred from the actual width, not assumed, so this works
        # for both action spaces (see train_han_mlp.USE_RELAY_ACTION):
        #   with relay   : n_tasks * (1 + n_relays + n_mcs)
        #   without relay: n_tasks * (1 + n_mcs)
        bw = action[:, :n_tasks]
        has_relay = action.shape[-1] >= n_tasks * (1 + n_relays + n_mcs)
        if has_relay:
            relay = action[:, n_tasks:n_tasks + n_tasks * n_relays].reshape(
                action.shape[0], n_tasks, n_relays
            )
            mcs = action[:, n_tasks + n_tasks * n_relays:].reshape(
                action.shape[0], n_tasks, n_mcs
            )
        else:
            relay = torch.zeros(action.shape[0], n_tasks, n_relays,
                                device=action.device, dtype=action.dtype)
            mcs = action[:, n_tasks:].reshape(action.shape[0], n_tasks, n_mcs)
        return {"bandwidth": bw, "relay": relay, "mcs": mcs}


class MLPCritic(nn.Module):
    def __init__(self, state_dim: int = 256, action_dim: int = 45,
                 hidden_dim: int = None):
        super().__init__()
        input_dim = state_dim + action_dim
        if hidden_dim is None:
            hidden_dim = min(512, max(256, input_dim // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: torch.Tensor, action: torch.Tensor):
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if graph_emb.shape[0] != action.shape[0]:
            if graph_emb.shape[0] == 1:
                graph_emb = graph_emb.expand(action.shape[0], -1)
            elif action.shape[0] == 1:
                action = action.expand(graph_emb.shape[0], -1)
        return self.net(torch.cat([graph_emb, action], dim=-1))


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks, n_relays, n_mcs = 5, 5, 3
    for use_relay in (False, True):
        action_dim = n_tasks * (1 + (n_relays if use_relay else 0) + n_mcs)
        actor  = MLPActor(action_dim=action_dim, n_tasks=n_tasks).to(DEVICE)
        critic = MLPCritic(action_dim=action_dim).to(DEVICE)

        graph_emb = torch.randn(1, 256, device=DEVICE)
        msg_embs  = torch.randn(n_tasks, 256, device=DEVICE)

        action = actor(graph_emb, message_embs=msg_embs)
        value  = critic(graph_emb, action)
        parsed = actor.parse_action(action, n_tasks, n_relays, n_mcs)

        assert action.shape == (1, action_dim), action.shape
        assert parsed["mcs"].shape == (1, n_tasks, n_mcs), parsed["mcs"].shape
        assert parsed["relay"].shape == (1, n_tasks, n_relays), parsed["relay"].shape
        print(f"use_relay={use_relay}  action_dim={action_dim}  "
              f"BW sum={action[0, :n_tasks].sum().item():.4f}  "
              f"V={value.item():.4f}  PASSED")
