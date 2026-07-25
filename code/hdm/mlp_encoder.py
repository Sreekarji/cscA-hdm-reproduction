"""
MLPEncoder: flat per-task encoder for HAN ablation (Fig 13a).
Drop-in replacement for HANNetwork — identical encode_state() interface.
No graph attention; each task encoded independently from its own features.
"""
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Flat 3-layer MLP encoder.
    Replaces HANNetwork in HANMLPTrainer.
    Returns same shapes: (graph_emb [1,256], None, message_embs [n_tasks,256]).
    """

    def __init__(
        self,
        task_input_dim: int = 6,
        csca_input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()
        self.task_input_dim = task_input_dim

        self.task_net = nn.Sequential(
            nn.Linear(task_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.csca_net = nn.Sequential(
            nn.Linear(csca_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.graph_net = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_state(self, state: dict, intent_vectors=None):
        device = next(self.parameters()).device

        msg_feats = torch.tensor(
            state["SCt"]["message_features"], dtype=torch.float, device=device
        )

        if intent_vectors is not None:
            iv = torch.tensor(intent_vectors, dtype=torch.float, device=device)
            task_inp = torch.cat([msg_feats, iv], dim=-1)
        else:
            task_inp = torch.cat([
                msg_feats,
                torch.zeros(msg_feats.shape[0], 2, device=device)
            ], dim=-1)

        csca_feats = torch.tensor(
            state["Rt"]["csca_features"], dtype=torch.float, device=device
        )

        message_embs = self.task_net(task_inp)
        csca_embs = self.csca_net(csca_feats)
        csca_mean = csca_embs.mean(dim=0, keepdim=True)
        task_mean = message_embs.mean(dim=0, keepdim=True)
        graph_emb = self.graph_net(
            torch.cat([csca_mean, task_mean], dim=-1)
        )

        return graph_emb, None, message_embs
