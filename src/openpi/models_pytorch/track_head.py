"""
TrackHead: Predicts mesh point tracks using query points as input.

Uses cross-attention: Q = query_point_embeddings, K/V = prefix_embeds (images + language).
Action head remains completely unchanged.

39 points: 7 agent view + 25 uniform grid eye-in-hand + 7 gripper eye-in-hand
Output: (batch, action_horizon, 39, 3) - 3D coordinates per timestep
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class TrackHead(nn.Module):
    """
    Predicts track trajectories for 39 query points.

    Query points are the initial (t=0) positions of the 39 mesh points.
    The head uses cross-attention to attend to the shared prefix (images + language)
    and predicts where each point moves over the action horizon.

    Architecture:
        query_points [B, 39, 3] -> query_embed [B, 39, D]
        Cross-attention: Q=query_embed, K,V=prefix_embeds
        Output: [B, 39, action_horizon, 3]
    """

    def __init__(
        self,
        prefix_dim: int,
        n_track_points: int = 39,
        action_horizon: int = 50,
        query_point_dim: int = 3,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.n_track_points = n_track_points
        self.action_horizon = action_horizon
        self.prefix_dim = prefix_dim
        self.hidden_dim = hidden_dim

        # Embed query points (x, y, z) to hidden dim
        self.query_embed = nn.Sequential(
            nn.Linear(query_point_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project prefix to hidden_dim for cross-attention (prefix may have different dim)
        self.prefix_proj = nn.Linear(prefix_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    batch_first=True,
                )
            )
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
            )
        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Output: predict (action_horizon, 3) per query point
        self.output_proj = nn.Linear(hidden_dim, action_horizon * 3)

    def forward(
        self,
        prefix_embeds: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prefix_embeds: [B, seq_len, prefix_dim] - concatenated image + language embeddings
            prefix_pad_mask: [B, seq_len] - True where valid, False where padding
            query_points: [B, 39, 3] - initial positions of 39 mesh points (normalized)

        Returns:
            tracks: [B, action_horizon, 39, 3] - predicted 3D positions per timestep
        """
        B = prefix_embeds.shape[0]
        device = prefix_embeds.device

        # Embed query points: [B, 39, hidden_dim]
        query_feat = self.query_embed(query_points)

        # Project prefix: [B, seq_len, hidden_dim]
        prefix_feat = self.prefix_proj(prefix_embeds)

        # For cross-attention: key_padding_mask True = ignore
        # PyTorch MultiheadAttention: key_padding_mask True = mask out
        key_padding_mask = ~prefix_pad_mask  # [B, seq_len], True = padding

        # Cross-attention layers
        for i, (attn, ffn) in enumerate(zip(self.cross_attn_layers, self.ffn_layers)):
            # Cross-attn: Q=query_feat, K,V=prefix_feat
            attn_out, _ = attn(
                query_feat,
                prefix_feat,
                prefix_feat,
                key_padding_mask=key_padding_mask,
            )
            query_feat = self.norm1[i](query_feat + attn_out)
            query_feat = self.norm2[i](query_feat + ffn(query_feat))

        # Output: [B, 39, action_horizon*3] -> [B, 39, action_horizon, 3]
        out = self.output_proj(query_feat)
        out = out.view(B, self.n_track_points, self.action_horizon, 3)
        # [B, 39, action_horizon, 3] -> [B, action_horizon, 39, 3]
        return out.permute(0, 2, 1, 3)
