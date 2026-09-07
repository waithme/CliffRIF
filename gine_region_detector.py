#!/usr/bin/env python3
"""Three-layer GINE with final-layer-only region supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


@dataclass
class MultiScaleRegionConfig:
    atom_in_dim: int
    bond_in_dim: int
    num_tasks: int
    hidden_dim: int = 300
    num_layers: int = 3
    dropout: float = 0.1


class Head(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GINEBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv = GINEConv(mlp, edge_dim=hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        update = self.conv(x, edge_index, edge_attr)
        return x + self.dropout(F.relu(self.norm(update)))


class GINEMultiScaleRegionDetector(nn.Module):
    """GINE3 whose region mask and pooling use only the final GINE layer."""

    def __init__(self, cfg: MultiScaleRegionConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.atom_encoder = nn.Linear(cfg.atom_in_dim, h)
        self.bond_encoder = nn.Linear(cfg.bond_in_dim, h)
        self.layers = nn.ModuleList([GINEBlock(h, cfg.dropout) for _ in range(cfg.num_layers)])
        self.mask_heads = nn.ModuleList([Head(h, 1, cfg.dropout)])
        self.region_fusion = Head(h, h, cfg.dropout)
        self.region_existence_head = Head(2 * h, 1, cfg.dropout)
        # Activity prediction uses only the final-layer whole-molecule
        # mean representation. Region and existence predictions are
        # auxiliary objectives and do not enter the activity head.
        self.y_head = nn.Sequential(
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, cfg.num_tasks),
        )

    def forward(self, batch) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        x = F.relu(self.atom_encoder(batch.x.float()))
        edge_attr = F.relu(self.bond_encoder(batch.edge_attr.float()))
        layer_h = []
        for layer in self.layers:
            x = layer(x, batch.edge_index.long(), edge_attr)
            layer_h.append(x)

        final_h = layer_h[-1]
        final_mask_logit = self.mask_heads[0](final_h).squeeze(-1)
        mask_logits = [final_mask_logit]
        probability = torch.sigmoid(final_mask_logit).unsqueeze(-1)
        numerator = global_mean_pool(final_h * probability, batch.batch.long())
        denominator = global_mean_pool(probability, batch.batch.long()).clamp_min(1e-6)
        final_region_summary = numerator / denominator

        graph_h = global_mean_pool(final_h, batch.batch.long())
        raw_region_h = self.region_fusion(final_region_summary)
        region_existence_logit = self.region_existence_head(
            torch.cat([graph_h, raw_region_h], dim=-1)
        ).squeeze(-1)
        all_y = self.y_head(graph_h)
        task_id = batch.task_id.long().view(-1)
        y_pred = all_y.gather(1, task_id.unsqueeze(1)).squeeze(1)
        return {
            "mask_logits": mask_logits,
            "region_existence_logit": region_existence_logit,
            "y_pred": y_pred,
        }


def dice_loss(logits, targets, eps: float = 1e-6):
    probability = torch.sigmoid(logits)
    intersection = (probability * targets).sum()
    return 1.0 - (2.0 * intersection + eps) / (probability.sum() + targets.sum() + eps)
