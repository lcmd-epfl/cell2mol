"""
MetalOxHeteroGNN: one architecture for every variant, wired from the
variant's node types and edge types.

  per-node-type Linear encoder -> num_layers x HeteroConv(GINEConv per edge
  type, sum aggregation) with residual ReLU -> MLP on the metal node ->
  logits over OX_STATES.

Encoders are registered as `<node_type>_encoder` attributes (not a
ModuleDict) so state_dict keys match the checkpoints saved by the original
per-variant modules.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, HeteroConv, Linear

from .constants import OX_STATES
from .variants import EdgeType, Variant, get_variant


def make_gine(hidden_dim: int, edge_dim: int) -> GINEConv:
    return GINEConv(
        nn=nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        ),
        edge_dim=edge_dim,
    )


class MetalOxHeteroGNN(nn.Module):
    def __init__(
        self,
        node_dims: dict[str, int],
        edge_dims: dict[EdgeType, int],
        hidden_dim: int = 128,
        num_classes: int = len(OX_STATES),
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_types = list(node_dims)
        for node_type, dim in node_dims.items():
            setattr(self, f"{node_type}_encoder", Linear(dim, hidden_dim))

        self.convs = nn.ModuleList(
            HeteroConv(
                {
                    edge_type: make_gine(hidden_dim, dim)
                    for edge_type, dim in edge_dims.items()
                },
                aggr="sum",
            )
            for _ in range(num_layers)
        )

        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, num_classes),
        )

    @property
    def node_dims(self) -> dict[str, int]:
        return {
            nt: getattr(self, f"{nt}_encoder").in_channels for nt in self.node_types
        }

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            nt: getattr(self, f"{nt}_encoder")(data[nt].x) for nt in self.node_types
        }

        for conv in self.convs:
            x_new = conv(x_dict, data.edge_index_dict, data.edge_attr_dict)

            for node_type, x_old in x_dict.items():
                if node_type in x_new:
                    x_dict[node_type] = F.relu(x_new[node_type] + x_old)

        return self.classifier(x_dict["metal"])


def build_model(
    variant: Variant | str,
    node_dims: dict[str, int],
    *,
    hidden_dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.2,
) -> MetalOxHeteroGNN:
    if isinstance(variant, str):
        variant = get_variant(variant)

    missing = [nt for nt in variant.node_types if nt not in node_dims]
    if missing:
        raise ValueError(f"node_dims is missing {missing} for variant {variant.name!r}")

    return MetalOxHeteroGNN(
        node_dims={nt: node_dims[nt] for nt in variant.node_types},
        edge_dims=variant.edge_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def checkpoint_metadata(
    model: MetalOxHeteroGNN, variant: Variant, hparams: dict
) -> dict:
    """Everything `load_model` needs besides the weights. `<nt>_dim` keys are
    the format the original modules wrote, kept so old and new checkpoints
    load the same way."""
    return {
        "variant": variant.name,
        **{f"{nt}_dim": dim for nt, dim in model.node_dims.items()},
        **hparams,
    }


def load_model(
    checkpoint_path: str | Path,
    variant: Variant | str | None = None,
    device: torch.device | str = "cpu",
    expected_node_dims: dict[str, int] | None = None,
) -> MetalOxHeteroGNN:
    """
    Rebuild the model a checkpoint was trained with. `variant` may be omitted
    for checkpoints that record it (everything saved by metal_ox); legacy
    checkpoints from the per-variant scripts need it passed explicitly.

    `expected_node_dims` (e.g. a builder's `node_dims`) guards against loading
    a checkpoint whose features no longer match the current builder code.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    variant = variant or checkpoint.get("variant")
    if variant is None:
        raise ValueError(
            f"{checkpoint_path} has no 'variant' recorded; pass variant= explicitly"
        )
    if isinstance(variant, str):
        variant = get_variant(variant)
    if checkpoint.get("variant", variant.name) != variant.name:
        raise ValueError(
            f"{checkpoint_path} was trained as {checkpoint['variant']!r}, not {variant.name!r}"
        )

    node_dims = {
        nt: checkpoint[f"{nt}_dim"]
        for nt in variant.node_types
        if f"{nt}_dim" in checkpoint
    }

    if expected_node_dims is not None:
        expected = {nt: expected_node_dims[nt] for nt in variant.node_types}
        if node_dims != expected:
            raise ValueError(
                f"{checkpoint_path} node dims {node_dims} don't match the current "
                f"{variant.name!r} builder {expected} -- features have changed since it was trained"
            )

    model = build_model(
        variant,
        node_dims,
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_layers=checkpoint.get("num_layers", 4),
        dropout=checkpoint.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model
