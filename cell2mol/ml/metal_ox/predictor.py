"""
Metal oxidation-state probabilities from a trained metal_ox GNN, for use
inside cell2mol (charge balancing, see cell2mol.charge.balancing).

The default checkpoint is cell2mol/models/metal_ox_gnn.pt; the variant and
training details are stored inside it. The default variant uses no charge
features, so its prediction does not depend on the charge assignment it is
asked to rescue. torch and torch_geometric are imported only when a predictor
is loaded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "metal_ox_gnn.pt"


class MetalOxPredictor:
    def __init__(self, checkpoint: str | Path):
        import torch

        from .model import load_model
        from .variants import get_variant, make_builder

        self.checkpoint = Path(checkpoint)
        variant_name = torch.load(
            self.checkpoint, map_location="cpu", weights_only=False
        ).get("variant")
        if variant_name is None:
            raise ValueError(f"{self.checkpoint} records no variant")

        self.variant = get_variant(variant_name)
        self.builder = make_builder(self.variant)
        self.model = load_model(
            self.checkpoint,
            self.variant,
            "cpu",
            expected_node_dims=self.builder.node_dims,
        )

    def probabilities(self, metal) -> dict[int, float] | None:
        """{oxidation state: probability} for `metal`, or None for a metal
        without a resolved coordination sphere (its graph has no edges)."""
        if not metal.coord_nr:
            return None

        from .training import predict_oxidation_state

        return predict_oxidation_state(self.model, self.builder.build(metal), "cpu")[
            "probabilities"
        ]


@lru_cache(maxsize=4)
def load_predictor(checkpoint: str | Path | None = None) -> MetalOxPredictor:
    """Load (once per path) the predictor for `checkpoint`, default DEFAULT_CHECKPOINT."""
    return MetalOxPredictor(checkpoint or DEFAULT_CHECKPOINT)
