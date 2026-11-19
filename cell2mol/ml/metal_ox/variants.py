"""
Registry of every model variant: which graph builder + feature flags it uses,
and which node/edge types its MetalOxHeteroGNN is wired with.

This module is deliberately cell2mol-free (builders are imported lazily in
`make_builder`), so training from a pre-built bundle only needs torch +
torch_geometric -- e.g. on a separate GPU cluster.

Two families:
  - "simple": metal-centered graph of the metal + its coord_sphere_atoms,
    with feature groups switched on one at a time (see SimpleGraphBuilder).
  - "full":   the original metal + atom + haptic-group + ligand graph (see
    FullGraphBuilder); its variants only toggle which node types get rows,
    so all four share one fixed architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import ATOM_TYPES, METAL_TYPES

EdgeType = tuple[str, str, str]

# Edge lists keep the exact insertion order of the original per-variant
# modules: HeteroConv sums per-dst messages in dict order, so reordering would
# perturb float results of the existing checkpoints.
_COORD_EDGES: list[tuple[EdgeType, int]] = [
    (("metal", "coordinates", "atom"), 4),
    (("atom", "rev_coordinates", "metal"), 4),
]

_SIMPLE_LIGAND_EDGES: list[tuple[EdgeType, int]] = [
    (("metal", "binds", "ligand"), 4),
    (("ligand", "rev_binds", "metal"), 4),
    (("ligand", "ligand_contains", "atom"), 1),
    (("atom", "rev_ligand_contains", "ligand"), 1),
]

_SIMPLE_HAPTIC_EDGES: list[tuple[EdgeType, int]] = [
    (("metal", "haptic_to", "haptic"), 4),
    (("haptic", "rev_haptic_to", "metal"), 4),
    (("haptic", "contains", "atom"), 1),
    (("atom", "rev_contains", "haptic"), 1),
]

_FULL_EDGES: list[tuple[EdgeType, int]] = [
    (("metal", "coordinates", "atom"), 4),
    (("atom", "rev_coordinates", "metal"), 4),
    (("metal", "haptic_to", "haptic"), 4),
    (("haptic", "rev_haptic_to", "metal"), 4),
    (("haptic", "contains", "atom"), 1),
    (("atom", "rev_contains", "haptic"), 1),
    (("metal", "binds", "ligand"), 1),
    (("ligand", "rev_binds", "metal"), 1),
    (("ligand", "ligand_contains", "atom"), 1),
    (("atom", "rev_ligand_contains", "ligand"), 1),
]


@dataclass(frozen=True)
class Variant:
    name: str
    family: str  # "simple" | "full"
    description: str
    node_types: tuple[str, ...]
    edges: tuple[tuple[EdgeType, int], ...]
    builder_kwargs: dict[str, Any] = field(default_factory=dict)
    # Feature columns, per node type, that a failed cell2mol charge assignment
    # leaves at 0.0 (charges and their has_* flags, plus anything summed from
    # them). Used by charge dropout in training. Must match the feature order
    # in builders.py.
    charge_columns: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def edge_dims(self) -> dict[EdgeType, int]:
        return dict(self.edges)


def _simple(
    name: str,
    description: str,
    *,
    coord_modes: bool = False,
    ligand_node: bool = False,
    ligand_charge: bool = True,
    **flags,
) -> Variant:
    charge_columns = {}
    if flags.get("atom_charge"):
        # atom: one-hot + [Z, period, EN, mconnec, other connec] + charge [+ has_charge]
        first = len(ATOM_TYPES) + 5
        charge_columns["atom"] = (first, first + 1) if flags.get("atom_charge_flag") else (first,)
    if ligand_node and ligand_charge:
        charge_columns["ligand"] = (0, 1)  # totcharge, has_totcharge

    node_types = ["atom", "metal"]
    edges = list(_COORD_EDGES)
    if ligand_node:
        node_types.append("ligand")
        edges += _SIMPLE_LIGAND_EDGES
    if coord_modes:
        node_types.append("haptic")
        edges += _SIMPLE_HAPTIC_EDGES

    return Variant(
        name=name,
        family="simple",
        description=description,
        node_types=tuple(node_types),
        edges=tuple(edges),
        builder_kwargs=dict(coord_modes=coord_modes, ligand_node=ligand_node, ligand_charge=ligand_charge, **flags),
        charge_columns=charge_columns,
    )


def _full(name: str, description: str, *, include_haptic: bool, include_ligand: bool) -> Variant:
    m = len(METAL_TYPES)
    return Variant(
        name=name,
        family="full",
        description=description,
        node_types=("atom", "metal", "haptic", "ligand"),
        edges=tuple(_FULL_EDGES),
        builder_kwargs=dict(include_haptic=include_haptic, include_ligand=include_ligand),
        charge_columns={
            "atom": (len(ATOM_TYPES) + 3,),  # one-hot + [Z, period, radius] + charge
            "metal": (m + 6, m + 8),  # total_ligand_charge, coord_atom_charge_sum
            "ligand": (0,),  # totcharge
        },
    )


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        _simple(
            "simple",
            "metal + atom nodes; element/connectivity features only",
        ),
        _simple(
            "simple_charge",
            "+ atom formal charge (with has_charge flag)",
            atom_charge=True,
            atom_charge_flag=True,
        ),
        _simple(
            "simple_charge_coord_geom",
            "+ metal coordination geometry (one-hot, coord_nr, geom_deviation, rel_metal_radius, with has_* flags)",
            atom_charge=True,
            atom_charge_flag=True,
            coord_geom=True,
            geom_quality=True,
        ),
        _simple(
            "simple_charge_coord_geom_ligand",
            "simple_charge_coord_geom + per-metal coordination modes (kappa donors, eta-site nodes) + ligand nodes (totcharge, kappa/eta counts, mu, mixed)",
            atom_charge=True,
            atom_charge_flag=True,
            coord_geom=True,
            geom_quality=True,
            coord_modes=True,
            ligand_node=True,
        ),
        _simple(
            "simple_coord_geom_modes",
            "simple + coordination geometry + per-metal coordination modes + ligand nodes without charge -- all structural features, nothing from charge assignment",
            coord_geom=True,
            geom_quality=True,
            coord_modes=True,
            ligand_node=True,
            ligand_charge=False,
        ),
        _full(
            "full",
            "original model: atom + metal + haptic + ligand nodes",
            include_haptic=True,
            include_ligand=True,
        ),
        _full(
            "atom_metal",
            "original model, atom + metal nodes only",
            include_haptic=False,
            include_ligand=False,
        ),
        _full(
            "atom_metal_haptic",
            "original model, atom + metal + haptic nodes",
            include_haptic=True,
            include_ligand=False,
        ),
        _full(
            "atom_metal_ligand",
            "original model, atom + metal + ligand nodes",
            include_haptic=False,
            include_ligand=True,
        ),
    ]
}


def get_variant(name: str) -> Variant:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; choose from {list(VARIANTS)}")
    return VARIANTS[name]


def make_builder(variant: Variant | str):
    """Instantiate the graph builder for `variant` (imports cell2mol)."""
    if isinstance(variant, str):
        variant = get_variant(variant)

    from .builders import FullGraphBuilder, SimpleGraphBuilder, default_element_database

    edb = default_element_database()

    if variant.family == "simple":
        return SimpleGraphBuilder(edb, **variant.builder_kwargs)
    return FullGraphBuilder(edb, **variant.builder_kwargs)
