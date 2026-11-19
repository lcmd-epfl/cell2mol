"""
Graph builders: turn one cell2mol Metal into a metal-centered HeteroData.

SimpleGraphBuilder covers the "simple" family. Its feature groups are
switched on by flags (see variants.py for which flags each variant sets):

  atom_charge       atom.charge as an atom feature, 0.0-imputed when None.
  atom_charge_flag  ...plus a has_charge indicator. A bare 0.0 for a missing
                    charge is indistinguishable from a genuinely neutral atom
                    (very common for C/H), so the flag lets the model discount
                    imputed values. Missing charges are common on exactly the
                    refcodes where cell2mol's charge assignment failed.
  coord_geom        metal coord_geometry one-hot + coord_nr. coord_nr needs no
                    flag: 0 can't collide with a real CN once uncoordinated
                    metals are filtered out (`filter_coordinated_metals`), and
                    "Undefined" is its own vocabulary bucket.
  geom_quality      geom_deviation and rel_metal_radius, each with a has_*
                    flag (both are centered near 0, so the same imputation
                    argument as atom.charge applies). Requires coord_geom.
  coord_modes       build the coordination from cell2mol's per-metal
                    `ligand.coordination` (kappa donors + eta sites) instead of
                    coord_sphere_atoms: kappa donors join the metal directly
                    ("coordinates"); each eta site becomes a "haptic" node
                    ("haptic_to" / "contains") encoded by structure -- site
                    size, element counts, ring topology -- so every haptic
                    type cell2mol can name (eta5(Cp), eta6(benzene),
                    eta5(P5)-ring, any etaN(formula)) is represented without
                    an "other" bucket. Metal-metal contacts are not atom
                    nodes; their count is a metal feature, next to the number
                    of coordinating ligand atoms (the site count is coord_nr).
                    Needs the post-pydantic cell2mol API.
  ligand_node       one node per coordinating ligand, joined to the metal
                    ("binds") and to its donor atoms ("ligand_contains"):
                    per-metal kappa/eta counts, mu (#metals it binds),
                    mixed (kappa to one metal, eta to another), plus
                    totcharge + has flag when ligand_charge. Requires
                    coord_modes. No ligand.denticity: cell2mol sets it to None
                    for ligands bound to more than one metal.

Append new features at the end of a vector; charge-derived columns are
listed in variants.py (`charge_columns`) for charge dropout.

FullGraphBuilder is the original (non-simplified) model's builder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from cell2mol.coordination_sphere import shape_structure_references_simplified
from cell2mol.elementdata import ElementData
from cell2mol.operations import get_dist

from .compat import contains, labels2formula, rel_metal_radius
from .constants import ATOM_TYPES, METAL_TYPES, OX_TO_CLASS

# =============================================================================
# Vocabularies
# =============================================================================

# Imported from cell2mol rather than hand-copied so the one-hot can't drift
# out of sync with the labels cell2mol actually assigns.
COORD_GEOMETRY_TYPES = sorted(
    {
        entry[3]
        for entries in shape_structure_references_simplified.values()
        for entry in entries
    }
) + ["Undefined"]

# Topology flags stored by cell2mol on each haptic Group (from
# connectivity.analyze_topology); together with site size and composition
# they determine the haptic type name identify_haptic_mode assigns.
TOPOLOGY_FLAGS = [
    "is_connected",
    "all_atoms_in_rings",
    "is_single_simple_ring",
    "is_fused_5_5",
    "is_fused_5_6",
    "is_fused_5_7",
]

FULL_HAPTIC_TYPES = [
    "eta2(C2)",
    "eta3(C3)",
    "eta4(C4)",
    "eta4(C3O)",
    "eta5(C5)",
    "eta5(P5)",
    "eta5(As5)",
    "eta6(C6)",
    "eta7(C7)",
    "eta8(C8)",
    "eta5(Cp)",
    "eta6(benzene)",
    "eta7(CHT)",
    "eta8(COT)",
    "eta5,eta5(pentalene)",
    "eta5,eta6(indene)",
    "eta5,eta7(azulene)",
    "OTHER",
]

FULL_COORD_GEOMETRIES = [
    "Linear",
    "Bent (V-shape, 109.47°)",
    "Bent (L-shape, 90°)",
    "Trigonal planar",
    "T-shaped",
    "fac-Trivacant octahedron",
    "Tetrahedral",
    "Square planar",
    "Seesaw",
    "Trigonal bipyramidal",
    "Square pyramidal",
    "Pentagonal pyramidal",
    "Octahedral",
    "Trigonal prismatic",
    "Pentagonal bipyramidal",
    "Capped trigonal prismatic",
    "Hexagonal bipyramidal",
    "Cube",
    "Square antiprismatic",
    "Dodecahedral",
    "Tricapped trigonal prism J51",
    "Undefined",
    "OTHER",
]

# =============================================================================
# Small utilities
# =============================================================================


@dataclass
class ElementDatabaseAdapter:
    elementnr: dict[str, float]
    elementperiod: dict[str, float]
    CovalentRadius3: dict[str, float]
    ElectroNegativityPauling: dict[str, float]


def default_element_database() -> ElementDatabaseAdapter:
    elemdatabase = ElementData()
    return ElementDatabaseAdapter(
        elementnr=elemdatabase.elementnr,
        elementperiod=elemdatabase.elementperiod,
        CovalentRadius3=elemdatabase.CovalentRadius3,
        ElectroNegativityPauling=elemdatabase.ElectroNegativityPauling,
    )


def one_hot(value: str, vocab: list[str]) -> list[float]:
    x = [0.0] * len(vocab)
    if value in vocab:
        x[vocab.index(value)] = 1.0
    return x


def safe_get(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default))
    except AttributeError:
        return float(mapping[key]) if key in mapping else float(default)


def safe_float_attr(obj: Any, name: str, default: float = 0.0) -> float:
    value = getattr(obj, name, default)
    return float(default if value is None else value)


def value_with_flag(value: Any) -> tuple[float, float]:
    """(value, 1.0), or (0.0, 0.0) when `value` is None."""
    if value is None:
        return 0.0, 0.0
    return float(value), 1.0


def binding_ligands(metal: Any) -> list[Any]:
    """Ligands of `metal`'s parent Molecule that bind `metal` (needs the
    parent link, which metals loaded from a cell2mol Reference have)."""
    if not metal.check_parent("molecule"):
        return []
    mol = metal.get_parent("molecule")
    return [lig for lig in (mol.ligands or []) if contains(lig.metals, metal)]


def metal_atom_edge_feature(
    edb: ElementDatabaseAdapter, metal: Any, atom: Any
) -> torch.Tensor:
    d = get_dist(metal.coord, atom.coord)

    r_m = safe_get(edb.CovalentRadius3, metal.label)
    r_a = safe_get(edb.CovalentRadius3, atom.label)

    return torch.tensor(
        [
            d,
            d - r_a,
            d - r_m - r_a,
            (d - r_a) / r_m if r_m > 0 else 0.0,
        ],
        dtype=torch.float32,
    )


def metal_group_edge_feature(
    edb: ElementDatabaseAdapter, metal: Any, atoms: list[Any]
) -> torch.Tensor:
    """(mean dist, std dist, mean dist-minus-covalent-radius, atom count)
    between `metal` and a group of atoms (a haptic group, or one ligand's
    donor atoms)."""
    distances = []
    diffs = []

    for atom in atoms:
        d = get_dist(metal.coord, atom.coord)
        distances.append(d)
        diffs.append(d - safe_get(edb.CovalentRadius3, atom.label))

    distances = np.asarray(distances, dtype=float)
    diffs = np.asarray(diffs, dtype=float)

    return torch.tensor(
        [
            float(np.mean(distances)) if len(distances) else 0.0,
            float(np.std(distances)) if len(distances) > 1 else 0.0,
            float(np.mean(diffs)) if len(diffs) else 0.0,
            float(len(atoms)),
        ],
        dtype=torch.float32,
    )


def stack_or_empty(rows: list[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.stack(rows) if rows else torch.empty((0, dim), dtype=torch.float32)


def add_edges(
    data: HeteroData,
    edge_type: tuple[str, str, str],
    pairs: list[list[int]],
    attrs: list[torch.Tensor] | None = None,
) -> None:
    """Add `edge_type` and its reverse ("rev_" + relation). attrs=None means a
    constant 1-dim edge feature."""
    if not pairs:
        return

    src, rel, dst = edge_type
    edge_index = torch.tensor(pairs, dtype=torch.long).T
    edge_attr = (
        torch.stack(attrs)
        if attrs is not None
        else torch.ones((edge_index.shape[1], 1), dtype=torch.float32)
    )

    data[src, rel, dst].edge_index = edge_index
    data[src, rel, dst].edge_attr = edge_attr
    data[dst, f"rev_{rel}", src].edge_index = edge_index.flip(0)
    data[dst, f"rev_{rel}", src].edge_attr = edge_attr


def set_label(data: HeteroData, oxidation_state: int | None) -> None:
    if oxidation_state is None:
        return
    if oxidation_state not in OX_TO_CLASS:
        raise ValueError(f"Unsupported oxidation state: {oxidation_state}")
    data["metal"].y = torch.tensor([OX_TO_CLASS[oxidation_state]], dtype=torch.long)


# =============================================================================
# Simple family
# =============================================================================


def coordination_modes(metal) -> list[tuple[Any, list[Any], list[list[Any]], dict]]:
    """
    (ligand, kappa_atoms, eta_sites, ligand_coordination) for every ligand of
    `metal`'s Molecule that binds it, from cell2mol's `ligand.coordination`
    ({molecule metal index: {"kappa_atoms", "eta_groups"}}, ligand-local
    indices). ligand_coordination covers all metals (for mu / mixed).
    """
    if not metal.check_parent("molecule"):
        return []
    mol = metal.get_parent("molecule")
    metal_index = metal.get_parent_index("molecule")

    modes = []
    for ligand in mol.ligands or []:
        try:
            ligand_coordination = ligand.coordination
        except AttributeError as err:
            raise RuntimeError(
                "coord_modes needs the post-pydantic cell2mol API (ligand.coordination)"
            ) from err

        mode = ligand_coordination.get(metal_index)
        if mode is None:
            continue
        atoms = ligand.atoms or []
        modes.append(
            (
                ligand,
                [atoms[i] for i in mode["kappa_atoms"]],
                [[atoms[i] for i in site] for site in mode["eta_groups"]],
                ligand_coordination,
            )
        )
    return modes


def site_topology(metal, ligand, site: list[Any]) -> dict | None:
    """Topology of one eta site: from the haptic Group with exactly the site's
    atoms (metal-specific groups first, as cell2mol's coordination view
    does), else computed on the site atoms; None if neither works."""
    site_ids = {id(a) for a in site}
    groups = metal.groups if metal.groups is not None else ligand.groups

    for group in groups or []:
        if (
            group.is_haptic
            and group.topology
            and {id(a) for a in group.atoms or []} == site_ids
        ):
            return group.topology

    try:
        from cell2mol.connectivity import analyze_topology

        return dict(analyze_topology(site))
    except Exception:
        return None


class SimpleGraphBuilder:
    """Metal + its donor atoms (+ optional haptic-site and ligand nodes); see
    module docstring for the feature flags."""

    def __init__(
        self,
        elemdatabase: ElementDatabaseAdapter,
        *,
        atom_charge: bool = False,
        atom_charge_flag: bool = False,
        coord_geom: bool = False,
        geom_quality: bool = False,
        coord_modes: bool = False,
        ligand_node: bool = False,
        ligand_charge: bool = True,
    ):
        if atom_charge_flag and not atom_charge:
            raise ValueError("atom_charge_flag requires atom_charge")
        if geom_quality and not coord_geom:
            raise ValueError("geom_quality requires coord_geom")
        if ligand_node and not coord_modes:
            raise ValueError("ligand_node requires coord_modes")

        self.elemdatabase = elemdatabase
        self.atom_charge = atom_charge
        self.atom_charge_flag = atom_charge_flag
        self.coord_geom = coord_geom
        self.geom_quality = geom_quality
        self.coord_modes = coord_modes
        self.ligand_node = ligand_node
        self.ligand_charge = ligand_charge

    @property
    def node_dims(self) -> dict[str, int]:
        atom_dim = (
            len(ATOM_TYPES) + 5 + int(self.atom_charge) + int(self.atom_charge_flag)
        )

        metal_dim = len(METAL_TYPES) + 4
        if self.coord_geom:
            metal_dim += len(COORD_GEOMETRY_TYPES) + 1
        if self.geom_quality:
            metal_dim += 4
        if self.coord_modes:
            metal_dim += 2

        dims = {"atom": atom_dim, "metal": metal_dim}
        if self.ligand_node:
            dims["ligand"] = 2 * int(self.ligand_charge) + 5
        if self.coord_modes:
            dims["haptic"] = 2 + len(ATOM_TYPES) + 2 + len(TOPOLOGY_FLAGS)
        return dims

    def atom_feature(self, atom) -> torch.Tensor:
        label = atom.label
        edb = self.elemdatabase

        x = one_hot(label, ATOM_TYPES) + [
            safe_get(edb.elementnr, label),
            safe_get(edb.elementperiod, label),
            safe_get(edb.ElectroNegativityPauling, label),
            safe_float_attr(atom, "mconnec"),
            safe_float_attr(atom, "connec") - safe_float_attr(atom, "mconnec"),
        ]

        if self.atom_charge:
            charge, has_charge = value_with_flag(getattr(atom, "charge", None))
            x.append(charge)
            if self.atom_charge_flag:
                x.append(has_charge)

        return torch.tensor(x, dtype=torch.float32)

    def metal_feature(self, metal, modes=None) -> torch.Tensor:
        label = metal.label
        edb = self.elemdatabase

        x = one_hot(label, METAL_TYPES) + [
            safe_get(edb.elementnr, label),
            safe_get(edb.elementperiod, label),
            safe_get(edb.ElectroNegativityPauling, label),
            float(len(binding_ligands(metal))),
        ]

        if self.coord_geom:
            coord_geometry = getattr(metal, "coord_geometry", None)
            if coord_geometry is None:
                coord_geometry = "Undefined"
            x += one_hot(coord_geometry, COORD_GEOMETRY_TYPES)
            x.append(safe_float_attr(metal, "coord_nr"))

        if self.geom_quality:
            raw_deviation = getattr(metal, "geom_deviation", None)
            if raw_deviation == "Undefined":
                raw_deviation = None
            x += [
                *value_with_flag(raw_deviation),
                *value_with_flag(rel_metal_radius(metal)),
            ]

        if self.coord_modes:
            donors = {
                id(a)
                for _, kappa, sites, _ in modes
                for a in kappa + [a for s in sites for a in s]
            }
            # No coordination-site count (kappa donors + eta sites): it equals
            # metal.coord_nr, and the ligand nodes carry it per ligand.
            x += [
                float(len(donors)),  # coordinating ligand atoms
                float(
                    sum(
                        getattr(a, "subtype", None) == "metal"
                        for a in metal.coord_sphere_atoms or []
                    )
                ),
            ]

        return torch.tensor(x, dtype=torch.float32)

    def haptic_feature(self, metal, ligand, site: list[Any]) -> torch.Tensor:
        """[n, n_other_elements, element counts over ATOM_TYPES, has_topology,
        n_rings, TOPOLOGY_FLAGS] -- e.g. eta5(Cp) = 5 C in a single simple
        ring, eta5(P5)-ring = 5 P in one."""
        counts = [0.0] * len(ATOM_TYPES)
        for atom in site:
            if atom.label in ATOM_TYPES:
                counts[ATOM_TYPES.index(atom.label)] += 1.0

        topology = site_topology(metal, ligand, site)

        return torch.tensor(
            [float(len(site)), float(len(site)) - sum(counts)]
            + counts
            + [1.0 if topology else 0.0, float((topology or {}).get("n_rings", 0) or 0)]
            + [1.0 if (topology or {}).get(flag) else 0.0 for flag in TOPOLOGY_FLAGS],
            dtype=torch.float32,
        )

    def ligand_feature(self, ligand, kappa, sites, ligand_coordination) -> torch.Tensor:
        x = []
        if self.ligand_charge:
            x += list(value_with_flag(getattr(ligand, "totcharge", None)))

        all_modes = ligand_coordination.values()
        x += [
            float(len(kappa)),
            float(len(sites)),
            float(sum(len(s) for s in sites)),
            float(len(ligand_coordination)),  # mu
            1.0
            if any(m["kappa_atoms"] for m in all_modes)
            and any(m["eta_groups"] for m in all_modes)
            else 0.0,
        ]
        return torch.tensor(x, dtype=torch.float32)

    def build(self, metal, oxidation_state: int | None = None) -> HeteroData:
        if metal.coord_sphere_atoms is None:
            metal.get_coord_sphere_atoms()
        if self.coord_modes:
            return self._build_coord_modes(metal, oxidation_state)

        dims = self.node_dims
        data = HeteroData()

        atom_index: dict[int, int] = {}
        atom_x = []
        for atom in metal.coord_sphere_atoms or []:
            if id(atom) not in atom_index:
                atom_index[id(atom)] = len(atom_x)
                atom_x.append(self.atom_feature(atom))

        data["metal"].x = torch.stack([self.metal_feature(metal)])
        data["atom"].x = stack_or_empty(atom_x, dims["atom"])

        add_edges(
            data,
            ("metal", "coordinates", "atom"),
            [[0, atom_index[id(a)]] for a in metal.coord_sphere_atoms or []],
            [
                metal_atom_edge_feature(self.elemdatabase, metal, a)
                for a in metal.coord_sphere_atoms or []
            ],
        )

        set_label(data, oxidation_state)
        return data

    def _build_coord_modes(self, metal, oxidation_state: int | None) -> HeteroData:
        dims = self.node_dims
        edb = self.elemdatabase
        data = HeteroData()

        modes = coordination_modes(metal)
        kappa_atoms = [a for _, kappa, _, _ in modes for a in kappa]
        sites = [
            (ligand, site)
            for ligand, _, ligand_sites, _ in modes
            for site in ligand_sites
        ]

        atom_index: dict[int, int] = {}
        atom_x = []
        for atom in kappa_atoms + [a for _, site in sites for a in site]:
            if id(atom) not in atom_index:
                atom_index[id(atom)] = len(atom_x)
                atom_x.append(self.atom_feature(atom))

        data["metal"].x = torch.stack([self.metal_feature(metal, modes)])
        data["atom"].x = stack_or_empty(atom_x, dims["atom"])
        if self.ligand_node:
            data["ligand"].x = stack_or_empty(
                [self.ligand_feature(*m) for m in modes], dims["ligand"]
            )
        data["haptic"].x = stack_or_empty(
            [self.haptic_feature(metal, lig, site) for lig, site in sites],
            dims["haptic"],
        )

        add_edges(
            data,
            ("metal", "coordinates", "atom"),
            [[0, atom_index[id(a)]] for a in kappa_atoms],
            [metal_atom_edge_feature(edb, metal, a) for a in kappa_atoms],
        )

        if self.ligand_node:
            ml_pairs, ml_attr, la_pairs = [], [], []
            for l_idx, (_, kappa, ligand_sites, _) in enumerate(modes):
                donors = kappa + [a for site in ligand_sites for a in site]
                ml_pairs.append([0, l_idx])
                ml_attr.append(metal_group_edge_feature(edb, metal, donors))
                la_pairs += [[l_idx, atom_index[id(a)]] for a in donors]

            add_edges(data, ("metal", "binds", "ligand"), ml_pairs, ml_attr)
            add_edges(data, ("ligand", "ligand_contains", "atom"), la_pairs)

        add_edges(
            data,
            ("metal", "haptic_to", "haptic"),
            [[0, h] for h in range(len(sites))],
            [metal_group_edge_feature(edb, metal, site) for _, site in sites],
        )
        add_edges(
            data,
            ("haptic", "contains", "atom"),
            [[h, atom_index[id(a)]] for h, (_, site) in enumerate(sites) for a in site],
        )

        set_label(data, oxidation_state)
        return data


# =============================================================================
# Full family (original model)
# =============================================================================


def canonical_haptic_type(haptic_type: str | None, atoms: list[Any]) -> str:
    # NOTE: builds keys like "eta5_C5", which never match FULL_HAPTIC_TYPES'
    # "eta5(C5)" style -- so every haptic group currently maps to "OTHER".
    # Kept as-is because the existing full-family checkpoints were trained on
    # exactly these features.
    formula = labels2formula([atom.label for atom in atoms])

    if haptic_type:
        match = re.match(r"eta(\d+)", haptic_type)
        if match:
            key = f"eta{match.group(1)}_{formula}"
            return key if key in FULL_HAPTIC_TYPES else "OTHER"

    key = f"eta{len(atoms)}_{formula}"
    return key if key in FULL_HAPTIC_TYPES else "OTHER"


def canonical_coord_geometry(coord_geometry: str | None) -> str:
    if coord_geometry is None:
        return "Undefined"
    return coord_geometry if coord_geometry in FULL_COORD_GEOMETRIES else "OTHER"


def safe_geom_deviation(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ligand_charge_features(metal: Any) -> tuple[float, float, float]:
    """(total_ligand_charge, ligand_count, coord_atom_charge_sum) --
    charge-balance-style aggregates (metal_charge = mol_totcharge - sum of
    ligand charges is the classical rule of thumb). No ligand.denticity: it is
    None for any ligand bound to more than one metal (see
    SimpleGraphBuilder.ligand_feature)."""
    ligands = binding_ligands(metal)
    total_ligand_charge = sum(
        float(lig.totcharge) for lig in ligands if lig.totcharge is not None
    )

    coord_atom_charge_sum = sum(
        float(atom.charge)
        for atom in (metal.coord_sphere_atoms or [])
        if atom.charge is not None
    )

    return total_ligand_charge, float(len(ligands)), coord_atom_charge_sum


class FullGraphBuilder:
    """
    Original metal + atom + haptic-group + ligand graph, built from
    `metal.groups`. include_haptic/include_ligand only gate which node types
    get rows (with include_haptic=False, haptic-group atoms become plain atom
    nodes on "coordinates" edges); node dims stay fixed, so one architecture
    serves all four full-family variants.
    """

    def __init__(
        self,
        elemdatabase: ElementDatabaseAdapter,
        *,
        include_haptic: bool = True,
        include_ligand: bool = True,
    ):
        self.elemdatabase = elemdatabase
        self.include_haptic = include_haptic
        self.include_ligand = include_ligand

    @property
    def node_dims(self) -> dict[str, int]:
        return {
            "atom": len(ATOM_TYPES) + 5,
            "metal": len(METAL_TYPES) + 9 + len(FULL_COORD_GEOMETRIES),
            "haptic": len(FULL_HAPTIC_TYPES) + 1,
            "ligand": 4,
        }

    def atom_feature(self, atom) -> torch.Tensor:
        label = atom.label
        edb = self.elemdatabase

        return torch.tensor(
            one_hot(label, ATOM_TYPES)
            + [
                safe_get(edb.elementnr, label),
                safe_get(edb.elementperiod, label),
                safe_get(edb.CovalentRadius3, label),
                safe_float_attr(atom, "charge"),
                safe_float_attr(atom, "connec"),
            ],
            dtype=torch.float32,
        )

    def metal_feature(self, metal) -> torch.Tensor:
        label = metal.label
        edb = self.elemdatabase

        total_ligand_charge, ligand_count, coord_atom_charge_sum = (
            ligand_charge_features(metal)
        )

        return torch.tensor(
            one_hot(label, METAL_TYPES)
            + [
                safe_get(edb.elementnr, label),
                safe_get(edb.elementperiod, label),
                safe_get(edb.CovalentRadius3, label),
                safe_float_attr(metal, "coord_nr"),
                safe_geom_deviation(metal.geom_deviation),
                float(rel_metal_radius(metal) or 0.0),
                total_ligand_charge,
                ligand_count,
                coord_atom_charge_sum,
            ]
            + one_hot(
                canonical_coord_geometry(metal.coord_geometry), FULL_COORD_GEOMETRIES
            ),
            dtype=torch.float32,
        )

    def haptic_feature(self, group) -> torch.Tensor:
        haptic_type = canonical_haptic_type(
            getattr(group, "haptic_type", None), group.atoms
        )
        return torch.tensor(
            one_hot(haptic_type, FULL_HAPTIC_TYPES) + [float(len(group.atoms))],
            dtype=torch.float32,
        )

    def ligand_feature(self, ligand) -> torch.Tensor:
        return torch.tensor(
            [
                safe_float_attr(ligand, "totcharge"),
                1.0 if getattr(ligand, "is_haptic", False) else 0.0,
                float(len(ligand.atoms or [])),
                float(len(ligand.metals or [])),
            ],
            dtype=torch.float32,
        )

    def build(self, metal, oxidation_state: int | None = None) -> HeteroData:
        if metal.groups is None:
            metal.get_connected_groups()

        dims = self.node_dims
        data = HeteroData()
        edb = self.elemdatabase

        groups = [g for g in metal.groups if g.atoms]
        haptic_groups = [g for g in groups if g.is_haptic and self.include_haptic]

        atom_index: dict[int, int] = {}
        atom_x = []
        for group in groups:
            for atom in group.atoms:
                if id(atom) not in atom_index:
                    atom_index[id(atom)] = len(atom_x)
                    atom_x.append(self.atom_feature(atom))

        ligands = binding_ligands(metal) if self.include_ligand else []

        data["metal"].x = torch.stack([self.metal_feature(metal)])
        data["atom"].x = stack_or_empty(atom_x, dims["atom"])
        data["haptic"].x = stack_or_empty(
            [self.haptic_feature(g) for g in haptic_groups], dims["haptic"]
        )
        data["ligand"].x = stack_or_empty(
            [self.ligand_feature(lig) for lig in ligands], dims["ligand"]
        )

        ma_pairs, ma_attr = [], []
        mh_pairs, mh_attr = [], []
        ha_pairs = []

        h_idx = 0
        for group in groups:
            if group.is_haptic and self.include_haptic:
                mh_pairs.append([0, h_idx])
                mh_attr.append(metal_group_edge_feature(edb, metal, group.atoms))
                ha_pairs += [[h_idx, atom_index[id(a)]] for a in group.atoms]
                h_idx += 1
            else:
                for atom in group.atoms:
                    ma_pairs.append([0, atom_index[id(atom)]])
                    ma_attr.append(metal_atom_edge_feature(edb, metal, atom))

        add_edges(data, ("metal", "coordinates", "atom"), ma_pairs, ma_attr)
        add_edges(data, ("metal", "haptic_to", "haptic"), mh_pairs, mh_attr)
        add_edges(data, ("haptic", "contains", "atom"), ha_pairs)

        la_pairs = [
            [l_idx, atom_index[id(a)]]
            for l_idx, ligand in enumerate(ligands)
            for a in (ligand.atoms or [])
            if id(a) in atom_index
        ]
        add_edges(
            data, ("metal", "binds", "ligand"), [[0, i] for i in range(len(ligands))]
        )
        add_edges(data, ("ligand", "ligand_contains", "atom"), la_pairs)

        set_label(data, oxidation_state)
        return data
