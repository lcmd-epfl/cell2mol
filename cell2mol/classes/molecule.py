from __future__ import annotations

from cell2mol.utils.serialization import ModelFileMixin
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.ligand import Ligand
from cell2mol.classes.species import Species
from cell2mol.connectivity import split_species, merge_coordination_groups
from cell2mol.element_utils import (
    symbols2formula,
    get_df_block_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
    POST_TRANSITION_METALS,
    METALLOIDS,
)
from cell2mol.spin import determine_complex_spin
from cell2mol.operations import extract_from_list
from cell2mol.my_types import Spin, SubType, NDArray
from cell2mol.utils import config
import logging


if TYPE_CHECKING:
    from cell2mol.classes.group import Group

logger = logging.getLogger(__name__)


class Molecule(ModelFileMixin, Species):
    """Molecule structure with child ligands, metals, and coordination results."""

    ligands: list[Ligand] | None = Field(default=None)
    metals: list[Metal] | None = Field(default=None)
    spin: Spin | None = None
    ref_indices: list[int] | None = None
    cell_indices: list[int] | None = None

    unique_index: int | None = None
    totcharge_cif: int | None = None
    totcharge_agree: bool | None = None

    subtype: SubType | None = Field(default="molecule")

    # Per-molecule failures; a MoleculeSet folds them into its own flags.
    error_create_bonds: bool | None = None
    error_get_spin: bool | None = None

    @property
    def has_haptic_coordination(self) -> bool:
        """Whether any ligand-metal interaction contains an eta site.

        Derived from existing coordination; does not build missing groups.
        """
        return any(
            mode["eta_groups"]
            for ligand in self.ligands or []
            for mode in ligand.coordination.values()
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Discard obsolete molecule hapticity summaries from legacy pickle."""
        for name in ("is_haptic", "haptic_type"):
            state.get("__dict__", {}).pop(name, None)
            state.get("__pydantic_fields_set__", set()).discard(name)
        super().__setstate__(state)

    @property
    def contains_metal(self) -> bool:
        """Return whether d or f-block, s-block, or post-transition metals are present."""
        return (
            self.is_metal_complex
            or self.has_s_block_metal
            or self.is_post_transition_metal_complex
        )

    def assign_spin(self):
        """Assign and return spin multiplicity; record and raise assignment failures."""
        self.error_get_spin = False

        try:
            if self.is_metal_complex:
                self.spin = determine_complex_spin(self)
            else:
                if (self.eleccount - (self.totcharge or 0)) % 2 == 0:
                    self.spin = 1
                else:
                    self.spin = 2

        except Exception as e:
            self.error_get_spin = True
            raise RuntimeError(
                f"Spin assignment failed for molecule {self.formula}"
            ) from e

        logger.debug(
            "Assigned spin multiplicity | molecule=%s | spin=%s",
            self.formula,
            self.spin,
        )

        return self.spin

    def reset_charge(self):
        """Clear molecule, atom, ligand, and metal charge state."""
        super().reset_charge()
        if self.ligands is not None:
            for lig in self.ligands:
                lig.reset_charge()
        if self.metals is not None:
            for met in self.metals:
                met.reset_charge()

    def split_complex(self, use_bond_info: bool | None = None, post_tms: bool = False):
        """Split into metals and ligands, preserving shared atoms and parent mappings.

        With post_tms=True, treat only post-transition elements as metals."""

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        if self.atoms is None:
            self.set_atoms()

        if self.is_metal_free_molecule:
            logger.debug(
                "Molecule %s is not a complex. No splitting needed.", self.formula
            )
            self.ligands = None
            self.metals = None
            return self.ligands, self.metals

        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None

        cov_factor = getattr(self, "cov_factor", config.COV_FACTOR)
        metal_factor = getattr(self, "metal_factor", config.METAL_FACTOR)

        self.ligands = []
        self.metals = []

        # Metal indices refer to positions in this molecule.
        metal_idx: set[int]
        if post_tms:
            logger.info(
                "post_tms enabled: ONLY post-transition metals exist as metals."
            )
            metal_idx = {
                self.indices[i] for i in get_post_transition_metal_idxs(self.labels)
            }
        else:
            metal_idx = {self.indices[i] for i in get_df_block_metal_idxs(self.labels)}
            metal_idx.update(
                self.indices[i]
                for i in get_alkali_alkaline_earth_metal_idxs(self.labels)
            )

        post_tm_found = [
            label for label in self.labels if label in POST_TRANSITION_METALS
        ]
        metalloid_found = [label for label in self.labels if label in METALLOIDS]
        if post_tm_found:
            logger.info(
                "Post-transition metals found: %s %s ", self.formula, post_tm_found
            )
        if metalloid_found:
            logger.info("Metalloids found: %s %s", self.formula, metalloid_found)

        # Partition the remaining atoms into ligands.

        rest_idx = [i for i in self.indices if i not in metal_idx]

        assert self.radii is not None
        assert self.atoms is not None
        rest_labels = extract_from_list(rest_idx, self.labels, dimension=1)
        rest_coord = extract_from_list(rest_idx, self.coord.tolist(), dimension=1)
        rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)
        rest_radii = extract_from_list(rest_idx, self.radii.tolist(), dimension=1)
        rest_atoms = extract_from_list(rest_idx, self.atoms, dimension=1)

        rest_frac = (
            extract_from_list(rest_idx, self.frac_coord.tolist(), dimension=1)
            if self.frac_coord is not None
            else None
        )

        rest_atom_site_labels = (
            extract_from_list(rest_idx, self.atom_site_labels, dimension=1)
            if self.atom_site_labels is not None
            else None
        )

        if not rest_labels:
            logger.debug(
                "No ligands found in complex %s. Assigning metals only.",
                self.formula,
            )
            self.metals.extend(cast(Metal, self.atoms[i]) for i in metal_idx)
            return self.ligands, self.metals

        blocklist = cast(
            "list[list[int]]",
            split_species(
                labels=rest_labels,
                positions=np.asarray(rest_coord),
                radii=rest_radii,
                indices=None,  # Return indices local to the remaining atoms.
                atom_site_labels=rest_atom_site_labels,
                bond_data=bond_data,
                use_bond_info=use_bond_info,
                cov_factor=cov_factor,
            ),
        )

        logger.info("Received %d ligand blocks", len(blocklist))

        for block in blocklist:
            lig_indices = extract_from_list(block, rest_indices, dimension=1)
            lig_labels = extract_from_list(block, rest_labels, dimension=1)
            lig_coord = extract_from_list(block, rest_coord, dimension=1)
            lig_radii = extract_from_list(block, rest_radii, dimension=1)
            lig_atoms = extract_from_list(block, rest_atoms, dimension=1)

            lig_frac_coord = (
                extract_from_list(block, rest_frac, dimension=1)
                if rest_frac is not None
                else None
            )

            lig_atom_site_labels = (
                extract_from_list(block, rest_atom_site_labels, dimension=1)
                if rest_atom_site_labels is not None
                else None
            )

            logger.info("  Ligand: %s", symbols2formula(lig_labels))

            if lig_frac_coord is not None:
                ligand = Ligand.from_positional(
                    lig_labels, lig_coord, lig_frac_coord, radii=lig_radii
                )
            else:
                ligand = Ligand.from_positional(lig_labels, lig_coord, radii=lig_radii)

            ligand.set_origin("split_complex")
            ligand.add_parent(self, indices=lig_indices)

            unitcell_parent = self.get_parent("unitcell")
            if unitcell_parent is not None:
                ligand.add_parent(
                    unitcell_parent,
                    indices=[a.get_parent_index("unitcell") or 0 for a in lig_atoms],
                )

            reference_parent = self.get_parent("reference")
            if reference_parent is not None:
                ligand.add_parent(
                    reference_parent,
                    indices=[a.get_parent_index("reference") or 0 for a in lig_atoms],
                )

            ligand.set_adjacency_parameters(cov_factor, metal_factor)
            ligand.set_atoms(
                atomlist=lig_atoms,
                create_adjacencies=False,
                atom_site_labels=lig_atom_site_labels,
                use_bond_info=use_bond_info,
            )
            ligand.inherit_adjacency("molecule")

            self.ligands.append(ligand)

        self.metals.extend(cast(Metal, self.atoms[i]) for i in metal_idx)

        return self.ligands, self.metals

    def _log_metal_content(self) -> bool:
        """Log the metal category and return whether the molecule contains metals."""
        if self.is_metal_complex:
            logger.info("Has transition metals %s", self.formula)
            if not self.ligands:
                logger.debug("A metal cluster found")
        elif self.has_s_block_metal:
            logger.info("Has alkali or alkaline earth metals: %s", self.formula)
            if not self.ligands:
                logger.debug("Alkali/alkaline earth metal ion found")
        elif self.is_post_transition_metal_complex:
            logger.info("Has post transition metals: %s", self.formula)
            logger.debug("metals=%s", [met.label for met in self.metals or []])
            logger.debug("ligands=%s", [lig.formula for lig in self.ligands or []])
        else:
            logger.info("No metals found in molecule: %s", self.formula)
            return False
        return True

    def analyze_coordination(self):
        """Analyze metal environments, map and merge donor groups, and update ligands."""

        if not self._log_metal_content():
            return

        for met in self.metals or []:
            logger.debug(
                "Analyzing coordination for metal %s%s",
                met.label,
                (f" ({met.atom_site_label})" if met.atom_site_label else ""),
            )
            met.get_connected_metals()
            logger.debug("  Connected Metals: %s", [m.label for m in met.metals or []])
            met.get_connected_nonmetal_atoms()
            logger.debug(
                "  Connected Non-Metals: %s",
                [m.label for m in met.connected_nonmetal_atoms or []],
            )
            met.get_connected_groups()
            met.determine_coordination_geometry()
            met.compute_relative_metal_radius()
            met.build_coordination_sphere_formula()

        self.map_metal_groups_to_ligands()
        # self.merge_connected_groups()

        for lig in self.ligands or []:
            lig.update_hapticity()
            lig.update_denticity()
            logger.debug(
                f"Ligand: {lig.formula}, Groups: {[group.formula for group in lig.groups or []]}"
            )
            for group in lig.groups or []:
                logger.debug(
                    f"  Group: {group.formula}, Haptic: {group.haptic_type}, Connected Metal: {[met.atom_site_label for met in group.metals or []]}"
                )
                for atom in group.atoms or []:
                    logger.debug(
                        f"    Atom: {atom.atom_site_label}, neighbor_count: {atom.neighbor_count} metal_contact_count: {atom.metal_contact_count}"
                    )

    def detect_special_moieties(self):
        """Screen ligands in metal-containing molecules, or the whole metal-free species."""

        if not self.contains_metal:
            super().detect_special_moieties()
            return

        for lig in self.ligands or []:
            lig.detect_special_moieties()

    def map_metal_groups_to_ligands(self):
        """Collect unique donor groups per ligand and link bridging metals."""
        for ligand in self.ligands or []:
            ligand.groups = []
        groups_by_site: dict[tuple[int, frozenset[int]], Group] = {}

        for metal in self.metals or []:
            for group in metal.groups or []:
                ligand = cast("Ligand | None", group.get_parent("ligand"))
                if ligand is None:
                    continue
                indices = frozenset(group.get_parent_indices("ligand") or [])
                key = (id(ligand), indices)
                existing = groups_by_site.get(key)
                if existing is not None:
                    if existing.metals is None:
                        existing.metals = []
                    if not any(linked is metal for linked in existing.metals):
                        existing.metals.append(metal)
                    logger.debug(
                        "Bridging group detected: Metal %s linked to group %s in Ligand %s",
                        metal.label,
                        existing.formula,
                        ligand.formula,
                    )
                    continue

                groups_by_site[key] = group
                if ligand.groups is None:
                    ligand.groups = []
                ligand.groups.append(group)
                logger.info(
                    "Mapped %s (haptic_type: %s) from Metal %s to Ligand %s",
                    group.formula,
                    group.haptic_type,
                    metal.label,
                    ligand.formula,
                )

    def merge_connected_groups(self):
        """Merge overlapping or bonded ligand groups using connected-component topology."""
        for lig in self.ligands or []:
            if not lig.groups:
                continue

            n = len(lig.groups)
            adj = {i: set() for i in range(n)}

            assert self.adjacency_matrix is not None
            adjacency_matrix = self.adjacency_matrix
            group_atom_sets = [
                {atom.get_parent_index("molecule") for atom in g.atoms or []}
                for g in lig.groups
            ]

            for i in range(n):
                atoms_i = group_atom_sets[i]
                for j in range(i + 1, n):
                    atoms_j = group_atom_sets[j]

                    # Link groups that share atoms.
                    if not atoms_i.isdisjoint(atoms_j):
                        adj[i].add(j)
                        adj[j].add(i)
                        continue

                    connected = False
                    # Otherwise link groups joined by a bond.
                    for ai in atoms_i:
                        for aj in atoms_j:
                            if adjacency_matrix[ai, aj]:
                                connected = True
                                break
                        if connected:
                            break

                    if connected:
                        adj[i].add(j)
                        adj[j].add(i)

            # Find connected components of the group graph.
            visited = [False] * n
            merged_groups = []

            for i in range(n):
                if visited[i]:
                    continue

                stack = [i]
                visited[i] = True
                component = []

                while stack:
                    curr = stack.pop()
                    component.append(curr)
                    for nb in adj[curr]:
                        if not visited[nb]:
                            visited[nb] = True
                            stack.append(nb)

                if len(component) == 1:
                    merged_groups.append(lig.groups[component[0]])
                else:
                    groups_to_merge = [lig.groups[idx] for idx in component]
                    new_groups = merge_coordination_groups(self, groups_to_merge, lig)
                    if new_groups:
                        merged_groups.extend(new_groups)
                        logger.debug(
                            "Merged %d connected groups in %s → %s",
                            len(groups_to_merge),
                            lig.formula,
                            [g.haptic_type for g in new_groups],
                        )
                    else:
                        logger.warning("Failed to merge groups in %s", lig.formula)
                        merged_groups.extend(groups_to_merge)

            lig.groups = merged_groups

    def create_bonds(self):
        """Build atom bonds and metal links, recording bond-generation failures."""
        from cell2mol.charge.molecule_builder import (
            create_bonds_species,
            create_metal_ligand_bonds,
            create_metal_metal_bonds,
        )

        self.bonds = []
        if self.is_metal_free_molecule:
            result = create_bonds_species(self)
            if not result:
                logger.error("Error for metal-free molecule %s", self.formula)
                self.error_create_bonds = True
                return
            else:
                logger.debug("Bonds created for metal-free molecule %s", self.formula)

        # Ligand molecular forms are finalized during charge assignment.
        if (
            self.is_metal_complex
            or self.has_s_block_metal
            or self.is_post_transition_metal_complex
        ):
            for lig in self.ligands or []:
                if lig.smiles is None:
                    logger.error(
                        "Ligand %s has no SMILES after charge assignment. Cannot create bonds.",
                        lig.formula,
                    )
                result = create_bonds_species(lig)
                if not result:
                    logger.error("Error for ligand %s", lig.formula)
                    self.error_create_bonds = True
                    return

                positions = {id(atom): i for i, atom in enumerate(self.atoms or [])}
                for bond in lig.bonds:
                    indices = [
                        positions[id(lig.atoms[i])] for i in bond["atom_indices"]
                    ]
                    self.add_bond(indices, bond["order"])

                logger.debug("Bonds created for ligand %s", lig.formula)

        # Add metal–ligand and metal–metal links.
        if (
            self.is_metal_complex
            or self.has_s_block_metal
            or self.is_post_transition_metal_complex
        ):
            create_metal_ligand_bonds(self)
            create_metal_metal_bonds(self)

        self.error_create_bonds = False

    def __repr__(self, indirect: bool = False):
        """Return species details and counts of child ligands and metals."""
        to_print = ""
        to_print += "------------- Cell2mol MOLECULE Object --------------\n"
        to_print += Species.__repr__(self, indirect=True)
        if self.ligands is not None:
            to_print += f" Number of Ligands            = {len(self.ligands)}\n"
        if self.metals is not None:
            to_print += f" Number of Metals             = {len(self.metals)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use Molecule() with keyword arguments instead.")
    def from_positional(
        cls,
        symbols: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
    ) -> "Molecule":
        """Construct a molecule using the legacy positional interface."""
        return cls(
            symbols=symbols,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )
