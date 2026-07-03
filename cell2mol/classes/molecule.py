from __future__ import annotations
import pickle
from typing import cast

import numpy as np
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.ligand import Ligand
from cell2mol.classes.specie import Specie
from cell2mol.classes.charge_state import ChargeState
from cell2mol.connectivity import split_species, merge_multiple_groups
from cell2mol.element_utils import (
    labels2formula,
    get_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
    POST_TRANSITION_METALS,
    METALLOIDS,
)
from cell2mol.compare import compare_species, compare_metals
from cell2mol.charge.specie_assigner import set_charge_state, prepare_mol
from cell2mol.charge.smiles_handler import (
    create_bonds_specie,
    create_metal_ligand_bonds,
    create_metal_metal_bonds,
    correct_smiles_ligand,
)
from cell2mol.spin import assign_spin_complexes
from cell2mol.operations import extract_from_list
from cell2mol.elementdata import ElementData
from cell2mol.my_types import Format, Spin, SubType
from cell2mol.utils import config
import logging
from pathlib import Path

try:
    from typing import Self  # py3.11+
except ImportError:
    from typing_extensions import Self  # py3.10

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


class Molecule(Specie):
    """
    A molecule is a specie that contains other specie objects.
    """

    haptic_type: list[str] | None = None
    is_haptic: bool | None = None
    ligands: list[Ligand] | None = Field(default=None)
    metals: list[Metal] | None = Field(default=None)
    spin: Spin | None = None
    ref_indices: list[int] | None = None
    cell_indices: list[int] | None = None

    unique_index: int | None = None
    totcharge_cif: int | None = None

    ligand_smiles: str | list[str] | None = None
    subtype: SubType | None = Field(default="molecule")

    # Needed in interpret_molecule in process_xyz.py
    input_charge: int | None = None
    unique_species: list[Specie | Metal] | None = None
    unique_indices: list[int] | None = None
    species_list: list[Specie | Metal] | None = None
    selected_cs: list[object] | None = None
    error_get_poscharges: bool | None = None
    error_multiple_distrib: bool | None = None
    error_empty_distrib: bool | None = None
    error_assign_charge: bool | None = None
    error_create_bonds: bool | None = None
    error_get_spin: bool | None = None
    error_case: int | None = None

    @classmethod
    @deprecated("Use molecule() with the keyword arguments instead.")
    def from_positional(
        cls,
        labels: list[str],
        coord: np.ndarray | list[list[float]],
        frac_coord: np.ndarray | list[list[float]] | None = None,
        radii: np.ndarray | list[float] | None = None,
    ) -> "Molecule":
        return cls(
            labels=labels,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )

    def __repr__(self, indirect: bool = False):
        to_print = ""
        to_print += "------------- Cell2mol MOLECULE Object --------------\n"
        to_print += Specie.__repr__(self, indirect=True)
        if self.ligands is not None:
            if self.ligands is not None:
                # to_print += f" Ligands Smiles               = {self.ligand_smiles}\n"
                to_print += f" Number of Ligands            = {len(self.ligands)}\n"
        if self.metals is not None:
            if self.metals is not None:
                to_print += f" Number of Metals             = {len(self.metals)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    def get_spin(self):
        """
        Assign and return spin multiplicity for this molecule.
        Sets mol.error_get_spin on failure and raises an exception.
        """
        self.error_get_spin = False

        try:
            if self.iscomplex:
                self.spin = assign_spin_complexes(self)
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
        # First, uses the generic specie class function for itself and its atoms
        Specie.reset_charge(self)
        ## Then reset charge for the child classes
        if self.ligands is not None:
            for lig in self.ligands:
                lig.reset_charge()
        if self.metals is not None:
            for met in self.metals:
                met.reset_charge()

    def split_complex(self, use_bond_info: bool | None = None, post_tms: bool = False):
        """
        Split a complex into metals and ligands.

        Args:
        ----------
        use_bond_info : bool or None, optional
            Whether to use explicit bond information.
            If None, defaults to config.USE_BOND_INFO.
        post_tms : bool, optional
            If True, ONLY post-transition metals are treated as metals.
        """

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        # Ensure atoms exist
        if self.atoms is None:
            self.set_atoms()

        # Non-complex shortcut
        # If the molecule is not a complex and does not contain IA/IIA metals,
        # there is nothing to split.
        if self.is_non_complex_molecule:
            logger.debug(
                "Molecule %s is not a complex. No splitting needed.", self.formula
            )
            self.ligands = None
            self.metals = None
            return self.ligands, self.metals

        # Reference / bond information
        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None

        cov_factor = getattr(self, "cov_factor", config.COV_FACTOR)
        metal_factor = getattr(self, "metal_factor", config.METAL_FACTOR)

        self.ligands = []
        self.metals = []

        # ============================================================
        # Identify metal indices (GLOBAL index space: self.indices)
        # ============================================================
        metal_idx: set[int]
        if post_tms:
            logger.info(
                "post_tms enabled: ONLY post-transition metals exist as metals."
            )
            metal_idx = {
                self.indices[i] for i in get_post_transition_metal_idxs(self.labels)
            }
        else:
            metal_idx = {self.indices[i] for i in get_metal_idxs(self.labels)}
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

        # ============================================================
        # Remaining atoms → ligands
        # ============================================================

        rest_idx = [i for i in self.indices if i not in metal_idx]

        assert self.radii is not None
        assert self.atoms is not None
        rest_labels = extract_from_list(rest_idx, self.labels, dimension=1)
        rest_coord = extract_from_list(rest_idx, self.coord.tolist(), dimension=1)
        rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)
        rest_radii = extract_from_list(rest_idx, self.radii.tolist(), dimension=1)
        rest_atoms = extract_from_list(rest_idx, self.atoms, dimension=1)
        # logger.debug("Remaining atom indices: %s", rest_idx)

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

        # No ligands case
        if not rest_labels:
            logger.debug(
                "No ligands found in complex %s. Assigning metals only.",
                self.formula,
            )
            self.metals.extend(cast(Metal, self.atoms[i]) for i in metal_idx)
            return self.ligands, self.metals

        # ============================================================
        # Split ligands
        # ============================================================

        blocklist = cast(
            "list[list[int]]",
            split_species(
                labels=rest_labels,
                positions=np.asarray(rest_coord),
                radii=rest_radii,
                indices=None,  # rest_indices
                atom_site_labels=rest_atom_site_labels,
                bond_data=bond_data,
                use_bond_info=use_bond_info,
                cov_factor=cov_factor,
            ),
        )

        logger.info("Received %d ligand blocks", len(blocklist))
        # logger.debug("Blocks: %s", blocklist)

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

            logger.info("  Ligand: %s", labels2formula(lig_labels))

            # --- create ligand ---
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
            ligand.set_inherit_adjmatrix("molecule")

            self.ligands.append(ligand)

        # ============================================================
        # Metals
        # ============================================================

        self.metals.extend(cast(Metal, self.atoms[i]) for i in metal_idx)

        return self.ligands, self.metals

    def analyze_coordination(self):
        """
        Analyze the coordination environment of the molecule.
        Determines hapticity and connected metals for ligands.
        """

        if self.iscomplex:
            logger.info("Has transition metals %s", self.formula)
            if not self.ligands:
                logger.debug("A metal cluster found")
        elif self.has_ia_iia:
            logger.info("Has alkali or alkaline earth metals: %s", self.formula)
            if not self.ligands:
                logger.debug("Alkali/alkaline earth metal ion found")
        elif self.has_post_transition_metal:
            logger.info("Has post transition metals: %s", self.formula)
            logger.debug("metals=%s", [met.label for met in self.metals or []])
            logger.debug("ligands=%s", [lig.formula for lig in self.ligands or []])
        else:
            logger.info("No metals found in molecule: %s", self.formula)
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
            met.get_coordination_geometry()
            met.get_coord_sphere_formula()

        self.map_metal_groups_to_ligands()
        self.merge_connected_groups()

        for lig in self.ligands or []:
            lig.get_hapticity()
            lig.get_denticity()
            logger.debug(
                f"Ligand: {lig.formula}, Groups: {[group.formula for group in lig.groups or []]}"
            )
            for group in lig.groups or []:
                logger.debug(
                    f"  Group: {group.formula}, Haptic: {group.haptic_type}, Connected Metal: {[met.atom_site_label for met in group.metals or []]}"
                )
                for atom in group.atoms or []:
                    logger.debug(
                        f"    Atom: {atom.atom_site_label}, connec: {atom.connec} mconnec: {atom.mconnec}"
                    )

    def map_metal_groups_to_ligands(self):
        """
        Maps refined metal-coordinated groups to their parent ligands.
        Handles bridging coordination by comparing ligand atom indices.
        """
        # 1. Reset ligand coordination storage
        for lig in self.ligands or []:
            lig.groups = []

        # 2. Iterate through each metal and its identified groups
        for met in self.metals or []:
            groups = getattr(met, "groups", [])

            for group in groups:
                parent_ligand = cast("Ligand | None", group.get_parent("ligand"))
                # Get the local indices of these atoms within the ligand
                current_indices = set(group.get_parent_indices("ligand") or [])

                if parent_ligand:
                    # 3. Check if a group with the same ligand indices already exists
                    is_duplicate = False
                    for added_group in parent_ligand.groups or []:
                        added_indices = set(
                            added_group.get_parent_indices("ligand") or []
                        )

                        if current_indices == added_indices:
                            # It's the same coordination site! Just link the new metal.
                            if added_group.metals is None:
                                added_group.metals = []
                            if met not in added_group.metals:
                                added_group.metals.append(met)

                            logger.debug(
                                "Bridging group detected: Metal %s linked to existing group %s in Ligand %s",
                                met.label,
                                added_group.formula,
                                parent_ligand.formula,
                            )
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        # 4. New unique coordination group
                        if parent_ligand.groups is None:
                            parent_ligand.groups = []
                        parent_ligand.groups.append(group)
                        group.parent_ligand = parent_ligand

                        logger.info(
                            "Mapped %s (%s) from Metal %s to Ligand %s",
                            group.formula,
                            group.haptic_type,
                            met.label,
                            parent_ligand.formula,
                        )

    def merge_connected_groups(self):
        """
        Merge ligand groups that are graph-connected.

        Groups are merged if they:
        1) Are purely comprised of Carbon atoms (no heteroatoms allowed in the group),
        2) Share atoms (overlap), or
        3) Are directly bonded through the molecule adjacency graph (C-C bonds).

        This allows reconstruction of extended haptic domains (e.g., fused rings)
        while keeping heteroatom groups isolated.
        """
        for lig in self.ligands or []:
            if not lig.groups:
                continue

            n = len(lig.groups)
            adj = {i: set() for i in range(n)}

            assert self.adjmat is not None
            adjmat = self.adjmat  # Molecule adjacency matrix
            # mol_labels = self.labels  # Molecule atom labels
            group_atom_sets = [
                set([atom.get_parent_index("molecule") for atom in g.atoms or []])
                for g in lig.groups
            ]
            # --- NEW: Identify which groups are purely Carbon ---
            # We create a boolean mask: True if the group contains ONLY "C", False otherwise.
            # group_is_pure_carbon = [
            #     all(mol_labels[idx] == "C" for idx in atoms)
            #     for atoms in group_atom_sets
            # ]

            # --- Build connectivity graph between groups ---
            for i in range(n):
                # STRICT RULE: If group i has heteroatoms (e.g., N1), it is not mergeable.
                # if not group_is_pure_carbon[i]:
                #     continue

                atoms_i = group_atom_sets[i]

                for j in range(i + 1, n):
                    # STRICT RULE: If group j has heteroatoms, it is not mergeable.
                    # if not group_is_pure_carbon[j]:
                    #     continue

                    atoms_j = group_atom_sets[j]

                    # Condition 1: overlap
                    if not atoms_i.isdisjoint(atoms_j):
                        adj[i].add(j)
                        adj[j].add(i)
                        continue

                    # Condition 2: bonded continuity
                    connected = False
                    # Direct check: Is there ANY bond between ANY atom in i and ANY atom in j?
                    # This dual loop is safe for both Dense (numpy) and Sparse (scipy) matrices.
                    for ai in atoms_i:
                        for aj in atoms_j:
                            if adjmat[ai, aj]:  # Non-zero means bonded
                                connected = True
                                break
                        if connected:
                            break

                    if connected:
                        adj[i].add(j)
                        adj[j].add(i)

                    if connected:
                        adj[i].add(j)
                        adj[j].add(i)

            # --- Find connected components ---
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

                # --- Merge component ---
                if len(component) == 1:
                    merged_groups.append(lig.groups[component[0]])
                else:
                    groups_to_merge = [lig.groups[idx] for idx in component]
                    # Expecting a LIST of groups
                    new_groups = merge_multiple_groups(self, groups_to_merge, lig)
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

    def get_hapticity(self):
        if self.ligands is None:
            self.split_complex()
        self.is_haptic = False
        self.haptic_type = []
        if self.iscomplex:
            for lig in self.ligands or []:
                if lig.is_haptic is None:
                    lig.get_hapticity()
                if lig.is_haptic:
                    self.is_haptic = True
                for entry in lig.haptic_type or []:
                    self.haptic_type.append(entry)

        return self.haptic_type

    # def save(self, path):
    #     logger.info(f"SAVING cell2mol MOLECULE object to {path}")
    #     with open(path, "wb") as fil:
    #         pickle.dump(self, fil)

    def save(self, path: str | Path, *, format: Format = "json"):
        if format == "json":
            self._save_as_json(path)
        elif format == "pickle":
            self._save_as_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: str | Path, *, format: Format = "json") -> Self:
        if format == "json":
            return cls._load_from_json(path)
        elif format == "pickle":
            return cls._load_from_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @deprecated("Use json format instead")
    def _save_as_pickle(
        self,
        path: str | Path,
    ):
        logger.warning("Use json format instead")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    def _save_as_json(
        self,
        path: str | Path,
    ):
        if not str(path).endswith(".json"):
            logger.warning("Use `.json` extension instead for path: %s", path)
        # Pretty print with indent=4
        with open(path, "w") as fd:
            fd.write(self.to_json(indent=4))
        # Minified version
        # with open(path, "w") as fd:
        #     fd.write(self.to_json(separators=(",", ":")))

    @classmethod
    def _load_from_json(
        cls,
        path: str | Path,
    ) -> Self:
        with open(path, "r") as fd:
            return cls.from_json(fd.read())

    @classmethod
    @deprecated("Use json format instead")
    def _load_from_pickle(
        cls,
        path: str | Path,
    ):
        with open(path, "rb") as fil:
            return pickle.load(fil)

    def get_unique_species(self):
        logger.info("Getting unique species in molecule: %s", self.formula)

        self.unique_species = []
        self.unique_indices = []
        self.species_list = []

        typelist_mols = []
        typelist_ligs = []
        typelist_mets = []

        specs_found = -1

        # Case 1: simple molecule (not complex, not IA/IIA)
        if (
            not self.iscomplex
            and not self.has_ia_iia
            and not self.has_post_transition_metal
        ):
            found = False
            for ldx, typ in enumerate(typelist_mols):
                issame = compare_species(self, typ[0])
                if issame:
                    found = True
                    kdx = typ[1]
                    logger.debug("molecule is the same as type %s", ldx)

            if not found:
                specs_found += 1
                kdx = specs_found
                typelist_mols.append([self, kdx])
                self.unique_species.append(self)
                logger.debug(
                    "New molecule found: formula=%s, added at specie type %d",
                    self.formula,
                    kdx,
                )

            assert kdx is not None
            self.unique_indices.append(kdx)
            self.unique_index = kdx
            self.species_list.append(self)

        else:
            # Ensure ligands and metals are available
            if not self.ligands is not None:
                if self.iscomplex or self.has_ia_iia:
                    self.split_complex()
                elif self.has_post_transition_metal:
                    self.split_complex(post_tms=True)
            # Case 2: ligands
            for jdx, lig in enumerate(self.ligands or []):
                found = False
                for ldx, typ in enumerate(typelist_ligs):
                    if lig.is_nitrosyl is None:
                        lig.evaluate_as_nitrosyl()
                    if typ[0].is_nitrosyl is None:
                        typ[0].evaluate_as_nitrosyl()
                    if lig.haptic_type is None:
                        lig.get_hapticity()
                    if typ[0].haptic_type is None:
                        typ[0].get_hapticity()
                    lig_groups_labels = [g.labels for g in lig.groups or []]
                    typ_groups_labels = [g.labels for g in typ[0].groups or []]

                    if lig.is_nitrosyl and typ[0].is_nitrosyl:
                        issame = lig.NO_type == typ[0].NO_type
                    else:
                        if (
                            len(lig_groups_labels) == len(typ_groups_labels)
                            and sorted(lig_groups_labels) == sorted(typ_groups_labels)
                            and lig.haptic_type == typ[0].haptic_type
                        ):
                            # if lig.haptic_type == typ[0].haptic_type:
                            issame = compare_species(lig, typ[0])
                        else:
                            issame = False

                    if issame:
                        found = True
                        kdx = typ[1]
                        logger.debug(
                            "ligand %s (%d) is the same with type %d in typelist",
                            lig.formula,
                            jdx,
                            ldx,
                        )
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_ligs.append([lig, kdx])
                    self.unique_species.append(lig)
                    logger.debug(
                        "New ligand found: %s, added at specie type %d",
                        lig.formula,
                        kdx,
                    )

                assert kdx is not None
                self.unique_indices.append(kdx)
                lig.unique_index = kdx
                self.species_list.append(lig)

            # Case 3: metals
            for jdx, met in enumerate(self.metals or []):
                found = False
                kdx: int | None = None
                for ldx, typ in enumerate(typelist_mets):
                    issame = compare_metals(met, typ[0])
                    if issame:
                        found = True
                        kdx = typ[1]
                        logger.debug(
                            "metal %s (%d) is the same with type %d in typelist",
                            met.formula,
                            jdx,
                            ldx,
                        )
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_mets.append([met, kdx])
                    self.unique_species.append(met)
                    logger.debug(
                        "New metal found: formula=%s, added at specie type %d",
                        met.formula,
                        kdx,
                    )

                assert kdx is not None
                self.unique_indices.append(kdx)
                met.unique_index = kdx
                self.species_list.append(met)

        return self.unique_species

    def get_selected_cs(self):
        if not self.unique_species is not None:
            self.get_unique_species()

        self.selected_cs = []
        for unique_specie in self.unique_species or []:
            logger.info(
                "Get possible charge states for unique specie %s", unique_specie.formula
            )
            tmp = unique_specie.get_possible_cs()
            if tmp is None:
                self.selected_cs.append(None)
            elif len(tmp) == 0:
                self.selected_cs.append(None)
            elif unique_specie.subtype != "metal":
                self.selected_cs.append(
                    [
                        cs.corr_total_charge
                        for cs in cast("list[ChargeState]", unique_specie.possible_cs)
                    ]
                )
            else:
                self.selected_cs.append(unique_specie.possible_cs)

        for specie in self.species_list or []:
            logger.info(
                "Get possible charge states for species list %s", specie.formula
            )
            tmp = specie.get_possible_cs()
            if tmp is None:
                self.selected_cs.append(None)
            elif len(tmp) == 0:
                self.selected_cs.append(None)
            elif specie.subtype != "metal":
                self.selected_cs.append(
                    [
                        cs.corr_total_charge
                        for cs in cast("list[ChargeState]", specie.possible_cs)
                    ]
                )
            else:
                self.selected_cs.append(specie.possible_cs)

        if None in self.selected_cs:
            self.error_get_poscharges = True
        else:
            self.error_get_poscharges = False

    def assign_charges(self):
        logger.info("Assigning charges for molecule: %s", self.formula)
        for specie in self.unique_species or []:
            specie_unique_index = getattr(specie, "unique_index", None)
            if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
                for jdx, lig in enumerate(self.ligands or []):
                    if lig.unique_index == specie_unique_index:
                        set_charge_state(specie, lig, mode=1)
                for kdx, met in enumerate(self.metals or []):
                    if met.unique_index == specie_unique_index:
                        specie_charge = getattr(specie, "charge", None)
                        if specie_charge is not None:
                            met.set_charge(specie_charge)
            else:
                if self.unique_index == specie_unique_index:
                    set_charge_state(specie, self, mode=1)
        temp = []
        self.create_bonds()
        temp.append(self.error_create_bonds)
        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            prepare_mol(self)

        if any(temp):
            self.error_create_bonds = True
        else:
            self.error_create_bonds = False

        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            prepare_mol(self)
            logger.info("Complex %s %s", self.formula, self.totcharge)
            for jdx, lig in enumerate(self.ligands or []):
                logger.info(
                    "    Ligand %d %s %s %s",
                    jdx,
                    lig.formula,
                    lig.totcharge,
                    lig.smiles,
                )
            for kdx, met in enumerate(self.metals or []):
                logger.info("    Metal %d %s %s", kdx, met.formula, met.charge)
        else:
            logger.info(
                "Non-Complex %s %s %s", self.formula, self.totcharge, self.smiles
            )

    def create_bonds(self):
        # First part: Non-complex molecule
        if self.is_non_complex_molecule:
            # Creates bonds between molecule.atoms using the molecule.rdkit_object
            result = create_bonds_specie(self)
            if not result:
                logger.error("Error for non-complex molecule %s", self.formula)
                self.error_create_bonds = True
                return  # Exit the function entirely if creating bonds fails
            else:
                logger.debug("Bonds created for non-complex molecule %s", self.formula)

        # Second part: Complex molecule, add bonds for ligands
        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            for lig in self.ligands or []:
                if lig.smiles is None:
                    logger.error(
                        "Ligand %s has no SMILES after charge assignment. Cannot create bonds.",
                        lig.formula,
                    )
            self.ligand_smiles = []
            fix_zwitterions_ligands = []

            for lig in self.ligands or []:
                # Creates bonds between ligand.atoms, using the ligand.rdkit_object
                result = create_bonds_specie(lig)
                if not result:
                    logger.error("Error for ligand %s", lig.formula)

                    self.error_create_bonds = True
                    return  # Exit the function entirely if creating bonds fails for any ligand

                logger.debug("Bonds created for ligand %s", lig.formula)
                logger.debug("Correcting Smiles for ligand %s", lig.formula)
                result, fix_zwitterions = correct_smiles_ligand(lig)
                if not result:
                    logger.error(
                        "Error for ligand %s in correcting smiles", lig.formula
                    )
                    self.error_create_bonds = True
                    return  # Exit the function entirely

                logger.debug("Smiles corrected for ligand %s", lig.formula)
                if fix_zwitterions:
                    fix_zwitterions_ligands.append(lig)
                else:
                    self.ligand_smiles.append(lig.smiles or "")

            for lig in fix_zwitterions_ligands:
                for atom in lig.atoms or []:
                    atom.bonds = []

                    logger.debug(
                        "Re-running create_bonds_specie for ligand %s due to zwitterion correction.",
                        lig.formula,
                    )
                result = create_bonds_specie(lig)
                if not result:
                    logger.error(
                        "Error for ligand %s in re-creating bonds", lig.formula
                    )
                    self.error_create_bonds = True
                    return

                logger.debug(
                    "Bonds re-created for ligand %s after zwitterion correction.",
                    lig.formula,
                )
                self.ligand_smiles.append(lig.smiles or "")

        # Third part : adds metal-ligand bonds, metal-metal bonds, with a zero order
        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            create_metal_ligand_bonds(self)
            create_metal_metal_bonds(self)

        self.error_create_bonds = False

    def assess_errors(self):
        if self.error_get_poscharges:
            case = 5
        elif self.error_multiple_distrib:
            case = 6
        elif self.error_empty_distrib:
            case = 7
        elif self.error_assign_charge:
            case = 8
        elif self.error_create_bonds:
            case = 9
        elif self.error_get_spin:
            case = 10
        else:
            case = 0

        self.error_case = case
