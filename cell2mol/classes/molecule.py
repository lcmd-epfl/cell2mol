from __future__ import annotations
import pickle
from typing import cast

import numpy as np
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.ligand import Ligand
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import split_species, merge_multiple_groups
from cell2mol.element_utils import (
    labels2formula,
    get_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
    POST_TRANSITION_METALS,
    METALLOIDS,
)
from cell2mol.species_collection import (
    collect_plausible_charges,
    collect_unique_species,
    map_charges_to_molecules,
)
from cell2mol.charge.specie_assigner import (
    assemble_complex_charge_state,
)
from cell2mol.charge.smiles_handler import (
    create_bonds_specie,
    create_metal_ligand_bonds,
    create_metal_metal_bonds,
)
from cell2mol.spin import assign_spin_complexes
from cell2mol.operations import extract_from_list
from cell2mol.elementdata import ElementData
from cell2mol.my_types import Format, Spin, SubType, NDArray
from cell2mol.utils import config
import logging
from pathlib import Path

try:
    from typing import Self  # py3.11+
except ImportError:
    from typing_extensions import Self  # py3.10

logger = logging.getLogger(__name__)
elemdatabase = ElementData()

# Error rules, most severe first: the first hit is the primary ``error_case``
# and every hit is kept in ``error_cases_all``, so a structure that is short of
# hydrogens on both a coordinated donor and a carbon reports code 3 without
# hiding code 4. Shared with MoleculeSet so the two cannot drift apart.
MOLECULE_ERROR_RULES = [
    ("has_isolated_H", 1),
    ("missing_H_in_Water", 2),
    ("missing_H_on_CoordDonor", 3),
    ("missing_H_in_Carbon", 4),
    ("error_plausible_charges", 5),
    ("error_multiple_distrib", 6),
    ("error_empty_distrib", 7),
    ("error_assign_charge", 8),
    ("error_create_bonds", 9),
    ("error_get_spin", 10),
]


def assess_molecule_errors(obj):
    """Reduce the error flags of a molecule or molecule set to error codes.

    Sets ``error_cases_all`` to every code that fired and ``error_case`` to the
    first (most severe) of them, 0 when nothing fired.
    """
    triggered = [
        code for attr, code in MOLECULE_ERROR_RULES if getattr(obj, attr, False)
    ]
    obj.error_cases_all = triggered
    obj.error_case = triggered[0] if triggered else 0

    return obj.error_case


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
    totcharge_agree: bool | None = None

    ligand_smiles: str | list[str] | None = None
    subtype: SubType | None = Field(default="molecule")

    # Needed in interpret_molecule in process_xyz.py
    input_charge: int | None = None
    unique_species: list[Specie | Metal] | None = None
    unique_indices: list[int] | None = None
    species_list: list[Specie | Metal] | None = None
    # Plausible integer charges per specie: metal oxidation states, or a
    # ligand/molecule's total charges. NOT one entry per specie -- it is
    # unique_species first, then all of species_list, so species recur at
    # two indices. None marks a specie whose charges could not be found.
    plausible_charges: list[list[int] | None] | None = None
    error_plausible_charges: bool | None = None
    # True when this molecule is a single, dangling H/D atom.
    has_isolated_H: bool | None = None
    # Set when two entries sharing a unique_index -- i.e. copies of the SAME
    # specie -- enumerated to different charges. See
    # inconsistent_plausible_charges for the detail.
    error_inconsistent_plausible_charges: bool | None = None
    inconsistent_plausible_charges: dict[int, list[list[int] | None]] | None = None
    error_multiple_distrib: bool | None = None
    error_empty_distrib: bool | None = None
    error_assign_charge: bool | None = None
    error_create_bonds: bool | None = None
    error_get_spin: bool | None = None
    # Primary (first-matching, most severe) code, and every code that fired.
    error_case: int | None = None
    error_cases_all: list[int] | None = None

    @classmethod
    @deprecated("Use molecule() with the keyword arguments instead.")
    def from_positional(
        cls,
        labels: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
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

    @property
    def contains_metal(self) -> bool:
        """True if the molecule contains transition, alkali/alkaline-earth or
        post-transition metals."""
        return self.iscomplex or self.has_ia_iia or self.has_post_transition_metal

    def _log_metal_content(self) -> bool:
        """
        Log which metal category this molecule falls into and return whether it
        contains any metal at all.
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
            return False
        return True

    def analyze_coordination(self):
        """
        Analyze the metal coordination environment of this molecule, in place.

        Metal-free molecules return early without any analysis; molecules
        containing transition, alkali/alkaline-earth, or post-transition metals
        go through the full analysis below.

        For every metal:
            - ``get_connected_metals`` -> ``metal.metals``
            - ``get_connected_nonmetal_atoms`` -> ``metal.connected_nonmetal_atoms``
            - ``get_connected_groups`` -> ``metal.groups``
            - ``get_coordination_geometry`` -> ``coord_nr``, ``coord_geometry``,
              ``geom_deviation`` and the relative-metal-radius values
            - ``get_coord_sphere_formula`` -> ``coord_sphere_formula``

        Coordinating groups are then mapped onto ligands
        (``map_metal_groups_to_ligands``) and groups bridging the same ligand are
        merged (``merge_connected_groups``).

        Hapticity/denticity of every ligand of this molecule are then determined.

        Returns:
            None. All results are stored on the metal, group and ligand objects.
        """

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

    def detect_special_moieties(self):
        """
        Screen this molecule for structural motifs that need special treatment.

        In a metal-containing molecule the motifs sit on the ligands, so the
        screen is delegated to each of them (which adds the nitrosyl check). A
        metal-free molecule is screened as a whole by the Specie implementation.

        Returns:
            None. Flags are stored on the molecule or on the ligand objects.
        """

        if not self.contains_metal:
            super().detect_special_moieties()
            return

        for lig in self.ligands or []:
            lig.detect_special_moieties()

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
                            "Mapped %s (haptic_type: %s) from Metal %s to Ligand %s",
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

    def check_hydrogens(self):
        """Screen this molecule for missing hydrogens.

        Extends ``Specie.check_hydrogens`` (which already dispatches a complex
        to its ligands) with the dangling-hydrogen case, so that a molecule and
        a ``MoleculeSet`` report the same flags to ``assess_errors``.
        """
        self.has_isolated_H = self.natoms == 1 and self.labels[0] in {"H", "D"}
        if self.has_isolated_H:
            logger.warning("  Isolated hydrogen found %s", self.labels[0])

        return Specie.check_hydrogens(self)

    def get_unique_species(self):
        """Deduplicate this molecule's own species.

        A single molecule is just a collection of one, so this delegates to the
        shared routine. Use ``MoleculeSet`` when several molecules have to share
        one set of unique species -- deduplicating them one at a time would give
        each its own indices and its own charge decision.
        """
        logger.info("Getting unique species in molecule: %s", self.formula)

        (
            self.unique_species,
            self.unique_indices,
            self.species_list,
        ) = collect_unique_species([self])

        return self.unique_species

    def get_plausible_charges(self):
        if self.unique_species is None:
            self.get_unique_species()

        (
            self.plausible_charges,
            self.error_plausible_charges,
            self.inconsistent_plausible_charges,
        ) = collect_plausible_charges(
            self.unique_species, self.species_list, skip_missing_h=True
        )
        self.error_inconsistent_plausible_charges = bool(
            self.inconsistent_plausible_charges
        )

    def assign_charges(self):
        logger.info("Assigning charges for molecule: %s", self.formula)

        self.error_assign_charge = map_charges_to_molecules(self.unique_species, [self])

        self.create_bonds()

        if self.contains_metal and not self.error_create_bonds:
            assemble_complex_charge_state(self)
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
        elif not self.contains_metal:
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

        # Second part: Complex molecule, add bonds for ligands.
        # SMILES + zwitterion corrections are already baked into each ligand's
        # specie rdkit_obj/smiles (ChargeState), so this only builds the bond
        # graph -- no separate SMILES correction / zwitterion re-creation needed.
        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            self.ligand_smiles = []
            for lig in self.ligands or []:
                if lig.smiles is None:
                    logger.error(
                        "Ligand %s has no SMILES after charge assignment. Cannot create bonds.",
                        lig.formula,
                    )
                # Creates bonds between ligand.atoms, using the ligand.rdkit_object
                result = create_bonds_specie(lig)
                if not result:
                    logger.error("Error for ligand %s", lig.formula)
                    self.error_create_bonds = True
                    return  # Exit if creating bonds fails for any ligand

                logger.debug("Bonds created for ligand %s", lig.formula)
                self.ligand_smiles.append(lig.smiles or "")

        # Third part : adds metal-ligand bonds, metal-metal bonds, with a zero order
        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            create_metal_ligand_bonds(self)
            create_metal_metal_bonds(self)

        self.error_create_bonds = False

    def assess_errors(self):
        return assess_molecule_errors(self)
