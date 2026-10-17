from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.classes.atom import Atom
from cell2mol.spin import predict_metal_spin
from cell2mol.operations import get_dist
from cell2mol.elementdata import ElementData
from cell2mol.coordination_sphere import (
    build_coordination_groups,
    determine_coordination_geometry,
)
from cell2mol.my_types import CoordinationMode, RefList, Spin, SubType, NDArray
from cell2mol.connectivity import build_adjacency
from cell2mol.element_utils import symbols2formula
from cell2mol.utils import config

if TYPE_CHECKING:
    from cell2mol.classes.group import Group
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.species import Species
import logging

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


class Metal(Atom):
    """Metal atom with coordination geometry, oxidation states, and spin."""

    # References to connected atoms and groups in the parent molecule.
    metals: RefList[Metal] | None = Field(default=None)
    connected_nonmetal_atoms: RefList[Atom] | None = Field(default=None)
    groups: RefList[Group] | None = Field(default=None)
    coord_nr: int | None = None
    coord_geometry: str | None = None
    geom_deviation: float | Literal["Undefined"] | None = None
    rel_metal_radius: float | None = None
    coord_nr_with_metal_bonds: int | None = None
    coord_geometry_with_metal_bonds: str | None = None
    geom_deviation_with_metal_bonds: float | Literal["Undefined"] | None = None
    coord_sphere_atoms: list[Atom] | None = None
    coord_sphere_formula: str | None = None
    unique_index: int | None = None
    plausible_os: list[int] | None = None
    spin: Spin | None = None
    valence_elec: int | None = None
    removed_from_coordination: list[Atom] | None = None

    subtype: SubType = Field(default="metal")

    @property
    def coordination(self) -> dict[int, CoordinationMode]:
        """Inverse ligand view keyed by position in Molecule.ligands.

        Donor indices remain local to the corresponding ligand. No inverse
        state is stored. Missing parent context yields an empty mapping.
        """
        mol = cast("Molecule | None", self.get_parent("molecule"))
        metal_index = self.get_parent_index("molecule")
        if mol is None or metal_index is None:
            return {}
        result: dict[int, CoordinationMode] = {}
        for ligand_index, ligand in enumerate(mol.ligands or []):
            mode = ligand.coordination.get(metal_index)
            if mode is not None:
                result[ligand_index] = mode
        return result

    @property
    def coordinated_ligands(self) -> list[int]:
        """Positions of coordinated ligands in Molecule.ligands."""
        return list(self.coordination)

    def _coordination_atom_indices(self, haptic: bool) -> list[int]:
        mol = cast("Molecule | None", self.get_parent("molecule"))
        if mol is None:
            return []
        indices: set[int] = set()
        for ligand_index, mode in self.coordination.items():
            ligand = (mol.ligands or [])[ligand_index]
            atom_map = ligand.get_parent_indices("molecule")
            if atom_map is None:
                raise ValueError(
                    "Coordinated ligand requires parent molecule atom indices"
                )
            local = (
                [index for site in mode["eta_groups"] for index in site]
                if haptic
                else mode["kappa_atoms"]
            )
            indices.update(atom_map[index] for index in local)
        return sorted(indices)

    @property
    def kappa_atoms(self) -> list[int]:
        """Molecule atom indices of localized ligand donors only."""
        return self._coordination_atom_indices(haptic=False)

    @property
    def eta_atoms(self) -> list[int]:
        """Molecule atom indices of haptic ligand atoms only."""
        return self._coordination_atom_indices(haptic=True)

    @property
    def coordinated_atoms(self) -> list[int]:
        """Union of kappa and eta ligand atoms; excludes metal-metal bonds.

        Unlike coord_sphere_atoms, this returns indices, not Atom objects.
        """
        return sorted(set(self.kappa_atoms) | set(self.eta_atoms))

    @property
    def coordination_number_atoms(self) -> int:
        """Unique ligand atom contacts, counting all eta atoms."""
        return len(self.coordinated_atoms)

    @property
    def coordination_number_sites(self) -> int:
        """Kappa donors plus eta sites; excludes metal-metal bonds."""
        return sum(
            len(mode["kappa_atoms"]) + len(mode["eta_groups"])
            for mode in self.coordination.values()
        )

    def compute_valence_electron_count(self, m_ox: int):
        """Compute and store the valence electron count for an oxidation state."""
        v_elec = elemdatabase.valenceelectrons[self.label] - m_ox
        if v_elec >= 0:
            self.valence_elec = v_elec
        else:
            self.valence_elec = elemdatabase.elementgroup[self.label] - m_ox

        return self.valence_elec

    def get_connected_metals(self, use_bond_info: bool | None = None):
        """Cache bonded metals using metal_contact_matrix or pairwise adjacency calculations."""
        if self.metals is not None:
            return self.metals

        mol = cast("Molecule | None", self.get_parent("molecule"))
        if mol is None:
            logger.debug("No parent molecule found, skipping metal connectivity check")
            return []

        self.metals = []
        self_mol_idx = self.get_parent_index("molecule")
        assert self_mol_idx is not None
        if mol.metal_contact_matrix is not None:
            mol_madjmat = mol.metal_contact_matrix
            for met in mol.metals or []:
                if met is self:
                    continue
                met_mol_idx = met.get_parent_index("molecule")
                assert met_mol_idx is not None
                if mol_madjmat[self_mol_idx][met_mol_idx] >= 1:
                    self.metals.append(met)
                    logger.debug(
                        "Metal %s%s is connected to Metal %s%s (existing mol.adjacency_matrix)",
                        self.label,
                        f" ({self.atom_site_label})" if self.atom_site_label else "",
                        met.label,
                        f" ({met.atom_site_label})" if met.atom_site_label else "",
                    )
            return self.metals

        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
        cov_factor = self.cov_factor
        metal_factor = self.metal_factor

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        for met in mol.metals or []:
            if met is self:
                continue

            tmplabels = [self.label, met.label]
            tmpcoord = np.asarray([self.coord, met.coord])
            atom_site_labels = (
                [self.atom_site_label, met.atom_site_label]
                if self.atom_site_label is not None and met.atom_site_label is not None
                else None
            )

            tmp_adjmat = build_adjacency(
                labels=tmplabels,
                positions=tmpcoord,
                atom_site_labels=atom_site_labels,
                bond_data=bond_data,
                use_bond_info=use_bond_info,
                cov_factor=cov_factor,
                metal_factor=metal_factor,
                metal_only=True,
            )
            if tmp_adjmat is None:
                continue
            else:
                if tmp_adjmat[0, 1] >= 1:
                    self.metals.append(met)
                    logger.debug(
                        "Metal %s%s is connected to Metal %s%s (newly computed, use_bond_info=%s)",
                        self.label,
                        f" ({self.atom_site_label})" if self.atom_site_label else "",
                        met.label,
                        f" ({met.atom_site_label})" if met.atom_site_label else "",
                        use_bond_info,
                    )
        return self.metals

    def get_connected_nonmetal_atoms(self):
        """Cache nonmetal atoms connected through the parent's metal_contact_matrix."""
        if self.connected_nonmetal_atoms is not None:
            return self.connected_nonmetal_atoms

        self.connected_nonmetal_atoms = []
        if not self.check_parent("molecule"):
            logger.debug("No parent molecule found, skipping atom connectivity check")
            return self.connected_nonmetal_atoms

        mol = cast("Species", self.get_parent("molecule"))
        self_mol_idx = self.get_parent_index("molecule")
        assert mol is not None
        assert self_mol_idx is not None
        if mol.metal_contact_matrix is None:
            logger.debug(
                "No mol.metal_contact_matrix found, skipping atom connectivity check"
            )
            return self.connected_nonmetal_atoms

        mol_madjmat = mol.metal_contact_matrix
        assert mol.atoms is not None
        for idx, val in enumerate(mol_madjmat[self_mol_idx]):
            if idx == self_mol_idx or val < 1:
                continue

            atom = mol.atoms[idx]

            if atom.subtype == "metal":
                continue
            self.connected_nonmetal_atoms.append(atom)
            logger.debug(
                "Metal %s%s is connected to Atom %s%s (existing mol.adjacency_matrix)",
                self.label,
                f" ({self.atom_site_label})" if self.atom_site_label else "",
                atom.label,
                f" ({atom.atom_site_label})" if atom.atom_site_label else "",
            )
        logger.debug(
            "Total connected non-metal atoms: %s", len(self.connected_nonmetal_atoms)
        )
        return self.connected_nonmetal_atoms

    def get_connected_groups(self) -> list[Group] | None:
        """Refine and cache coordinated groups; return None without a parent."""
        if not self.check_parent("molecule"):
            return None

        if self.groups is None:
            result = build_coordination_groups(self)
            self.groups = result

        return self.groups

    def determine_coordination_geometry(self):
        """Determine and store coordination geometry, also including bonded metals."""
        logger.debug(
            "Define coordination geometry of Metal %s%s",
            self.label,
            f" ({self.atom_site_label})" if self.atom_site_label else "",
        )

        coord_groups: list[Group | Metal] = list(self.get_connected_groups() or [])

        (self.coord_nr, self.coord_geometry, self.geom_deviation) = cast(
            "tuple[Any, ...]",
            determine_coordination_geometry(self, coord_groups),
        )

        if self.metals is None:
            self.get_connected_metals()

        if len(self.metals or []) > 0:
            connected_metals = self.metals or []
            whole_coord = coord_groups + connected_metals

            logger.debug("Including metal-metal bonds for: %s", self.label)

            (
                self.coord_nr_with_metal_bonds,
                self.coord_geometry_with_metal_bonds,
                self.geom_deviation_with_metal_bonds,
            ) = cast(
                "tuple[Any, ...]",
                determine_coordination_geometry(self, whole_coord),
            )

        return self.coord_geometry

    def get_coordination_sphere_atoms(self):
        """Cache neighbors from the parent's adjacency_matrix; return None without a parent."""
        if not self.check_parent("molecule"):
            logger.warning(f"Metal {self.label} has no parent molecule reference.")
            return None

        if self.coord_sphere_atoms is None:
            mol = cast("Species", self.get_parent("molecule"))
            midx = self.get_parent_index("molecule")

            if (
                mol is not None
                and mol.atoms is not None
                and mol.adjacency_matrix is not None
                and midx is not None
            ):
                self.coord_sphere_atoms = [
                    mol.atoms[i]
                    for i, val in enumerate(mol.adjacency_matrix[midx])
                    if val >= 1
                ]
            else:
                self.coord_sphere_atoms = []

        return self.coord_sphere_atoms

    def build_coordination_sphere_formula(self):
        """Build and store the coordination sphere formula from atom labels."""
        if self.coord_sphere_atoms is None:
            self.get_coordination_sphere_atoms()

        atom_labels = [at.label for at in self.coord_sphere_atoms or []]
        self.coord_sphere_formula = symbols2formula(atom_labels)

        return self.coord_sphere_formula

    def compute_relative_metal_radius(self) -> float:
        """Compute relative radius, weighting each haptic group as one donor.

        Subtract donor covalent radii from bond distances, then normalize the
        mean excess by the metal covalent radius. Return zero without donors.
        """
        if self.groups is None:
            self.get_connected_groups()

        diff_list: list[float] = []
        for group in self.groups or []:
            atom_diffs: list[float] = []
            for atom in group.atoms or []:
                distance = round(float(get_dist(list(self.coord), list(atom.coord))), 3)
                donor_radius = elemdatabase.CovalentRadius3[atom.label]
                atom_diffs.append(round(float(distance - donor_radius), 3))
            if group.is_haptic:
                if atom_diffs:
                    diff_list.append(round(float(np.mean(atom_diffs)), 3))
            else:
                diff_list.extend(atom_diffs)

        average = round(float(np.mean(diff_list)), 3) if diff_list else 0.0
        self.rel_metal_radius = round(
            average / elemdatabase.CovalentRadius3[self.label], 3
        )
        return self.rel_metal_radius

    def get_plausible_oxidation_states(self):
        """Look up and store plausible oxidation states."""
        from cell2mol.charge.oxidation_states import get_plausible_metal_os

        self.plausible_os = get_plausible_metal_os(self)
        return self.plausible_os

    def assign_spin(self):
        """Assign and store the metal spin multiplicity."""
        self.spin = predict_metal_spin(self)
        logger.info(
            "Spin multiplicity of the metal %s is assigned as %s", self.label, self.spin
        )

    def reset_charge(self) -> None:
        """Clear charge, oxidation-state candidates, spin, and valence count."""
        super().reset_charge()
        self.plausible_os = None
        self.spin = None
        self.valence_elec = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        """Return atom details, coordination formula, and oxidation-state candidates."""
        to_print = ""
        to_print += "------------- Cell2mol METAL Object --------------\n"
        to_print += Atom.__repr__(self, indirect=True)
        if self.coord_sphere_formula is not None:
            to_print += f" Coordination Sphere Formula  = {self.coord_sphere_formula}\n"
        if self.plausible_os is not None:
            to_print += f" Plausible Oxidation States   = {self.plausible_os}\n"
        to_print += "----------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use metal() with the keyword arguments instead.")
    def from_positional(
        cls,
        symbol: str,
        coord: NDArray | list[float],
        frac_coord: NDArray | list[float] | None = None,
        radius: float | None = None,
    ) -> "Metal":
        return cls(
            symbol=symbol,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radius=radius,
        )
