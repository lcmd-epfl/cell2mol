from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.classes.atom import Atom
from cell2mol.charge.charge_state_resolver import get_metal_poscharges
from cell2mol.spin import assign_spin_metal, predict_ox_state
from cell2mol.operations import compute_centroid, get_dist
from cell2mol.elementdata import ElementData
from cell2mol.coordination_sphere import (
    handle_metal_coordination,
    define_coordination_geometry,
)
from cell2mol.my_types import RefList, Spin, SubType
from cell2mol.connectivity import build_adjacency
from cell2mol.element_utils import labels2formula
from cell2mol.utils import config

if TYPE_CHECKING:
    from cell2mol.classes.group import Group
import logging

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


class Metal(Atom):
    # Cross-reference: points to other metals in parent Molecule
    metals: RefList[Metal] | None = Field(default=None)
    connected_nonmetal_atoms: RefList[Atom] | None = Field(default=None)
    # Cross-reference: points to Group objects
    groups: RefList[Group] | None = Field(default=None)
    coord_nr: int | None = None
    coord_geometry: str | Literal["Undefined"] | None = None
    geom_deviation: float | Literal["Undefined"] | None = None
    rel_metal_radius: float | None = None
    coord_nr_with_metal_bonds: int | None = None
    coord_geometry_with_metal_bonds: str | Literal["Undefined"] | None = None
    geom_deviation_with_metal_bonds: float | Literal["Undefined"] | None = None
    metal_factor: float | None = None
    cov_factor: float | None = None
    coord_sphere_atoms: list[Atom] | None = None
    coord_sphere_formula: str | None = None
    unique_index: int | None = None
    charge: int | None = None
    possible_cs: list[int] | None = None
    spin: Spin | None = None
    valence_elec: int | None = None

    subtype: SubType = Field(default="metal")

    @classmethod
    @deprecated("Use metal() with the keyword arguments instead.")
    def from_positional(
        cls, label: str, coord: list, frac_coord: list = None, radii: float = None
    ) -> None:
        return cls(label=label, coord=coord, frac_coord=frac_coord, radii=radii)

    #######################################################
    def get_valence_elec(self, m_ox: int):
        """Count valence electrons for a given transition metal and metal oxidation state"""
        v_elec = elemdatabase.valenceelectrons[self.label] - m_ox
        if v_elec >= 0:
            self.valence_elec = v_elec
        else:
            self.valence_elec = elemdatabase.elementgroup[self.label] - m_ox

        return self.valence_elec

    def get_connected_metals(self, use_bond_info: bool | None = None):
        if getattr(self, "metals", None) is not None:
            return self.metals

        self.metals = []
        mol = self.get_parent("molecule")
        self_mol_idx = self.get_parent_index("molecule")
        if mol.madjmat is not None:
            mol_madjmat = mol.madjmat
            for met in mol.metals:
                if met is self:
                    continue
                met_mol_idx = met.get_parent_index("molecule")
                if mol_madjmat[self_mol_idx][met_mol_idx] >= 1:
                    self.metals.append(met)
                    logger.debug(
                        "Metal %s%s is connected to Metal %s%s (existing mol.adjmat)",
                        self.label,
                        f" ({self.atom_site_label})" if self.atom_site_label else "",
                        met.label,
                        f" ({met.atom_site_label})" if met.atom_site_label else "",
                    )
            return self.metals

        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
        cov_factor = getattr(self, "cov_factor", config.COV_FACTOR)
        metal_factor = getattr(self, "metal_factor", config.METAL_FACTOR)

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        for met in mol.metals:
            if met is self:
                continue

            tmplabels = [self.label, met.label]
            tmpcoord = [self.coord, met.coord]
            atom_site_labels = [self.atom_site_label, met.atom_site_label]

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
        if getattr(self, "connected_nonmetal_atoms", None) is not None:
            return self.connected_nonmetal_atoms

        self.connected_nonmetal_atoms = []
        if not self.check_parent("molecule"):
            logger.debug("No parent molecule found, skipping atom connectivity check")
            return self.connected_nonmetal_atoms

        mol = self.get_parent("molecule")
        self_mol_idx = self.get_parent_index("molecule")
        if mol.madjmat is None:
            logger.debug("No mol.madjmat found, skipping atom connectivity check")
            return self.connected_nonmetal_atoms

        mol_madjmat = mol.madjmat
        for idx, val in enumerate(mol_madjmat[self_mol_idx]):
            if idx == self_mol_idx or val < 1:
                continue

            atom = mol.atoms[idx]

            if atom.subtype == "metal":
                continue
            self.connected_nonmetal_atoms.append(atom)
            logger.debug(
                "Metal %s%s is connected to Atom %s%s (existing mol.adjmat)",
                self.label,
                f" ({self.atom_site_label})" if self.atom_site_label else "",
                atom.label,
                f" ({atom.atom_site_label})" if atom.atom_site_label else "",
            )
        logger.debug(
            "Total connected non-metal atoms: %s", len(self.connected_nonmetal_atoms)
        )
        return self.connected_nonmetal_atoms

    def get_connected_groups(self):
        """
        Lazy load and Triggers the metal coordination refinement process
        and stores the resulting list of coordinated Group objects.
        """
        if not self.check_parent("molecule"):
            return None

        # Check if already computed
        current_groups = getattr(self, "groups", None)
        if current_groups is None:
            # handle_metal_coordination returns a list of Group objects
            result = handle_metal_coordination(self)
            self.groups = result

        return self.groups

    def get_coordination_geometry(self: object):
        logger.debug(
            "Define coordination geometry of Metal %s%s",
            self.label,
            f" ({self.atom_site_label})" if self.atom_site_label else "",
        )

        coord_groups = self.get_connected_groups()

        (self.coord_nr, self.coord_geometry, self.geom_deviation) = (
            define_coordination_geometry(self, coord_groups)
        )

        self.rel_metal_radius = self.get_relative_metal_radius()

        if self.metals is None:
            self.get_connected_metals()

        if len(self.metals) > 0:
            connected_metals = self.metals
            whole_coord = coord_groups + connected_metals

            logger.debug("Including metal-metal bonds for: %s", self.label)

            (
                self.coord_nr_with_metal_bonds,
                self.coord_geometry_with_metal_bonds,
                self.geom_deviation_with_metal_bonds,
            ) = define_coordination_geometry(self, whole_coord)

        return self.coord_geometry

    def get_coord_sphere_atoms(self):
        """
        Identifies atoms in the first coordination sphere of the metal.
        Uses the adjacency matrix from the parent molecule to find direct bonds.

        Returns:
            List[Atom]: A list of Atom objects directly bonded to this metal.
        """
        # Safety check: Ensure parent molecule existence
        if not self.check_parent("molecule"):
            logger.warning(f"Metal {self.label} has no parent molecule reference.")
            return None

        # Use cached value if exists, otherwise compute it
        if getattr(self, "coord_sphere_atoms", None) is None:
            mol = self.get_parent("molecule")
            midx = self.get_parent_index("molecule")

            # Extract atoms from adjacency matrix
            self.coord_sphere_atoms = (
                [mol.atoms[i] for i, val in enumerate(mol.adjmat[midx]) if val >= 1]
                if mol
                else []
            )

        return self.coord_sphere_atoms

    def get_coord_sphere_formula(self):
        """
        Constructs the chemical formula for the first coordination sphere.
        Returns:
            str: The chemical formula of the coordination sphere.
        """
        # Ensure coordination atoms are identified first
        if getattr(self, "coord_sphere_atoms", None) is None:
            self.get_coord_sphere_atoms()

        # Extract labels and generate the formula
        atom_labels = [at.label for at in self.coord_sphere_atoms]
        self.coord_sphere_formula = labels2formula(atom_labels)

        return self.coord_sphere_formula

    def get_relative_metal_radius(self):
        if self.groups is None:
            self.get_connected_groups()
        diff_list = []
        for group in self.groups:
            if not group.is_haptic:
                for atom in group.atoms:
                    diff = (
                        get_dist(self.coord, atom.coord)
                        - elemdatabase.CovalentRadius3[atom.label]
                    )
                    diff = round(float(diff), 3)
                    diff_list.append(diff)
            else:
                haptic_center_label = "C"
                haptic_center_coord = compute_centroid(
                    np.array([atom.coord for atom in group.atoms])
                )
                diff = (
                    get_dist(self.coord, haptic_center_coord)
                    - elemdatabase.CovalentRadius3[haptic_center_label]
                )
                diff = round(float(diff), 3)
                diff_list.append(diff)
        if len(diff_list) > 0:
            average = round(float(np.mean(diff_list)), 3)
        else:
            average = 0.0

        logger.debug("diff_list(distance-covalent_radius)=%s", diff_list)
        logger.debug("average=%s", average)

        self.rel_metal_radius = round(
            average / elemdatabase.CovalentRadius3[self.label], 3
        )

        logger.info(
            "rel_metal_radius=%s for Metal %s%s",
            self.rel_metal_radius,
            self.label,
            f" ({self.atom_site_label})" if self.atom_site_label else "",
        )
        return self.rel_metal_radius

    def get_possible_cs(self):
        self.possible_cs = get_metal_poscharges(self)
        return self.possible_cs

    def get_spin(self):
        self.spin = assign_spin_metal(self)
        logger.info(
            "Spin multiplicity of the metal %s is assigned as %s", self.label, self.spin
        )

    def predict_charge(self):
        # TODO: integrate metal OS prediction model here
        self.charge_by_ML = predict_ox_state(self)

    def reset_charge(self):
        Atom.reset_charge(
            self
        )  ## First uses the generic atom class function for itself
        if self.poscharges is not None:
            delattr(self, "poscharge")

    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self):
        to_print = ""
        to_print += "------------- Cell2mol METAL Object --------------\n"
        to_print += Atom.__repr__(self, indirect=True)
        if self.coord_sphere_formula is not None:
            to_print += f" Coordination Sphere Formula  = {self.coord_sphere_formula}\n"
        if self.possible_cs is not None:
            to_print += f" Possible Charges             = {self.possible_cs}\n"
        to_print += "----------------------------------------------------\n"
        return to_print
