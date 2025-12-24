from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.classes.atom import Atom
from cell2mol.charge_assignment import get_metal_poscharges
from cell2mol.spin import assign_spin_metal, predict_ox_state
from cell2mol.operations import compute_centroid, get_dist
from cell2mol.elementdata import ElementData
from cell2mol.coordination_sphere import define_coordination_geometry
from cell2mol.my_types import RefList, Spin, SubType
from cell2mol.connectivity import labels2formula, get_adjmatrix

if TYPE_CHECKING:
    from cell2mol.classes.group import Group

elemdatabase = ElementData()


###############
#### METAL ####
###############
class Metal(Atom):
    # Cross-reference: points to other metals in parent Molecule
    metals: RefList[Metal] = Field(default_factory=list)
    # Cross-reference: points to groups in ligands
    groups: RefList["Group"] = Field(default_factory=list)
    coord_nr: int | None = None
    coord_geometry: str | Literal["Undefined"] | None = None
    geom_deviation: float | Literal["Undefined"] | None = None
    rel_metal_radius: float | None = None
    coord_nr_with_metal_bonds: int | None = None
    coord_geometry_with_metal_bonds: str | Literal["Undefined"] | None = None
    geom_deviation_with_metal_bonds: float | Literal["Undefined"] | None = None
    metal_factor: float | None = None
    cov_factor: float | None = None
    coord_sphere: list[Atom] | None = None
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

    #######################################################
    def get_coord_sphere(self):
        if not self.check_parent("molecule"):
            return None
        mol = self.get_parent("molecule")
        pidx = self.get_parent_index("molecule")
        # if mol.adjmat is None: mol.get_adjmatrix()
        # adjmat = mol.adjmat.copy()

        ## Cordination sphere defined as a collection of atoms
        self.coord_sphere = []
        for idx, at in enumerate(mol.adjmat[pidx]):
            if at >= 1:
                self.coord_sphere.append(mol.atoms[idx])
        return self.coord_sphere

    #######################################################
    def get_coord_sphere_formula(self, debug: int = 1):
        if self.coord_sphere is None:
            self.get_coord_sphere()
        self.coord_sphere_formula = labels2formula(
            list([at.label for at in self.coord_sphere])
        )
        if debug >= 2:
            print(
                f"METAL.Get_coord_sphere_formula: molecule parent_index={self.get_parent_index('molecule')} metal={self.label} coord_sphere_formula={self.coord_sphere_formula}"
            )
        return self.coord_sphere_formula

    #######################################################
    def get_connected_groups(self, debug: int = 0):
        # metal.groups will be used for the calculation of the relative metal radius
        # and define the coordination geometry of the metal /hapicitiy/ hapttype
        if not self.check_parent("molecule"):
            return None

        refcell = self.get_parent("reference")
        geom_bond_cif = getattr(refcell, "geom_bond_cif", None)
        mol = self.get_parent("molecule")
        connected_groups = []
        for lig in mol.ligands:
            for group in lig.groups:
                if debug > 2:
                    print(group.formula, [m.atom_site_label for m in group.metals])
                for met in group.metals:
                    if self == met:
                        connected_groups.append(group)
                        if debug >= 0:
                            if self.atom_site_label is not None:
                                print(
                                    f"METAL.Get_connected_groups: Metal {self.label} ({self.atom_site_label}) is connected to group {group.formula}"
                                )
                            else:
                                print(
                                    f"METAL.Get_connected_groups: Metal {self.label} is connected to group {group.formula}"
                                )
        final_connected_groups = []
        groups_atom_site_labels = [
            [a.atom_site_label for a in g.atoms] for g in connected_groups
        ]
        # Remove duplicate groups based on atom_site_labels

        for g_labels, group in zip(groups_atom_site_labels, connected_groups):
            if not any(
                set(g_labels).issubset(set(other)) and set(g_labels) != set(other)
                for other in groups_atom_site_labels
            ):
                final_connected_groups.append(group)
        self.groups = final_connected_groups
        if debug >= 1:
            if self.atom_site_label is not None:
                print(
                    f"METAL.Get_connected_groups: {self.label} ({self.atom_site_label}) connected groups: {[g.formula for g in self.groups]}"
                )
            else:
                print(
                    f"METAL.Get_connected_groups: {self.label} connected groups: {[g.formula for g in self.groups]}"
                )
        return self.groups

    # def get_connected_groups(self, debug: int = 0):
    #     from cell2mol.connectivity import split_group

    #     # metal.groups will be used for the calculation of the relative metal radius
    #     # and define the coordination geometry of the metal /hapicitiy/ hapttype
    #     if not self.check_parent("molecule"):
    #         return None

    #     refcell = self.get_parent("reference")
    #     geom_bond_cif = getattr(refcell, "geom_bond_cif", None)
    #     mol = self.get_parent("molecule")

    #     connected_groups = []
    #     if refcell is not None and refcell.exist_cif_bond_moiety:
    #         for lig in mol.ligands:
    #             for group in lig.groups:
    #                 if debug > 2:
    #                     print(group.formula)
    #                 ligand_indices = [a.get_parent_index("ligand") for a in group.atoms]
    #                 tmplabels = []
    #                 tmpcoord = []
    #                 atom_site_labels = []

    #                 tmplabels.append(self.label)
    #                 tmpcoord.append(self.coord)
    #                 atom_site_labels.append(self.atom_site_label)

    #                 tmplabels.extend(group.labels)
    #                 tmpcoord.extend(group.coord)

    #                 atom_site_labels.extend([atom.atom_site_label for atom in group.atoms])
    #                 if debug >= 2: print(f"METAL.Get_connected_groups: {tmplabels=} {atom_site_labels=}")
    #                 isgood, tmpadjmat, tmpadjnum = get_adjmatrix_from_cif_bonds(tmplabels, tmpcoord, atom_site_labels, geom_bond_cif, metal_only=True)
    #                 if isgood :
    #                     if debug > 2: print(group.formula, tmpadjmat, tmpadjnum)
    #                     if all(tmpadjnum[1:]):
    #                         self.groups.append(group)
    #                         if debug >= 0:
    #                             print(
    #                                 f"METAL.Get_connected_groups: Metal {self.label} is connected to all atoms in {group.formula}"
    #                             )

    #                     elif any(tmpadjnum[1:]):
    #                         if debug > 1:
    #                             print(
    #                                 f"METAL.Get_connected_groups: {self.label} is connected to {group.formula} but not all atoms are connected"
    #                             )
    #                         conn_idx = [
    #                             idx for idx, num in enumerate(tmpadjnum[1:]) if num == 1
    #                         ]
    #                         conn_ligand_indices = [
    #                             ligand_indices[idx]
    #                             for idx, num in enumerate(tmpadjnum[1:])
    #                             if num == 1
    #                         ]
    #                         if debug > 1:
    #                             print(
    #                                 f"METAL.Get_connected_groups: {tmpadjnum[1:]=} {conn_idx=} {conn_ligand_indices=} {ligand_indices=}"
    #                             )
    #                         splitted_groups = split_group(
    #                             group, conn_idx, conn_ligand_indices, debug=debug
    #                         )
    #                         for g in splitted_groups:
    #                             self.groups.append(g)
    #                             if debug > 1:
    #                                 print(
    #                                     f"METAL.Get_connected_groups: {self.label} is connected to {g.formula} after split_group"
    #                                 )
    #                     else:
    #                         if debug > 1:
    #                             print(
    #                                 f"METAL.Get_connected_groups: {self.label} is not connected to {group.formula}"
    #                             )
    #     else:
    #         for lig in mol.ligands:
    #             for group in lig.groups:
    #                 if debug > 2:
    #                     print(group.formula)
    #                 ligand_indices = [a.get_parent_index("ligand") for a in group.atoms]
    #                 tmplabels = []
    #                 tmpcoord = []
    #                 tmplabels.append(self.label)
    #                 tmpcoord.append(self.coord)
    #                 tmplabels.extend(group.labels)
    #                 tmpcoord.extend(group.coord)
    #                 if debug > 2:
    #                     print(tmplabels, tmpcoord)
    #                 isgood, tmpadjmat, tmpadjnum, warning = get_adjmatrix(
    #                     tmplabels, tmpcoord, metal_only=True
    #                 )
    #                 if isgood:
    #                     if debug > 2:
    #                         print(group.formula, tmpadjmat, tmpadjnum)
    #                     if all(tmpadjnum[1:]):
    #                         if debug > 1:
    #                             print(
    #                                 f"METAL.Get_connected_groups: {self.label} is connected to {group.formula} with all atoms"
    #                             )
    #                         connected_groups.append(group)
    #                     elif any(tmpadjnum[1:]):
    #                         if debug > 1:
    #                             print(
    #                                 f"METAL.Get_connected_groups: {self.label} is connected to {group.formula} but not all atoms are connected"
    #                             )
    #                         conn_idx = [
    #                             idx for idx, num in enumerate(tmpadjnum[1:]) if num == 1
    #                         ]
    #                         conn_ligand_indices = [
    #                             ligand_indices[idx]
    #                             for idx, num in enumerate(tmpadjnum[1:])
    #                             if num == 1
    #                         ]
    #                         if debug > 1:
    #                             print(
    #                                 f"METAL.Get_connected_groups: {tmpadjnum[1:]=} {conn_idx=} {conn_ligand_indices=} {ligand_indices=}"
    #                             )
    #                         splitted_groups = split_group(
    #                             group, conn_idx, conn_ligand_indices, debug=debug
    #                         )
    #                         for g in splitted_groups:
    #                             connected_groups.append(g)
    #                             if debug > 1:
    #                                 print(
    #                                     f"METAL.Get_connected_groups: {self.label} is connected to {g.formula} after split_group"
    #                                 )
    #                     else:
    #                         if debug > 1:
    #                             print(
    #                                 f"METAL.Get_connected_groups: {self.label} is not connected to {group.formula}"
    #                             )

    #     final_connected_groups = []
    #     groups_atom_site_labels = [
    #         [a.atom_site_label for a in g.atoms] for g in connected_groups
    #     ]
    #     # Remove duplicate groups based on atom_site_labels

    #     for g_labels, group in zip(groups_atom_site_labels, connected_groups):
    #         if not any(set(g_labels).issubset(set(other)) and set(g_labels) != set(other) for other in groups_atom_site_labels):
    #             final_connected_groups.append(group)
    #     self.groups = final_connected_groups
    #     if debug >= 1:
    #         print(
    #             f"METAL.Get_connected_groups: {self.label} ({self.atom_site_label}) connected groups: {[g.formula for g in self.groups]}"
    #         )
    #     return self.groups

    #######################################################
    def get_relative_metal_radius(self, debug: int = 0):
        if self.groups is None:
            self.get_connected_groups(debug=debug)
        diff_list = []
        for group in self.groups:
            if not group.is_haptic:
                for atom in group.atoms:
                    diff = round(
                        float(
                            get_dist(self.coord, atom.coord)
                            - elemdatabase.CovalentRadius3[atom.label]
                        ),
                        3,
                    )
                    diff_list.append(diff)
            else:
                haptic_center_label = "C"
                haptic_center_coord = compute_centroid(
                    np.array([atom.coord for atom in group.atoms])
                )
                diff = round(
                    float(
                        get_dist(self.coord, haptic_center_coord)
                        - elemdatabase.CovalentRadius3[haptic_center_label]
                    ),
                    3,
                )
                diff_list.append(diff)
        average = round(float(np.average(diff_list)), 3)

        if debug > 1:
            print(f"METAL.Get_relative_metal_radius: {diff_list=}")
            print(f"METAL.Get_relative_metal_radius: {average=}")

        self.rel_metal_radius = round(
            average / elemdatabase.CovalentRadius3[self.label], 3
        )

        return self.rel_metal_radius

    #######################################################
    def get_connected_metals(self, debug: int = 2):
        self.metals = []
        mol = self.get_parent("molecule")
        refcell = self.get_parent("reference")
        if refcell is not None and refcell.exist_cif_bond_moiety:
            pidx = self.get_parent_index("molecule")
            # print(f"METAL.Get_connected_metals: {self.label} {pidx=} {mol.metals=}")
            for met in mol.metals:
                if met == self:
                    continue
                met_idx = met.get_parent_index("molecule")
                if mol.adjmat[pidx, met_idx] == 1:
                    if debug > 1:
                        print(
                            f"METAL.Get_connected_metals: {self.label} ({self.atom_site_label}) is connected to {met.label} ({met.atom_site_label})"
                        )
                    self.metals.append(met)
                else:
                    if debug > 1:
                        print(
                            f"METAL.Get_connected_metals: {self.label} ({self.atom_site_label}) is NOT connected to {met.label} ({met.atom_site_label})"
                        )
        else:
            for met in mol.metals:
                if met == self:
                    continue
                tmplabels = []
                tmpcoord = []
                tmplabels.append(self.label)
                tmpcoord.append(self.coord)
                tmplabels.append(met.label)
                tmpcoord.append(met.coord)

                if debug > 2:
                    print(tmplabels, tmpcoord)

                isgood, tmpadjmat, tmpadjnum, warning = get_adjmatrix(
                    tmplabels, tmpcoord, metal_only=True
                )
                if isgood:
                    if debug > 2:
                        print(met.label, tmpadjmat, tmpadjnum)
                    if all(tmpadjnum[1:]):
                        self.metals.append(met)
                    if debug > 1:
                        print(
                            f"METAL.Get_connected_metals: {self.label} ({self.atom_site_label}) is connected to {met.label} ({met.atom_site_label})"
                        )
                else:
                    if debug > 1:
                        print(
                            f"METAL.Get_connected_metals: {self.label} ({self.atom_site_label}) is NOT connected to {met.label} ({met.atom_site_label})"
                        )
        return self.metals

    #######################################################
    def get_coordination_geometry(self: object, debug: int = 0):
        if debug >= 1:
            if self.atom_site_label is not None:
                print(
                    f"\nMETAL.Get_coord_geometry: {self.label} ({self.atom_site_label})"
                )
            else:
                print(f"\nMETAL.Get_coord_geometry: {self.label} (No atom site label)")

        coord_group = self.get_connected_groups(debug=debug)

        self.coord_nr, self.coord_geometry, self.geom_deviation = (
            define_coordination_geometry(self, coord_group, debug=debug)
        )

        if debug >= 3:
            print(f"METAL.Get_coord_geometry:\n{coord_group=}")
        if debug >= 1:
            print(f"METAL.Get_coord_geometry: coord_nr={self.coord_nr}")
        if debug >= 2:
            print(
                f"METAL.Get_coord_geometry: {self.coord_geometry=} {self.geom_deviation=}"
            )

        self.rel_metal_radius = self.get_relative_metal_radius(debug=debug)
        if debug >= 2:
            print(f"METAL.Get_coord_geometry: {self.rel_metal_radius=}")

        if self.metals is None:
            self.get_connected_metals(debug=debug)

        if len(self.metals) > 0:
            bonded_metals = self.metals
            whole_coord = coord_group + bonded_metals

            if debug >= 1:
                print(
                    f"\nMETAL.Get_coord_geometry Including metal-metal bonds for: {self.label}"
                )

            (
                self.coord_nr_with_metal_bonds,
                self.coord_geometry_with_metal_bonds,
                self.geom_deviation_with_metal_bonds,
            ) = define_coordination_geometry(self, whole_coord, debug=debug)
            if debug >= 1:
                print(
                    f"METAL.Get_coord_geometry Including metal-metal bonds {self.coord_nr_with_metal_bonds=}"
                )
            if debug >= 2:
                print(
                    f"METAL.Get_coord_geometry Including metal-metal bonds: {self.coord_geometry_with_metal_bonds=} {self.geom_deviation_with_metal_bonds=}"
                )

        return self.coord_geometry

    #######################################################
    def get_possible_cs(self, debug: int = 0):
        self.possible_cs = get_metal_poscharges(self)
        return self.possible_cs

    #######################################################
    def get_spin(self, debug: int = 0):
        self.spin = assign_spin_metal(self, debug=debug)
        if debug >= 1:
            print(
                f"GET_SPIN: Spin multiplicity of the metal {self.label} is assigned as {self.spin}"
            )

    #######################################################
    def predict_charge(self, debug: int = 0):
        self.charge_by_ML = predict_ox_state(self, debug=debug)

    #######################################################
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
