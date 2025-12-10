from __future__ import annotations
from typing import Annotated
from typing_extensions import deprecated
import numpy as np

from pydantic import Field, PlainSerializer
from cell2mol.classes.atom import Atom
from cell2mol.classes.group import Group
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import (
    get_adjmatrix,
    get_adjmatrix_from_cif_bonds,
    check_blocklist,
    split_species,
    split_group,
)
from cell2mol.other import extract_from_list, get_angle
from cell2mol.elementdata import ElementData
from cell2mol.my_types import (
    HapticType,
    SubType,
    NOType,
)
from cell2mol.utils.pydantic import serialize_circular_references

elemdatabase = ElementData()


################
#### LIGAND ####
################
class Ligand(Specie):
    model_config = {"arbitrary_types_allowed": True}

    NO_type: NOType | None = None
    # Cross-reference: points to atoms already in self.atoms
    connected_atoms: Annotated[
        list[Atom | str] | None, PlainSerializer(serialize_circular_references)
    ] = Field(default=None)
    connected_idx: list[int] | None = None
    denticity: int | None = None
    # Ownership: groups are children of this ligand
    groups: list[Group] | None = Field(default=None)
    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    is_nitrosyl: bool | None = None
    is_silylyne: bool | None = None
    # Cross-reference: points to metals in parent Molecule
    metals: Annotated[
        list[Metal | str] | None, PlainSerializer(serialize_circular_references)
    ] = Field(default=None)
    unique_index: int | None = None

    subtype: SubType = Field(default="ligand")

    @classmethod
    @deprecated("Use ligand(**kwargs) with keyword arguments")
    def from_positional(
        cls, labels: list, coord: list, frac_coord: list = None, radii: list = None
    ) -> None:
        return cls(labels=labels, coord=coord, frac_coord=frac_coord, radii=radii)
        # self.evaluate_as_nitrosyl() ### move to the split_complexes function

    #######################################################
    def __repr__(self):
        to_print = ""
        to_print += "------------- Cell2mol LIGAND Object --------------\n"
        to_print += Specie.__repr__(self, indirect=True)
        if self.groups is not None:
            to_print += f" Number of Groups             = {len(self.groups)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    #######################################################
    def get_connected_metals(self, debug: int = 0):
        self.metals = []
        refcell = self.get_parent("reference")
        geom_bond_cif = getattr(refcell, "geom_bond_cif", None)
        mol = self.get_parent("molecule")

        if refcell is not None and refcell.exist_cif_bond_moiety:
            for met in mol.metals:
                tmplabels = self.labels.copy()
                tmpcoord = self.coord.copy()
                atom_site_labels = [atom.atom_site_label for atom in self.atoms]
                tmplabels.append(met.label)
                tmpcoord.append(met.coord)
                atom_site_labels.append(met.atom_site_label)

                isgood, tmpadjmat, tmpadjnum = get_adjmatrix_from_cif_bonds(
                    tmplabels,
                    tmpcoord,
                    atom_site_labels,
                    geom_bond_cif,
                    metal_only=True,
                )
                if isgood and any(tmpadjnum) > 0:
                    self.metals.append(met)
                    if debug >= 0:
                        print(
                            f"LIGAND.Get_connected_metals: {self.formula} is connected to {met.label}"
                        )
        else:
            for met in mol.metals:
                tmplabels = self.labels.copy()
                tmpcoord = self.coord.copy()
                tmplabels.append(met.label)
                tmpcoord.append(met.coord)
                isgood, tmpadjmat, tmpadjnum, warning = get_adjmatrix(
                    tmplabels, tmpcoord, metal_only=True
                )
                if isgood and any(tmpadjnum) > 0:
                    self.metals.append(met)

        return self.metals

    #######################################################
    def evaluate_as_nitrosyl(self, debug: int = 0):
        self.is_nitrosyl = False
        if self.natoms == 2 and "N" in self.labels and "O" in self.labels:
            self.is_nitrosyl = True
            self.get_nitrosyl_geom(debug=debug)
        return self.is_nitrosyl

    #######################################################
    def get_nitrosyl_geom(self: object, thres: float = 160, debug: int = 0) -> str:
        # Function that determines whether the M-N-O angle of a Nitrosyl "ligand" is "Bent" or "Linear"
        # Each case is treated differently
        #:return NO_type: "Linear" or "Bent"
        if self.atoms is None:
            self.set_atoms()
        if self.metals is None:
            self.get_connected_metals()

        for idx, a in enumerate(self.atoms):
            if a.label == "N":
                central = a.coord.copy()
            if a.label == "O":
                extreme = a.coord.copy()

        dist = []
        for idx, met in enumerate(self.metals):
            metal = np.array(met.coord)
            dist.append(np.linalg.norm(central - metal))
        tgt = np.argmin(dist)
        metal = self.metals[tgt].coord.copy()
        if debug >= 2:
            print("LIGAND.GET_NITRO_GEOM: coords:", central, extreme, metal)

        vector1 = np.subtract(np.array(central), np.array(extreme))
        vector2 = np.subtract(np.array(central), np.array(metal))
        if debug >= 2:
            print("LIGAND.GET_NITRO_GEOM: NITRO Vectors:", vector1, vector2)

        angle = get_angle(vector1, vector2)
        if debug >= 2:
            print("NITRO ANGLE:", angle, np.degrees(angle))

        if np.degrees(angle) > float(thres):
            self.NO_type = "Linear"
        else:
            self.NO_type = "Bent"

        return self.NO_type

    #######################################################
    def get_connected_idx(self, debug: int = 0):
        ## Remember madjmat should not be computed at the ligand level. Since the metal is not there.
        ## Now we operate at the molecular level. We get the parent molecule, and the indices of the ligand atoms in the molecule
        self.connected_idx = []
        if self.madjnum is None:
            self.inherit_adjmatrix("molecule")
        if debug > 2:
            print(
                f"LIGAND.GET_CONNECTED_IDX: {self.formula} {self.madjmat=} {self.madjnum=}"
            )
        for idx, con in enumerate(self.madjnum):
            if con > 0:
                self.connected_idx.append(idx)
        return self.connected_idx

    #######################################################
    def get_connected_atoms(self, debug: int = 0):
        if self.atoms is None:
            self.set_atoms()
        if self.connected_idx is None:
            self.get_connected_idx()
        self.connected_atoms = []
        for idx, at in enumerate(self.atoms):
            if idx in self.connected_idx and at.mconnec > 0:
                self.connected_atoms.append(at)
            elif idx in self.connected_idx and at.mconnec == 0:
                print("WARNING: Atom appears in connected_idx, but has mconnec=0")
        return self.connected_atoms

    #######################################################
    def check_coordination(self, debug: int = 0):
        if self.groups is None:
            self.split_ligand(debug=debug)
        for g in self.groups:
            if g.checked_coordination is None:
                g.check_coordination(debug=debug)

    #######################################################
    def get_denticity(self, debug: int = 0):
        if self.groups is None:
            self.split_ligand(debug=debug)
        if debug > 1:
            print(
                f"LIGAND.Get_denticity: checking connectivity of ligand {self.formula}"
            )
        if debug > 1:
            print(
                f"LIGAND.Get_denticity: initial connectivity is {len(self.connected_idx)}"
            )
        self.denticity = 0
        for g in self.groups:
            # if debug > 0: print(f"LIGAND.Get_denticity: checking denticity of group \n{g}\n{g.madjnum=}\n{g.madjmat=}")
            self.denticity += g.get_denticity(
                debug=debug
            )  ## A check is also performed at the group level
        if debug > 0:
            print(
                f"LIGAND.Get_denticity: final connectivity of ligand {self.formula} is {self.denticity}"
            )
        return self.denticity

    #######################################################
    def split_ligand(self, debug: int = 0):
        def is_single_sublist(intermediate_list):
            return (
                isinstance(intermediate_list, list)
                and len(intermediate_list) == 1
                and isinstance(intermediate_list[0], list)
            )

        if hasattr(self, "cov_factor"):
            cov_factor = self.cov_factor

        refcell = self.get_parent("reference")
        geom_bond_cif = getattr(refcell, "geom_bond_cif", None)

        if debug > 0:
            print(f"\nLIGAND.SPLIT_LIGAND: splitting {self.formula} into groups")

        # Split the "ligand to obtain the groups
        self.groups = []

        # Identify Connected and Unconnected atoms (to the metal)
        if self.connected_idx is None:
            self.get_connected_idx()
        connected_idx = self.connected_idx

        if debug >= 2:
            print(f"\tLIGAND.SPLIT_LIGAND: {self.indices=}")
            print(f"\tLIGAND.SPLIT_LIGAND: {connected_idx=}")
        conn_labels = extract_from_list(connected_idx, self.labels, dimension=1)
        conn_coord = extract_from_list(connected_idx, self.coord, dimension=1)
        if self.frac_coord is not None:
            conn_frac_coord = extract_from_list(
                connected_idx, self.frac_coord, dimension=1
            )
        conn_radii = extract_from_list(connected_idx, self.radii, dimension=1)
        conn_atoms = extract_from_list(connected_idx, self.atoms, dimension=1)
        if self.atom_site_labels is not None:
            conn_atom_site_labels = extract_from_list(
                connected_idx, self.atom_site_labels, dimension=1
            )
        else:
            conn_atom_site_labels = None
        if debug >= 2:
            print(f"\tLIGAND.SPLIT_LIGAND: {conn_labels=}")
        if debug >= 2:
            print(f"\tLIGAND.SPLIT_LIGAND: {conn_atom_site_labels=}")
        if (
            refcell is not None
            and refcell.exist_cif_bond_moiety
            and refcell.geom_bond_cif is not None
        ):
            blocklist = split_species(
                conn_labels,
                conn_coord,
                radii=conn_radii,
                atom_site_labels=conn_atom_site_labels,
                geom_bond_cif=refcell.geom_bond_cif,
                cov_factor=self.cov_factor,
                debug=debug,
            )
        else:
            blocklist = split_species(
                conn_labels, conn_coord, radii=conn_radii, debug=debug
            )
        if debug >= 2:
            print(f"\tLIGAND.SPLIT_LIGAND: {blocklist=}")

        blocklist = check_blocklist(conn_labels, conn_coord, blocklist)
        if debug >= 2:
            print(f"\tLIGAND.SPLIT_LIGAND: After Checking {blocklist=}")
        ## Arranges Groups
        for b in blocklist:
            if debug >= 2:
                print(f"\tLIGAND.SPLIT_LIGAND: block={b}")
            gr_indices = extract_from_list(b, connected_idx, dimension=1, debug=debug)
            if debug > 1:
                print(f"\tLIGAND.SPLIT_LIGAND: {gr_indices=}")
            gr_labels = extract_from_list(b, conn_labels, dimension=1, debug=debug)
            gr_coord = extract_from_list(b, conn_coord, dimension=1)
            if self.frac_coord is not None:
                gr_frac_coord = extract_from_list(b, conn_frac_coord, dimension=1)
            gr_radii = extract_from_list(b, conn_radii, dimension=1)
            gr_atoms = extract_from_list(b, conn_atoms, dimension=1)
            if self.atom_site_labels is not None:
                gr_atom_site_labels = extract_from_list(
                    b, conn_atom_site_labels, dimension=1
                )
            else:
                gr_atom_site_labels = None
            # Create Group Object
            if self.frac_coord is not None:
                newgroup = Group.from_positional(
                    gr_labels, gr_coord, gr_frac_coord, radii=gr_radii
                )
            else:
                newgroup = Group.from_positional(gr_labels, gr_coord, radii=gr_radii)

            # For debugging
            newgroup.origin = "split_ligand"
            # Define the ligand as parent of the group. Bottom-Up hierarchy
            newgroup.add_parent(self, indices=gr_indices)
            # Pass the ligand atoms to the groud
            newgroup.set_atoms(
                atomlist=gr_atoms,
                atom_site_labels=gr_atom_site_labels,
                geom_bond_cif=geom_bond_cif,
            )
            # Inherit the adjacencies from molecule
            newgroup.inherit_adjmatrix("ligand")
            # Associate the Groups with the Metals
            newgroup.get_connected_metals(debug=debug)
            newgroup.get_closest_metal(debug=debug)
            newgroup.get_hapticity(debug=debug)
            # if refcell.exist_cif_bond_moiety:
            #     print(f"\tSPLIT_LIGAND: skipping check_coordination for group {newgroup.formula} because it is based on CIF bond information")
            #     newgroup.checked_coordination = True
            #     newgroup.get_denticity(debug=debug)
            #     self.groups.append(newgroup)
            # else:

            (
                newgroup,
                final_group_indices,
                final_ligand_indices,
                group_metals_indices,
            ) = newgroup.check_coordination(debug=debug)
            print(f"\tLIGAND.SPLIT_LIGAND: {newgroup.formula} {newgroup.labels}")
            print(f"\tLIGAND.SPLIT_LIGAND: {final_group_indices=}")
            print(f"\tLIGAND.SPLIT_LIGAND: {final_ligand_indices=}")
            print(f"\tLIGAND.SPLIT_LIGAND: {group_metals_indices=}")
            # print(f"\tLIGAND.SPLIT_LIGAND: connected to {[newgroup.metals[kdx].atom_site_label for kdx in group_metals_indices]}")
            if not is_single_sublist(
                final_group_indices
            ):  # atoms in new group are connected to different metals
                if debug > 1:
                    print(
                        f"\tenterting SPLIT_GROUP for the GROUP {newgroup.formula} with {final_group_indices=} {[met.label for met in newgroup.metals]}"
                    )
                for kdx, (conn_idx, metal_idx) in enumerate(
                    zip(final_group_indices, group_metals_indices)
                ):
                    group_metals = [newgroup.metals[midx] for midx in metal_idx]
                    if debug > 1:
                        print(
                            f"\tenterting SPLIT_GROUP for the GROUP {newgroup.labels} with {conn_idx=} {[met.atom_site_label for met in group_metals]=}"
                        )
                    splitted_groups = split_group(
                        newgroup,
                        conn_idx,
                        final_ligand_indices[kdx],
                        group_metals,
                        debug=debug,
                    )
                    for g in splitted_groups:
                        self.groups.append(g)
            else:
                if debug > 1:
                    print(
                        f"\tGROUP {newgroup.formula} with {final_group_indices=} connected to {[met.label for met in newgroup.metals]}"
                    )
                conn_idx = final_group_indices[0]
                group_metals = [newgroup.metals[kdx] for kdx in group_metals_indices[0]]
                if len(conn_idx) == len(newgroup.atoms):
                    if debug > 1:
                        print("\tLIGAND.SPLIT_LIGAND: new group is found")
                    newgroup.get_denticity(debug=debug)
                    # Top-down hierarchy
                    self.groups.append(newgroup)
                elif len(conn_idx) == 0:
                    if debug > 1:
                        print("\tLIGAND.SPLIT_LIGAND: no group is found")
                    continue
                else:
                    if debug > 1:
                        print(
                            f"\tenterting SPLIT_GROUP for the GROUP {newgroup.formula} with {conn_idx=}"
                        )
                    splitted_groups = split_group(
                        newgroup,
                        conn_idx,
                        final_ligand_indices[0],
                        group_metals,
                        debug=debug,
                    )
                    for g in splitted_groups:
                        self.groups.append(g)
        if debug > 0:
            print(
                f"\tLIGAND.SPLIT_LIGAND: found groups {[group.formula for group in self.groups]}"
            )
        if debug > 3:
            print(f"{self.groups}")

        return self.groups

    #######################################################
    def get_hapticity(self, debug: int = 0):
        if self.groups is None:
            self.split_ligand(debug=debug)
        self.is_haptic = False
        self.haptic_type = []
        for gr in self.groups:
            if gr.is_haptic is None:
                gr.get_hapticity(debug=debug)
            for entry in gr.haptic_type:
                self.haptic_type.append(entry)
            # if gr.is_haptic:
            #     self.is_haptic = True
            #     self.haptic_type = gr.haptic_type
            # for entry in gr.haptic_type:
            #     if entry not in self.haptic_type:
            #         self.haptic_type.append(entry)
        if len(self.haptic_type) > 0:
            self.is_haptic = True

        return self.haptic_type
