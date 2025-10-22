from __future__ import annotations
from typing import Optional
from typing_extensions import deprecated
import numpy as np
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import (
    labels2electrons,
    labels2formula,
    get_adjmatrix,
    is_single_ring,
    get_adjmatrix_from_cif_bonds,
)
from cell2mol.other import compute_centroid
from cell2mol.elementdata import ElementData
from cell2mol.coordination_sphere import (
    coordination_correction_for_haptic,
    coordination_correction_for_nonhaptic,
)
from cell2mol.my_types import (
    HapticType,
    SubType,
)

elemdatabase = ElementData()


###############
#### GROUP ####
###############
class Group(Specie):
    checked_coordination: bool | None = None
    closest_metal: Optional[Metal] = None
    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    metals: list[Metal] | None = None
    denticity: int | None = None

    subtype: SubType = Field(default="group")

    @classmethod
    @deprecated("Use group(**kwargs) with keyword arguments")
    def from_positional(
        cls, labels: list, coord: list, frac_coord: list = None, radii: list = None
    ) -> None:
        return cls(labels=labels, coord=coord, frac_coord=frac_coord, radii=radii)

    #######################################################
    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    #######################################################
    def __repr__(self):
        to_print = ""
        to_print += "------------- Cell2mol GROUP Object --------------\n"
        to_print += Specie.__repr__(self, indirect=True)
        if self.metals is not None:
            to_print += f" Number of Metals             = {len(self.metals)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    #######################################################
    def remove_atom(self, index: int, debug: int = 0):
        if debug > 0:
            print(
                f"GROUP.REMOVE_ATOM: deleting atom {index=} from group with {self.natoms} atoms"
            )
        if index > self.natoms:
            return None
        if self.atoms is None:
            self.set_atoms()
        self.atoms.pop(index)
        self.labels.pop(index)
        self.coord.pop(index)
        self.radii.pop(index)
        self.formula = labels2formula(self.labels)
        self.eleccount = labels2electrons(
            self.labels
        )  ### Assuming neutral specie (so basically this is the sum of atomic numbers)
        self.natoms = len(self.labels)
        self.iscomplex = any(
            (elemdatabase.elementblock[label] == "d")
            or (elemdatabase.elementblock[label] == "f")
            for label in self.labels
        )
        if debug > 0:
            print("GROUP.REMOVE_ATOM. Group after removing atom:")
        if debug > 0:
            print(self)
        if self.natoms > 0:
            if self.closest_metal is not None:
                self.get_closest_metal()
            if self.is_haptic is not None:
                self.get_hapticity()
            if self.centroid is not None:
                self.get_centroid()
            if self.frac_coord is not None:
                self.frac_coord.pop(index)
            if self.adjmat is not None:
                self.get_adjmatrix()
            if self.madjmat is not None:
                self.get_metal_adjmatrix()

    #######################################################
    def get_closest_metal(self, debug: int = 0):
        apos = compute_centroid(np.array(self.coord))
        dist = []
        mol = self.get_parent("molecule")
        for met in mol.metals:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        # finds the closest Metal Atom (tgt)
        self.closest_metal = mol.metals[np.argmin(dist)]
        return self.closest_metal

    #######################################################
    def get_connected_metals(self, debug: int = 0):
        # metal.groups will be used for the calculation of the relative metal radius
        # and define the coordination geometry of the metal /hapicitiy/ hapttype
        self.metals = []

        refcell = self.get_parent("reference")
        geom_bond_cif = getattr(refcell, "geom_bond_cif", None)

        lig = self.get_parent("ligand")

        if lig.metals is None:
            lig.get_connected_metals()

        if refcell is not None and refcell.exist_cif_bond_moiety:
            for met in lig.metals:
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
                            f"GROUP.Get_connected_metals: {self.formula} is connected to {met.label}"
                        )
        else:
            for met in lig.metals:
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
    def get_hapticity(self, debug: int = 0):
        if self.atoms is None:
            self.set_atoms()
        self.is_haptic = False  ## old self.hapticity
        self.haptic_type = []  ## old self.hapttype
        totnum = len(self.labels)
        numC = self.labels.count(
            "C"
        )  # Carbon is the most common connected atom in ligands with hapticity
        numAs = self.labels.count(
            "As"
        )  # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
        numP = self.labels.count("P")
        numO = self.labels.count("O")  # For h4-Enone
        # numN = self.labels.count("N")

        ## Carbon-based Haptic Ligands
        if numC == 2 and totnum == 2:
            self.haptic_type = ["h2-Benzene", "h2-Butadiene", "h2-ethylene"]
            self.is_haptic = True
        elif numC == 3 and numO == 0 and totnum == 3:
            self.haptic_type = ["h3-Allyl", "h3-Cp"]
            self.is_haptic = True
        elif numC == 3 and numO == 1 and totnum == 4:
            self.haptic_type = ["h4-Enone"]
            self.is_haptic = True
        elif numC == 4 and totnum == 4:
            self.haptic_type = ["h4-Butadiene", "h4-Benzene"]
            self.is_haptic = True
        elif numC == 5 and totnum == 5:
            self.haptic_type = ["h5-Cp"]
            self.is_haptic = True
        elif numC == 6 and totnum == 6:
            self.haptic_type = ["h6-Benzene"]
            self.is_haptic = True
        elif numC == 7 and totnum == 7:
            self.haptic_type = ["h7-Cycloheptatrienyl"]
            self.is_haptic = True
        elif numC == 8 and totnum == 8:
            self.haptic_type = ["h8-Cyclooctatetraenyl"]
            self.is_haptic = True
        # Other less common types of haptic ligands
        elif numC == 0 and numAs == 5 and totnum == 5:
            self.haptic_type = ["h5-AsCp"]
            self.is_haptic = True
        elif numC == 0 and numP == 5 and totnum == 5:
            self.haptic_type = ["h5-Pentaphosphole"]
            self.is_haptic = True
        # elif numC == 1 and numP == 1 and totnum == 2:
        #     self.haptic_type = ["h2-P=C"]
        #     self.is_haptic = True
        elif is_single_ring(self.labels, self.coord):
            self.haptic_type = [f"{len(self.labels)}-ring {self.formula}"]
            self.is_haptic = True

        return self.haptic_type

    #######################################################
    def check_coordination(self, debug: int = 0):
        if self.is_haptic is None:
            self.get_hapticity()
        if self.atoms is None:
            self.set_atoms()
        if self.is_haptic:
            self, conn_idx, final_ligand_indices, group_metals_indices = (
                coordination_correction_for_haptic(self, debug=debug)
            )
        if self.is_haptic is False:
            self, conn_idx, final_ligand_indices, group_metals_indices = (
                coordination_correction_for_nonhaptic(self, debug=debug)
            )
        self.checked_coordination = True
        return self, conn_idx, final_ligand_indices, group_metals_indices

    #######################################################
    def get_denticity(self, debug: int = 0):
        if self.checked_coordination is None:
            self.check_coordination(debug=debug)
        self.denticity = 0
        for a in self.atoms:
            self.denticity += a.mconnec
        return self.denticity
