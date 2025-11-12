from __future__ import annotations
import numpy as np
from typing import Annotated, Any
from typing_extensions import deprecated
from pydantic import Field, PlainSerializer, computed_field
from cell2mol.connectivity import (
    get_radii,
    get_adjmatrix,
    get_adjmatrix_from_cif_bonds,
)
from cell2mol.elementdata import ElementData
from cell2mol.utils import BaseModel
from cell2mol.utils.pydantic import serialize_circular_references
from cell2mol.my_types import (
    NDArray,
)

elemdatabase = ElementData()


###############
#### ATOM #####
###############
class Atom(BaseModel):
    label: str
    coord: list[float]
    frac_coord: list[float] | None = None
    radii: float | None = None
    parents: Annotated[list[object], PlainSerializer(serialize_circular_references)] = (
        Field(default_factory=list)
    )
    parents_index: list[int] = Field(default_factory=list)

    # Seem to be set in various places depending on the context, might need to check this
    cov_factor: float | None = None
    metal_factor: float | None = None
    atom_site_label: str | None = None
    connec: int | None = None
    mconnec: int | None = None
    adjacency: list[int] = Field(default_factory=list)
    metal_adjacency: list[int] = Field(default_factory=list)
    metal_factor: float | None = None
    charge: int | None = None
    bonds: Annotated[list[object], PlainSerializer(serialize_circular_references)] = (
        Field(default_factory=list)
    )

    version: str = Field(default="2.0", frozen=True)
    type: str = Field(default="atom", frozen=True)
    # Originally atom does not have a subtype, but we add it for consistency with other classes
    subtype: str = Field(default="atom")

    @computed_field
    @property
    def atnum(self) -> int:
        return elemdatabase.elementnr[self.label]

    @computed_field
    @property
    def block(self) -> str:
        return elemdatabase.elementblock[self.label]

    @computed_field
    @property
    def formula(self) -> str:
        return self.label

    def model_post_init(self, __context: Any) -> None:
        self.radii = self.radii or get_radii(self.label)

    @classmethod
    @deprecated("Use atom() with the keyword arguments instead.")
    def from_positional(
        cls, label: str, coord: list, frac_coord: list = None, radii: float = None
    ) -> "Atom":
        """
        Creates an atom instance using positional arguments.

        Args:
            label: Atomic symbol
            coord: Cartesian coordinates [x,y,z]
            frac_coord: Optional fractional coordinates
            radii: Optional atomic radius

        Returns:
            atom: New atom instance
        """
        return cls(label=label, coord=coord, frac_coord=frac_coord, radii=radii)

        ############

    def add_parent(self, parent: object, index: int, overwrite: bool = True):
        ## associates a parent specie to self. The atom indices of self in parent are given in "indices"
        ## if parent of the same subtype already in self.parent then it is overwritten
        ## this is to avoid having a substructure (e.g. a ligand) in more than one superstructure (e.g. a molecule)
        append = True
        for idx, p in enumerate(self.parents):
            if p.subtype == parent.subtype:
                if overwrite:
                    self.parents[idx] = parent
                    self.parents_index[idx] = index
                append = False
        if append:
            self.parents.append(parent)
            self.parents_index.append(index)

    ############
    def check_parent(self, subtype: str):
        ## checks if parent of a given subtype exists
        for p in self.parents:
            if p.subtype == subtype:
                return True
        return False

    ############
    def get_parent(self, subtype: str):
        ## retrieves parent of a given subtype
        for p in self.parents:
            if p.subtype == subtype:
                return p
        return None

    ############
    def get_parent_index(self, subtype: str):
        ## retrieves parent of a given subtype
        for idx, p in enumerate(self.parents):
            if p.subtype == subtype:
                return self.parents_index[idx]
        return None

    #######################################################
    def check_connectivity(self, other: object, debug: int = 0):
        ## Checks whether two atoms are connected (through the adjacency)
        if not isinstance(other, type(self)):
            return False
        labels = list([self.label, other.label])
        coords = list([self.coord, other.coord])
        refcell = self.get_parent("reference")
        if refcell is not None and refcell.exist_cif_bond_moiety:
            atom_site_labels = [self.atom_site_label, other.atom_site_label]
            isgood, adjmat, adjnum = get_adjmatrix_from_cif_bonds(
                labels, coords, atom_site_labels, refcell.geom_bond_cif
            )
        else:
            isgood, adjmat, adjnum, warning = get_adjmatrix(labels, coords)
        if isgood and adjnum[0] > 0:
            return True
        else:
            return False

    #######################################################
    def add_bond(self, newbond: object, debug: int = 0):
        at1 = newbond.atom1
        at2 = newbond.atom2
        found = False
        for b in self.bonds:
            if (b.atom1 == at1 and b.atom2 == at2) or (
                b.atom1 == at2 and b.atom2 == at1
            ):
                if debug > 0:
                    print("ATOM.ADD_BOND found the same bond with atoms:")
                if debug > 0:
                    print(f"atom1: {b.atom1}")
                if debug > 0:
                    print(f"atom2: {b.atom2}")
                found = True  ### It means that the same bond has already been defined
        if not found:
            self.bonds.append(newbond)

    #######################################################
    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        self.cov_factor = cov_factor
        self.metal_factor = metal_factor

    #######################################################
    def reset_charge(self) -> None:
        self.charge = None
        self.possible_cs = None

    #######################################################
    def set_charge(self, charge: int) -> None:
        self.charge = int(charge)

    #######################################################
    def set_adjacencies(self, adjmat, madjmat, adjnum: int, madjnum: int):
        self.connec = int(adjnum)
        self.mconnec = int(madjnum)
        self.adjacency = []
        self.metal_adjacency = []
        for idx, c in enumerate(
            adjmat
        ):  ## The atom only receives one row of adjmat, so this is not a matrix anymore. Keep in mind that the idx are the indices of parent
            if c > 0:
                self.adjacency.append(idx)
        for idx, c in enumerate(
            madjmat
        ):  ## The atom only receives one row of madjmat, so this is not a matrix anymore
            if c > 0:
                self.metal_adjacency.append(idx)

    #######################################################
    @property
    def closest_metal(self):
        return self.get_closest_metal()

    def get_closest_metal(self, debug: int = 0):
        ## Here, the list of metal atoms must be provided
        apos = self.coord
        dist = []
        mol = self.get_parent("molecule")
        for met in mol.metals:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        closest_metal = mol.metals[np.argmin(dist)]
        return closest_metal

    #######################################################
    def information(self, cov_factor: float, metal_factor: float) -> None:
        self.cov_factor = cov_factor
        self.metal_factor = metal_factor

    def set_atom_site_label(self, atom_site_label: str) -> None:
        self.atom_site_label = atom_site_label

    #######################################################
    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    #######################################################
    def __repr__(self, indirect: bool = False):
        to_print = ""
        if not indirect:
            to_print += "------------- Cell2mol ATOM Object ----------------\n"
        to_print += f" Version                      = {self.version}\n"
        to_print += f" Type                         = {self.type}\n"
        if self.subtype is not None:
            to_print += f" Sub-Type                     = {self.subtype}\n"
        to_print += f" Label                        = {self.label}\n"
        to_print += f" Atomic Number                = {self.atnum}\n"
        idx = self.get_parent_index("molecule")
        if idx is not None:
            to_print += f" Index in Molecule            = {idx}\n"
        idx = self.get_parent_index("ligand")
        if idx is not None:
            to_print += f" Index in Ligand              = {idx}\n"
        # if self.occurrence is not None:
        #     to_print += f" Occurrence in Parent         = {self.occurrence}\n"
        if self.mconnec is not None:
            to_print += f" Metal Adjacency (mconnec)    = {self.mconnec}\n"
        if self.connec is not None:
            to_print += f" Regular Adjacencies (connec) = {self.connec}\n"
        if self.charge is not None:
            to_print += f" Atom Charge                  = {self.charge}\n"
        if not indirect:
            to_print += "----------------------------------------------------\n"
        return to_print

    #######################################################
    def reset_mconnec(self, met, diff: int = -1, debug: int = 0):
        if debug >= 2:
            print(
                f"ATOM.RESET_MCONN: resetting mconnec (and connec) for atom {self.label=}"
            )
        if debug >= 2:
            print(f"ATOM.RESET_MCONN: initial {self.connec=} {self.mconnec=}")
        if debug >= 2:
            print(
                f"ATOM.RESET_MCONN: initial = {self.adjacency=} {self.metal_adjacency=}"
            )
        self.mconnec += diff
        self.connec += diff

        if debug >= 2:
            print(f"ATOM.RESET_MCONN: initial {met.connec=} {met.mconnec=}")
        if debug >= 2:
            print(
                f"ATOM.RESET_MCONN: initial = {met.adjacency=} {met.metal_adjacency=}"
            )

        # Corrects data of metal object
        met.mconnec += diff
        met.connec += diff

        exists = self.check_parent("ligand")
        if exists:
            lig = self.get_parent("ligand")
            lig_idx = self.get_parent_index("ligand")

            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: resetting mconnec (and connec) for atom {self.label=} in ligadn {lig_idx=}"
                )
            if debug > 2:
                print("ATOM.RESET_MCONN: updating ligand atoms and madjnum")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: {lig.natoms=}")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: {lig.labels=}")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: initial {lig.madjnum=} {len(lig.madjnum)}")

            # Nothing in madjmat of the ligand object, all zeros
            if debug > 2:
                print("ATOM.RESET_MCONN: updating ligand atoms and adjnum")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: initial {lig.adjnum=} {len(lig.adjnum)}")
            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: initial {lig.atoms[lig_idx].connec=} {lig.atoms[lig_idx].mconnec=}"
                )
            if debug > 2:
                print(f"ATOM.RESET_MCONN: initial {lig.madjnum[lig_idx]=}")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: initial {lig.adjnum[lig_idx]=}")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: initial {met.connec=} {met.mconnec=}")
            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: initial {lig.madjmat[lig_idx]=} {lig.adjmat[lig_idx]=}"
                )

            # Corrects data in metal_adjacency number of the ligand class
            lig.madjnum[lig_idx] += diff
            # Corrects data in adjacency number of the ligand class
            lig.adjnum[lig_idx] += diff

            lig.atoms[lig_idx].set_adjacencies(
                lig.adjmat[lig_idx],
                lig.madjmat[lig_idx],
                lig.adjnum[lig_idx],
                lig.madjnum[lig_idx],
            )

            # we should delete the adjacencies, but not a priority
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: final {lig.madjnum=}")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: final {lig.adjnum=}")
            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: final {lig.atoms[lig_idx].connec=} {lig.atoms[lig_idx].mconnec=}"
                )
            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: final {lig.atoms[lig_idx].adjacency=} {lig.atoms[lig_idx].metal_adjacency=}"
                )
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: final {lig.madjnum[lig_idx]=}")
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: final {lig.adjnum[lig_idx]=}")
            lig.get_connected_idx(debug=debug)
            lig.get_connected_atoms(debug=debug)

        exists = self.check_parent("molecule")
        if exists:
            mol = self.get_parent("molecule")
            mol_idx = self.get_parent_index("molecule")
            met_idx = met.get_parent_index("molecule")

            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: resetting mconnec (and connec) for atom {self.label=} in molecule {mol_idx=} with metal {met_idx=}"
                )
            if debug >= 2:
                print("ATOM.RESET_MCONN: updating molecule atoms and madjnum")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: {mol.natoms=}")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: {mol.labels=}")
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: initial {mol.madjnum=} {len(mol.madjnum)}")
            if debug >= 2:
                print("ATOM.RESET_MCONN: updating molecule atoms and adjnum")
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: initial {mol.adjnum=} {len(mol.adjnum)}")
            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: initial {mol.atoms[mol_idx].mconnec=} {mol.atoms[mol_idx].connec=}"
                )
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: initial {met.mconnec=} {met.connec=}")

            if debug > 2:
                print(f"ATOM.RESET_MCONN: initial {mol.madjnum[mol_idx]=}")
            if debug > 2:
                print(f"ATOM.RESET_MCONN: initial {mol.adjnum[mol_idx]=}")
            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: initial {mol.madjnum[met_idx]=} {mol.madjnum[mol_idx]=}"
                )
            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: initial {mol.adjnum[met_idx]=} {mol.adjnum[mol_idx]=}"
                )
            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: initial {mol.madjmat[met_idx,mol_idx]=} {mol.madjmat[mol_idx,met_idx]=}"
                )
            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: initial {mol.adjmat[met_idx,mol_idx]=} {mol.adjmat[mol_idx,met_idx]=}"
                )

            # Corrects data in metal_adjacency number of the molecule class
            mol.madjnum[mol_idx] += diff
            mol.madjnum[met_idx] += diff

            # Corrects data in metal_adjacency matrix of the molecule class
            mol.madjmat[mol_idx, met_idx] += diff
            mol.madjmat[met_idx, mol_idx] += diff

            # Corrects data in adjacency number of the molecule class
            mol.adjnum[mol_idx] += diff
            mol.adjnum[met_idx] += diff

            # Corrects data in adjacency matrix of the molecule class
            mol.adjmat[mol_idx, met_idx] += diff
            mol.adjmat[met_idx, mol_idx] += diff

            self.set_adjacencies(
                mol.adjmat[mol_idx],
                mol.madjmat[mol_idx],
                mol.adjnum[mol_idx],
                mol.madjnum[mol_idx],
            )

            met.set_adjacencies(
                mol.adjmat[met_idx],
                mol.madjmat[met_idx],
                mol.adjnum[met_idx],
                mol.madjnum[met_idx],
            )

            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: final {mol.atoms[mol_idx].connec=} {mol.atoms[mol_idx].mconnec=}"
                )
            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: final {mol.atoms[mol_idx].adjacency=} {mol.atoms[mol_idx].metal_adjacency=}"
                )
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: final {met.connec=} {met.mconnec=}")
            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: final {met.adjacency=} {met.metal_adjacency=}"
                )
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: final {mol.madjnum[mol_idx]=}")
            if debug >= 2:
                print(f"ATOM.RESET_MCONN: final {mol.adjnum[mol_idx]=}")
            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: final {mol.madjnum[met_idx]=} {mol.madjnum[mol_idx]=}"
                )
            if debug >= 2:
                print(
                    f"ATOM.RESET_MCONN: final {mol.adjnum[met_idx]=} {mol.adjnum[mol_idx]=}"
                )
            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: final {mol.madjmat[met_idx,mol_idx]=} {mol.madjmat[mol_idx,met_idx]=}"
                )
            if debug > 2:
                print(
                    f"ATOM.RESET_MCONN: final {mol.adjmat[met_idx,mol_idx]=} {mol.adjmat[mol_idx,met_idx]=}"
                )
