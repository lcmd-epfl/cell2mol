from __future__ import annotations
from typing import Any
from typing_extensions import deprecated
import numpy as np

from pydantic import Field, computed_field
from cell2mol.classes.atom import Atom
from cell2mol.classes.metal import Metal
from cell2mol.connectivity import (
    get_adjacency_types,
    get_element_count,
    labels2electrons,
    labels2formula,
    get_adjmatrix,
)
from cell2mol.connectivity import (
    get_metal_idxs,
    get_radii,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
)
from cell2mol.connectivity import (
    compare_atoms,
    compare_species,
    get_adjmatrix_from_cif_bonds,
)

from cell2mol.charge_assignment import (
    get_protonation_states_specie,
    get_possible_charge_state,
)


from cell2mol.other import extract_from_list, compute_centroid
from cell2mol.elementdata import ElementData
from cell2mol.utils import BaseModel
from cell2mol.my_types import (
    SubType,
    NDArray,
    ChargeState,
)

elemdatabase = ElementData()


##################################
####  CLASSES FOR CELL2MOL 2  ####
##################################
class Specie(BaseModel):
    # Positional arguments
    labels: list[str]
    coord: list[list[float]]
    frac_coord: list[list[float]] | None = None
    radii: list[float] | None = None

    # Optional arguments
    parents: list[Specie] = Field(default_factory=list)
    parents_indices: list[list[int]] = Field(default_factory=list)
    cov_factor: float = Field(default=1.3)
    metal_factor: float = Field(default=1.0)

    # Defined in other methods
    adj_types: NDArray | None = None  # TOFIX romaingrx: NDarray not pydantic compatible
    adjmat: NDArray | None = None
    adjnum: list | None = None
    atnums: list | None = None
    atom_site_labels: list | None = None
    atomic_charges: list | None = None
    atoms: list | None = None
    centroid: list | None = None
    element_count: int | None = None
    frac_centroid: list | None = None
    madjmat: NDArray | None = None
    madjnum: list | None = None
    protonation_states: list | None = None
    rdkit_obj: object | None = None
    smiles: str | None = None
    subtype: SubType | None = None
    totcharge: int | None = None

    charge_state: ChargeState | None = None
    possible_cs: list[ChargeState] | None = None

    # TODO romaingrx: need clarification where we need this, it seems to be used
    # for molecules and ligands
    origin: str | None = None

    # Frozen fields
    type: str = Field(default="specie", frozen=True)
    version: str = Field(default="2.0", frozen=True)

    @computed_field
    @property
    def formula(self) -> str:
        return labels2formula(self.labels)

    @computed_field
    @property
    def eleccount(self) -> int:
        # Assuming neutral specie (so basically this is the sum of atomic numbers)
        return labels2electrons(self.labels)

    @computed_field
    @property
    def natoms(self) -> int:
        return len(self.labels)

    @computed_field
    @property
    def iscomplex(self) -> bool:
        return any(
            (elemdatabase.elementblock[label] == "d")
            or (elemdatabase.elementblock[label] == "f")
            for label in self.labels
        )

    @computed_field
    @property
    def has_IA_IIA(self) -> bool:
        return any(
            (elemdatabase.elementgroup[label] == 1 and label != "H" and label != "D")
            or (elemdatabase.elementgroup[label] == 2)
            for label in self.labels
        )

    @computed_field
    @property
    def has_post_transition_metal(self) -> bool:
        post_transition_metals = {"Al", "Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi"}
        return (
            not self.iscomplex
            and not self.has_IA_IIA
            and any(label in post_transition_metals for label in self.labels)
        )

    @computed_field
    @property
    def indices(self) -> list[int]:
        ## Indices might be the atom ordering within a given specie. e.g. 1st, 2nd, 3rd atom of a specie.
        return list(range(self.natoms))

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        assert len(self.labels) == len(self.coord)
        if self.frac_coord is not None:
            assert len(self.coord) == len(self.frac_coord)
        if self.radii is not None:
            assert len(self.labels) == len(self.radii)
        else:
            self.radii = get_radii(self.labels)

    @classmethod
    @deprecated("Use specie() with the keyword arguments instead.")
    def from_positional(
        cls, labels: list, coord: list, frac_coord: list = None, radii: list = None
    ) -> None:
        return cls(labels=labels, coord=coord, frac_coord=frac_coord, radii=radii)

    ############
    def add_parent(self, parent: object, indices: list, overwrite: bool = True):
        ## associates a parent specie to self. The atom indices of self in parent are given in "indices"
        ## if parent of the same subtype already in self.parent then it is overwritten
        ## this is to avoid having a substructure (e.g. a ligand) in more than one superstructure (e.g. a molecule)

        # 1st-evaluates parent
        append = True
        for idx, p in enumerate(self.parents):
            if p.subtype == parent.subtype:
                if overwrite:
                    self.parents[idx] = parent
                    self.parents_indices[idx] = indices
                append = False
        if append:
            self.parents.append(parent)
            self.parents_indices.append(indices)

        # 2nd-evaluates parents of parent
        if hasattr(parent, "parents"):
            for jdx, p2 in enumerate(parent.parents):
                append = True
                for idx, p in enumerate(self.parents):
                    if p.subtype == p2.subtype:
                        if overwrite:
                            self.parents[idx] = p2
                            self.parents_indices[idx] = parent.get_parent_indices(
                                p2.subtype
                            )
                        append = False
                if append:
                    self.parents.append(p2)
                    self.parents_indices.append(parent.get_parent_indices(p2.subtype))

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
    def get_parent_indices(self, subtype: str):
        ## retrieves parent of a given subtype
        for idx, p in enumerate(self.parents):
            if p.subtype == subtype:
                return self.parents_indices[idx]
        return None

    ############
    def get_centroid(self):
        self.centroid = compute_centroid(np.array(self.coord))
        if self.frac_coord is not None:
            self.frac_centroid = compute_centroid(np.array(self.frac_coord))
            # If fractional coordinates exists, then also computes their centroid
        return self.centroid

    ############
    def set_fractional_coord(self, frac_coord: list, debug: int = 0) -> None:
        assert len(frac_coord) == len(self.coord)
        self.frac_coord = frac_coord

    ############
    # def get_fractional_coord(self, cell_vector=None, debug: int=0) -> None:
    #     if cell_vector is None:
    #         if self.check_parent("cell"):
    #             cell = self.get_parent("cell")
    #             if hasattr(cell,"cellvec"): cell_vector = cell.cell_vector.copy()
    #         else:     print("SPECIE.GET_FRACTIONAL_COORD: get_fractional coordinates. Missing cell vector. Please provide it"); return None
    #     if debug > 1: print(f"SPECIE.GET_FRACTIONAL_COORD: Using cell_vector:{cell_vector}")
    #     self.frac_coord = cart2frac(self.coord, cell_vector)
    #     return self.frac_coord

    ############
    def get_atomic_numbers(self):
        if self.atoms is None:
            self.set_atoms()
        self.atnums = []
        for at in self.atoms:
            self.atnums.append(at.atnum)
        return self.atnums

    ############
    def set_element_count(self, heavy_only: bool = False):
        self.element_count = get_element_count(self.labels, heavy_only=heavy_only)
        return self.element_count

    ############
    def set_adj_types(self):
        if self.adjmat is None:
            self.get_adjmatrix()
        self.adj_types = get_adjacency_types(self.labels, self.adjmat)
        return self.adj_types

    ############
    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        # Stores the covalentradii factor and metal factor that were used to generate the molecule
        self.cov_factor = cov_factor
        self.metal_factor = metal_factor

    ############
    def reset_charge(self):
        self.totcharge = None
        self.atomic_charges = None
        self.smiles = None
        self.rdkit_obj = None
        self.possible_cs = None

        for a in self.atoms:
            a.reset_charge()

    ############
    def set_charges(
        self,
        totcharge: int = None,
        atomic_charges: list = None,
        smiles: str = None,
        rdkit_obj: object = None,
    ) -> None:
        ## Sets total charge
        if totcharge is not None:
            self.totcharge = totcharge
        elif totcharge is None and atomic_charges is not None:
            self.totcharge = np.sum(atomic_charges)
        elif totcharge is None and atomic_charges is None:
            self.totcharge = "Unknown"
        ## Sets atomic charges
        if atomic_charges is not None:
            self.atomic_charges = atomic_charges
            if self.atoms is None:
                self.set_atoms()
            for idx, a in enumerate(self.atoms):
                a.set_charge(self.atomic_charges[idx])
        if smiles is not None:
            self.smiles = smiles
        if rdkit_obj is not None:
            self.rdkit_obj = rdkit_obj

    ############
    def set_atoms(
        self,
        atomlist: list | None = None,
        create_adjacencies: bool = False,
        atom_site_labels: list = None,
        geom_bond_cif: list = None,
        debug: int = 0,
    ):
        debug = 0
        ## If the atom objects already exist, and you want to set them in self from a different specie
        if atomlist is not None:
            if debug >= 2:
                print(f"SPECIE.SET_ATOMS: received {atomlist=}")
            self.atoms = atomlist.copy()
            for idx, at in enumerate(self.atoms):
                at.add_parent(self, index=idx)
        ## If not, that is, if the atom objects must be created from scratch....
        else:
            self.atoms = []

            metal_idxs = get_metal_idxs(self.labels)
            alkali_alkaline_earth_metal_idxs = get_alkali_alkaline_earth_metal_idxs(
                self.labels
            )

            for idx, label in enumerate(self.labels):
                if debug >= 2:
                    print(f"SPECIE.SET_ATOMS: creating atom for label {label}")
                ## For each label in labels, create an atom class object.
                ismetal = (
                    elemdatabase.elementblock[label] == "d"
                    or elemdatabase.elementblock[label] == "f"
                )
                # non transition metals
                # if len(get_non_transition_metal_idxs([label])) > 0: ismetal = True
                if len(get_alkali_alkaline_earth_metal_idxs([label])) > 0:
                    ismetal = True
                if len(metal_idxs) == 0 and len(alkali_alkaline_earth_metal_idxs) == 0:
                    if len(get_post_transition_metal_idxs([label])) > 0:
                        ismetal = True

                if ismetal:
                    if debug >= 2:
                        print(f"SPECIE.SET_ATOMS: {label} identified as metal")
                if debug >= 2:
                    print(f"SPECIE.SET_ATOMS: {ismetal=}")
                if self.frac_coord is not None:
                    if ismetal:
                        newatom = Metal.from_positional(
                            label,
                            self.coord[idx],
                            self.frac_coord[idx],
                            radii=self.radii[idx],
                        )
                    else:
                        newatom = Atom.from_positional(
                            label,
                            self.coord[idx],
                            self.frac_coord[idx],
                            radii=self.radii[idx],
                        )
                else:
                    if ismetal:
                        newatom = Metal.from_positional(
                            label, self.coord[idx], radii=self.radii[idx]
                        )
                    else:
                        newatom = Atom.from_positional(
                            label, self.coord[idx], radii=self.radii[idx]
                        )
                if debug >= 2:
                    print(f"SPECIE.SET_ATOMS: added atom to specie: {self.formula}")
                newatom.add_parent(self, index=idx)
                self.atoms.append(newatom)

        if atom_site_labels is not None:
            if debug >= 2:
                print(f"SPECIE.SET_ATOMS: received {atom_site_labels=}")
            self.atom_site_labels = atom_site_labels.copy()
            for idx, at in enumerate(self.atoms):
                at.set_atom_site_label(atom_site_labels[idx])

        if create_adjacencies:
            if self.adjmat is None:
                self.get_adjmatrix(geom_bond_cif)
            if self.madjmat is None:
                self.get_metal_adjmatrix(geom_bond_cif)
            if self.adjmat is not None and self.madjmat is not None:
                for idx, at in enumerate(self.atoms):
                    at.set_adjacencies(
                        self.adjmat[idx],
                        self.madjmat[idx],
                        self.adjnum[idx],
                        self.madjnum[idx],
                    )

    #######################################################
    def inherit_adjmatrix(self, parent_subtype: str, debug: int = 0):
        exists = self.check_parent(parent_subtype)
        if not exists:
            print(f"SPECIE.INHERIT. {parent_subtype=} does not exist")
            return None
        parent = self.get_parent(parent_subtype)
        indices = self.get_parent_indices(parent_subtype)
        if parent.madjnum is None:
            print(f"SPECIE.INHERIT. {parent_subtype=} does not have madjnum")
            return None
        # print(f"SPECIE.INHERIT. found self in parent ({parent_subtype}) with {indices=}")
        # print(f"SPECIE.INHERIT: parent data:\n{parent.labels=}\n{parent.madjmat=}\n{parent.madjnum=}\n{parent.adjmat=}\n{parent.adjnum=}")
        # print(f"SPECIE.INHERIT: {parent.madjmat.shape=} {parent.madjnum.shape=} {parent.adjmat.shape=} {parent.adjnum.shape=}")
        self.madjmat = np.stack(
            extract_from_list(indices, parent.madjmat, dimension=2), axis=0
        )
        self.madjnum = np.stack(
            extract_from_list(indices, parent.madjnum, dimension=1), axis=0
        )
        self.adjmat = np.stack(
            extract_from_list(indices, parent.adjmat, dimension=2), axis=0
        )
        self.adjnum = np.stack(
            extract_from_list(indices, parent.adjnum, dimension=1), axis=0
        )
        # print(f"SPECIE.INHERIT: self data:\n{self.labels=}\n{self.madjmat=}\n{self.madjnum=}\n{self.adjmat=}\n{self.adjnum=}")
        # print(f"SPECIE.INHERIT: {self.madjmat.shape=} {self.madjnum.shape=} {self.adjmat.shape=} {self.adjnum.shape=}")

    ############
    def get_adjmatrix(self, geom_bond_cif: list = None, debug: int = 0):
        refcell = self.get_parent("reference")
        if (
            refcell is not None
            and getattr(refcell, "exist_cif_bond_moiety", False)
            and geom_bond_cif is not None
            and self.atom_site_labels is not None
        ):
            print("SPECIE.GET_ADJMATRIX: Based on bond information from CIF")
            isgood, adjmat, adjnum = get_adjmatrix_from_cif_bonds(
                self.labels,
                self.coord,
                self.atom_site_labels,
                geom_bond_cif,
                metal_only=False,
            )
        else:
            print("SPECIE.GET_ADJMATRIX: Based on interatomic distances")
            isgood, adjmat, adjnum, warning = get_adjmatrix(
                self.labels, self.coord, self.cov_factor, self.radii
            )

        if isgood:
            self.adjmat = adjmat
            self.adjnum = adjnum
        else:
            self.adjmat = None
            self.adjnum = None
        return self.adjmat, self.adjnum

    ############
    def get_metal_adjmatrix(self, geom_bond_cif: list = None, debug: int = 0):
        refcell = self.get_parent("reference")
        if (
            refcell is not None
            and getattr(refcell, "exist_cif_bond_moiety", False)
            and geom_bond_cif is not None
            and self.atom_site_labels is not None
        ):
            print("SPECIE.GET_METAL_ADJMATRIX: Based on bond information from CIF")
            isgood, madjmat, madjnum = get_adjmatrix_from_cif_bonds(
                self.labels,
                self.coord,
                self.atom_site_labels,
                geom_bond_cif,
                metal_only=True,
            )
        else:
            print("SPECIE.GET_METAL_ADJMATRIX: Based on interatomic distances")
            isgood, madjmat, madjnum, warning = get_adjmatrix(
                self.labels, self.coord, self.cov_factor, self.radii, metal_only=True
            )

        if isgood:
            self.madjmat = madjmat
            self.madjnum = madjnum
        else:
            self.madjmat = None
            self.madjnum = None
        return self.madjmat, self.madjnum

    ############
    def get_occurrence(self, substructure: object) -> int:
        occurrence = 0
        ## Ligands in Complexes or Groups in Ligands
        done = False
        # TOFIX @romaingrx: check if we can't pass the actual class as type
        if "subtype" in substructure and "subtype" in self:
            if substructure.subtype == "ligand" and self.subtype == "molecule":
                if self.ligands is None:
                    self.split_complex()
                if self.ligands is not None:
                    for lig in self.ligands:
                        issame = compare_species(substructure, lig, debug=1)
                        if issame:
                            occurrence += 1
                    done = True
            elif substructure.subtype == "group" and self.subtype == "ligand":
                if self.ligands is None:
                    self.split_complex()
                if self.ligands is not None:
                    for lig in self.ligands:
                        if lig.groups is None:
                            self.split_ligand()
                        for g in lig.groups:
                            issame = compare_species(substructure, g, debug=1)
                            if issame:
                                occurrence += 1
                done = True
        ## Atoms in Species
        if not done:
            if substructure.type == "atom" and self.type == "specie":
                if self.atoms is None:
                    self.set_atoms()
                for at in self.atoms:
                    issame = compare_atoms(substructure, at)
                    if issame:
                        occurrence += 1
        return occurrence

    ############
    def get_protonation_states(self, debug: int = 0):
        # !!! WARNING. FUNCTION defined at the "specie" level, but will only do something for ligands and organic (iscomplex == False) molecules
        if self.subtype == "group":
            if self.denticity is None:
                self.get_denticity()
            if self.is_haptic is None:
                self.get_hapticity()
            self.protonation_states = None
        elif self.subtype == "ligand":
            # if self.groups is None: self.split_ligand()
            if self.is_haptic is None:
                self.get_hapticity()
            if self.denticity is None:
                self.get_denticity()
            if self.is_nitrosyl is None:
                self.evaluate_as_nitrosyl()
            self.protonation_states = get_protonation_states_specie(self, debug=debug)
        else:
            if self.is_haptic is None:
                self.get_hapticity()
            self.protonation_states = get_protonation_states_specie(self, debug=debug)
        return self.protonation_states

    ############
    def get_possible_cs(self, debug: int = 0):
        ## Arranges a list of possible charge_states associated with this species,
        ## which is later managed at the cell level to determine the good one
        if self.possible_cs is None:
            if self.subtype == "ligand" or (
                self.subtype == "molecule"
                and not self.iscomplex
                and not self.has_IA_IIA
            ):
                print(
                    f"SPECIE.GET_POSSIBLE_CS: {self.formula} {self.protonation_states=}"
                )
                if self.protonation_states is None:
                    self.get_protonation_states(debug=debug)
                    print(
                        f"SPECIE.GET_POSSIBLE_CS: Obtained {self.formula} {self.protonation_states=}"
                    )

                self.possible_cs = get_possible_charge_state(self, debug=debug)
        return self.possible_cs

    ############
    def print_xyz(self):
        print(self.natoms)
        print("")
        for idx, label in enumerate(self.labels):
            print(
                "%s  %.6f  %.6f  %.6f"
                % (label, self.coord[idx][0], self.coord[idx][1], self.coord[idx][2])
            )

    ############
    ## This defines the sum operation between two species. To be implemented
    def __add__(self, other):
        if not isinstance(other, type(self)):
            return self
        return self

    ############
    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        to_print = ""
        if not indirect:
            to_print += "------------- Cell2mol SPECIE Object --------------\n"
        to_print += f" Version                      = {self.version}\n"
        to_print += f" Type                         = {self.type}\n"
        if self.subtype is not None:
            to_print += f" Sub-Type                     = {self.subtype}\n"
        to_print += f" Number of Atoms              = {self.natoms}\n"
        to_print += f" Formula                      = {self.formula}\n"
        to_print += f" Covalent Radii Factor        = {self.cov_factor}\n"
        to_print += f" Metal Radii Factor           = {self.metal_factor}\n"
        if self.adjmat is not None:
            to_print += " Has Adjacency Matrix         = YES\n"
        else:
            to_print += " Has Adjacency Matrix         = NO \n"
        if self.totcharge is not None:
            to_print += f" Total Charge                 = {self.totcharge}\n"
        if self.subtype == "molecule" and self.spin is not None:
            to_print += f" Spin                         = {self.spin}\n"
        if self.smiles is not None:
            to_print += f" Smiles                       = {self.smiles}\n"
        if self.origin is not None:
            to_print += f" Origin                       = {self.origin}\n"
        if not indirect:
            to_print += "---------------------------------------------------\n"
        return to_print
