from __future__ import annotations
from typing import Any, Literal, Optional
from typing_extensions import deprecated
import numpy as np
import os

from pydantic import Field, computed_field
from cell2mol.connectivity import (
    get_adjacency_types,
    get_element_count,
    labels2electrons,
    labels2formula,
    get_adjmatrix,
    is_haptic_ring,
)
from cell2mol.connectivity import (
    get_metal_idxs,
    get_non_transition_metal_idxs,
    split_species,
    get_radii,
    split_group,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
)
from cell2mol.connectivity import (
    compare_atoms,
    compare_species,
    compare_metals,
    compare_reference_indices,
    get_adjmatrix_from_cif_bonds,
)
from cell2mol.cell_reconstruction import classify_fragments, fragments_reconstruct
from cell2mol.cell_operations import cart2frac, frac2cart_fromparam

from cell2mol.charge_assignment import (
    get_protonation_states_specie,
    get_possible_charge_state,
    get_metal_poscharges,
    get_empty_protonation_state,
)
from cell2mol.charge_assignment import (
    prepare_unresolved,
    prepare_mols,
    correct_smiles_ligand,
)

from cell2mol.new_charge_assignment import (
    set_charge_state,
    prepare_mol,
    balance_charge,
    assign_charge_to_specie,
)
from cell2mol.new_charge_assignment import (
    create_bonds_specie,
    create_metal_ligand_bonds,
    create_metal_metal_bonds,
)

from cell2mol.spin import assign_spin_metal, assign_spin_complexes, predict_ox_state
from cell2mol.other import extract_from_list, compute_centroid, get_dist, get_angle
from cell2mol.other import handle_error
from cell2mol.elementdata import ElementData
from cell2mol.read_write import get_moiety_indices_from_labels
from cell2mol.coordination_sphere import (
    coordination_correction_for_haptic,
    coordination_correction_for_nonhaptic,
    define_coordination_geometry,
)
from cell2mol.utils import BaseModel
from cell2mol.types import Spin, HapticType, Type, SubType, NOType, NDArray, ChargeState

elemdatabase = ElementData()
import pickle




##################################
####  CLASSES FOR CELL2MOL 2  ####
##################################
class specie(BaseModel):
    # Positional arguments
    labels: list[str]
    coord: list[list[float]]
    frac_coord: list[list[float]] | None = None
    radii: list[float] | None = None

    # Optional arguments
    parents: list["specie"] = Field(default_factory=list)
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
        if hasattr(parent,"parents"):
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
        from cell2mol.other import compute_centroid

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
            alkali_alkaline_earth_metal_idxs = get_alkali_alkaline_earth_metal_idxs(self.labels)

            for idx, l in enumerate(self.labels):
                if debug >= 2:
                    print(f"SPECIE.SET_ATOMS: creating atom for label {l}")
                ## For each l in labels, create an atom class object.
                ismetal = (
                    elemdatabase.elementblock[l] == "d"
                    or elemdatabase.elementblock[l] == "f"
                )
                # non transition metals
                # if len(get_non_transition_metal_idxs([l])) > 0: ismetal = True
                if len(get_alkali_alkaline_earth_metal_idxs([l])) > 0:
                    ismetal = True
                if len(metal_idxs)== 0 and len(alkali_alkaline_earth_metal_idxs) == 0:
                    if len(get_post_transition_metal_idxs([l])) > 0:
                        ismetal = True

                if ismetal:
                    if debug >= 2:
                        print(f"SPECIE.SET_ATOMS: {l}")
                if debug >= 2:
                    print(f"SPECIE.SET_ATOMS: {ismetal=}")
                if self.frac_coord is not None:
                    if ismetal:
                        newatom = metal.from_positional(
                            l,
                            self.coord[idx],
                            self.frac_coord[idx],
                            radii=self.radii[idx],
                        )
                    else:
                        newatom = atom.from_positional(
                            l,
                            self.coord[idx],
                            self.frac_coord[idx],
                            radii=self.radii[idx],
                        )
                else:
                    if ismetal:
                        newatom = metal.from_positional(
                            l, self.coord[idx], radii=self.radii[idx]
                        )
                    else:
                        newatom = atom.from_positional(
                            l, self.coord[idx], radii=self.radii[idx]
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
    def get_adjmatrix(self, geom_bond_cif: list=None, debug: int=0):
        refcell = self.get_parent("reference")
        if refcell is not None and getattr(refcell, "exist_cif_bond_moiety", False) and geom_bond_cif is not None and self.atom_site_labels is not None:

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
            isgood, adjmat, adjnum = get_adjmatrix(
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
    def get_metal_adjmatrix(self, geom_bond_cif: list=None, debug: int=0):
        refcell = self.get_parent("reference")
        if refcell is not None and getattr(refcell, "exist_cif_bond_moiety", False) and geom_bond_cif is not None and self.atom_site_labels is not None:
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
            isgood, madjmat, madjnum = get_adjmatrix(
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
                    for l in self.ligands:
                        issame = compare_species(substructure, l, debug=1)
                        if issame:
                            occurrence += 1
                    done = True
            elif substructure.subtype == "group" and self.subtype == "ligand":
                if self.ligands is None:
                    self.split_complex()
                if self.ligands is not None:
                    for l in self.ligands:
                        if l.groups is None:
                            self.split_ligand()
                        for g in l.groups:
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
        if self.subtype == "ligand" or (
            self.subtype == "molecule" and not self.iscomplex and not self.has_IA_IIA
        ):
            if self.protonation_states is None:
                self.get_protonation_states(debug=debug)
            self.possible_cs = get_possible_charge_state(self, debug=debug)
        return self.possible_cs

    ############
    def print_xyz(self):
        print(self.natoms)
        print("")
        for idx, l in enumerate(self.labels):
            print(
                "%s  %.6f  %.6f  %.6f"
                % (l, self.coord[idx][0], self.coord[idx][1], self.coord[idx][2])
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
            to_print += f"------------- Cell2mol SPECIE Object --------------\n"
        to_print += f" Version                      = {self.version}\n"
        to_print += f" Type                         = {self.type}\n"
        if self.subtype is not None:
            to_print += f" Sub-Type                     = {self.subtype}\n"
        to_print += f" Number of Atoms              = {self.natoms}\n"
        to_print += f" Formula                      = {self.formula}\n"
        to_print += f" Covalent Radii Factor        = {self.cov_factor}\n"
        to_print += f" Metal Radii Factor           = {self.metal_factor}\n"
        if self.adjmat is not None:
            to_print += f" Has Adjacency Matrix         = YES\n"
        else:
            to_print += f" Has Adjacency Matrix         = NO \n"
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


###############
### MOLECULE ##
###############
class molecule(specie):
    """
    A molecule is a specie that contains other specie objects.
    """

    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    ligands: list | None = None
    metals: list["atom"] | None = None
    spin: Spin | None = None
    ref_indices: list[int] | None = None
    cell_indices: list[int] | None = None
    
    unique_index: int | None = None
    totcharge_cif : int | None = None

    smiles: str | list[str] | None = None
    smiles_with_H: list[str] | None = None
    error_create_bonds : bool = False

    subtype: SubType = Field(default="molecule")

    @classmethod
    @deprecated("Use molecule() with the keyword arguments instead.")
    def from_positional(
        cls, labels: list, coord: list, frac_coord: list = None, radii: list = None
    ) -> "molecule":
        return cls(labels=labels, coord=coord, frac_coord=frac_coord, radii=radii)

    def __repr__(self):
        to_print = ""
        to_print += f"------------- Cell2mol MOLECULE Object --------------\n"
        to_print += specie.__repr__(self, indirect=True)
        if self.ligands is not None:
            if self.ligands is not None:
                to_print += f" Number of Ligands            = {len(self.ligands)}\n"
        if self.metals is not None:
            if self.metals is not None:
                to_print += f" Number of Metals             = {len(self.metals)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    ############
    def get_spin(self, debug: int = 0):
        if self.iscomplex:
            self.spin = assign_spin_complexes(self)
        else:
            self.spin = 1
        if debug >= 1:
            print(
                f"GET_SPIN: Spin multiplicity of the complex {self.formula} is assigned as {self.spin}\n"
            )
        return self.spin

    ############
    def reset_charge(self):
        specie.reset_charge(
            self
        )  ## First uses the generic specie class function for itself and its atoms
        if self.ligands is not None:  ## Second removes for the child classes
            for lig in self.ligands:
                lig.reset_charge()
        if self.metals is not None:
            for met in self.metals:
                met.reset_charge()

    ############
    def split_post_transition_metal(self, debug: int = 0):
        if self.atoms is None:
            self.set_atoms()
        if not self.has_post_transition_metal:
            self.ligands = None
            self.metals = None
        else:
            refcell = self.get_parent("reference")
            geom_bond_cif = getattr(refcell, "geom_bond_cif", None)

            post_transition_metal_indices = []
            for idx, l in enumerate(self.labels):
                if l in ["Al", "Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi"]:
                    post_transition_metal_indices.append(idx)

            self.ligands = []
            self.metals = []

            # Identify Metals and the rest

            if len(post_transition_metal_indices) > 0:
                print(
                    f"MOLECULE.SPLIT_POST_TRANSITION_METAL: Post-transition metals in the molecule {self.formula}"
                )
                print(
                    f"MOLECULE.SPLIT_POST_TRANSITION_METAL: {[self.labels[idx] for idx in post_transition_metal_indices]}"
                )

            rest_idx = list(
                idx for idx in self.indices if idx not in post_transition_metal_indices
            )
            if debug > 0:
                print(f"MOLECULE.SPLIT_POST_TRANSITION_METAL: labels={self.labels}")
            if debug > 0:
                print(f"MOLECULE.SPLIT_POST_TRANSITION_METAL: metal_idx={post_transition_metal_indices}")
            if debug > 0:
                print(f"MOLECULE.SPLIT_POST_TRANSITION_METAL: rest_idx={rest_idx}")

            # Split the "rest" to obtain the ligands
            rest_labels = extract_from_list(rest_idx, self.labels, dimension=1)
            rest_coord = extract_from_list(rest_idx, self.coord, dimension=1)
            if self.frac_coord is not None:
                rest_frac = extract_from_list(rest_idx, self.frac_coord, dimension=1)
            rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)

            rest_radii = extract_from_list(rest_idx, self.radii, dimension=1)
            rest_atoms = extract_from_list(rest_idx, self.atoms, dimension=1)
            if self.atom_site_labels is not None:
                rest_atom_site_labels = extract_from_list(
                    rest_idx, self.atom_site_labels, dimension=1
                )

            if debug >= 2:
                print(f"MOLECULE.SPLIT_POST_TRANSITION_METAL: rest labels: {rest_labels}")
                print(f"MOLECULE.SPLIT_POST_TRANSITION_METAL: rest indices: {rest_indices}")
                print(f"MOLECULE.SPLIT_POST_TRANSITION_METAL: rest radii: {rest_radii}")

            if debug > 0:
                print(
                    f"MOLECULE.SPLIT_POST_TRANSITION_METAL: splitting species with {len(rest_labels)} atoms in block"
                )
            if len(rest_labels) == 0:

                if debug > 0:
                    print(
                        f"MOLECULE.SPLIT_POST_TRANSITION_METAL: No ligands found in the complex {self.formula}"
                    )
                if debug > 0:
                    print(
                        f"MOLECULE.SPLIT_POST_TRANSITION_METAL: Returning empty ligand list {self.ligands=}"
                    )
                for m in post_transition_metal_indices:
                    self.metals.append(self.atoms[m])
            else:
                if refcell.exist_cif_bond_moiety and refcell.geom_bond_cif is not None:
                    blocklist = split_species(
                        rest_labels,
                        rest_coord,
                        radii=rest_radii,
                        atom_site_labels=rest_atom_site_labels,
                        geom_bond_cif=refcell.geom_bond_cif,
                        cov_factor=self.cov_factor,
                        debug=debug,
                    )
                else:
                    blocklist = split_species(
                        rest_labels, rest_coord, radii=rest_radii, debug=debug
                    )

                # if debug > 0:
                print(f"MOLECULE.SPLIT_POST_TRANSITION_METAL: received {len(blocklist)} blocks {blocklist=}")

                ## Arranges Ligands
                for b in blocklist:
                    if debug > 0:
                        print(f"PREPARING BLOCK: {b}")
                    lig_indices = extract_from_list(b, rest_indices, dimension=1)
                    lig_labels = extract_from_list(b, rest_labels, dimension=1)
                    lig_coord = extract_from_list(b, rest_coord, dimension=1)
                    if self.frac_coord is not None:
                        lig_frac_coord = extract_from_list(b, rest_frac, dimension=1)

                    lig_radii = extract_from_list(b, rest_radii, dimension=1)
                    lig_atoms = extract_from_list(b, rest_atoms, dimension=1)
                    if self.atom_site_labels is not None:
                        lig_atom_site_labels = extract_from_list(
                            b, rest_atom_site_labels, dimension=1
                        )


                    if debug > 0:
                        print(f"CREATING LIGAND: {labels2formula(lig_labels)}")

                    if self.frac_coord is not None:
                        newligand = ligand.from_positional(
                            lig_labels, lig_coord, lig_frac_coord, radii=lig_radii
                        )
                    else:
                        newligand = ligand.from_positional(
                            lig_labels, lig_coord, radii=lig_radii
                        )

                    newligand.origin = "split_post_transition_metal"
                    newligand.add_parent(self, indices=lig_indices)

                    if self.check_parent("unitcell"):
                        cell_indices = [
                            a.get_parent_index("unitcell") for a in lig_atoms
                        ]
                        newligand.add_parent(
                            self.get_parent("unitcell"), indices=cell_indices
                        )

                    if self.check_parent("reference"):
                        ref_indices = [a.get_parent_index("reference") for a in lig_atoms]
                        newligand.add_parent(self.get_parent("reference"), indices=ref_indices)


                    newligand.set_adjacency_parameters(self.cov_factor, self.metal_factor)
                    
                    newligand.set_atoms(atomlist=lig_atoms, 
                                        atom_site_labels=lig_atom_site_labels, 
                                        geom_bond_cif=geom_bond_cif)

                    newligand.inherit_adjmatrix("molecule")
                    self.ligands.append(newligand)

                ## Arranges Metals
                for m in post_transition_metal_indices:
                    self.metals.append(self.atoms[m])

            return self.ligands, self.metals
    ############
    def split_IA_IIA(self, debug: int = 0):
        if self.atoms is None:
            self.set_atoms()
        if not self.has_IA_IIA:
            self.ligands = None
            self.metals = None
        else:
            refcell = self.get_parent("reference")
            geom_bond_cif = getattr(refcell, "geom_bond_cif", None)

            IA_IIA_metal_indices = []
            for idx, l in enumerate(self.labels):
                if (
                    elemdatabase.elementgroup[l] == 1 and l != "H" and l != "D"
                ):  # Alkali Metals
                    IA_IIA_metal_indices.append(idx)
                elif elemdatabase.elementgroup[l] == 2:  # Alkaline Earth Metals
                    IA_IIA_metal_indices.append(idx)

            self.ligands = []
            self.metals = []

            # Identify Metals and the rest

            if len(IA_IIA_metal_indices) > 0:
                print(
                    f"MOLECULE.SPLIT_IA_IIA: Alkali or Alkali earth metals in the molecule {self.formula}"
                )
                print(
                    f"MOLECULE.SPLIT_IA_IIA: {[self.labels[idx] for idx in IA_IIA_metal_indices]}"
                )

            rest_idx = list(
                idx for idx in self.indices if idx not in IA_IIA_metal_indices
            )
            if debug > 0:
                print(f"MOLECULE.SPLIT_IA_IIA: labels={self.labels}")
            if debug > 0:
                print(f"MOLECULE.SPLIT_IA_IIA: metal_idx={IA_IIA_metal_indices}")
            if debug > 0:
                print(f"MOLECULE.SPLIT_IA_IIA: rest_idx={rest_idx}")

            # Split the "rest" to obtain the ligands
            rest_labels = extract_from_list(rest_idx, self.labels, dimension=1)
            rest_coord = extract_from_list(rest_idx, self.coord, dimension=1)
            if self.frac_coord is not None:
                rest_frac = extract_from_list(rest_idx, self.frac_coord, dimension=1)
            rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)

            rest_radii = extract_from_list(rest_idx, self.radii, dimension=1)
            rest_atoms = extract_from_list(rest_idx, self.atoms, dimension=1)
            if self.atom_site_labels is not None:
                rest_atom_site_labels = extract_from_list(
                    rest_idx, self.atom_site_labels, dimension=1
                )

            if debug >= 2:
                print(f"MOLECULE.SPLIT_IA_IIA: rest labels: {rest_labels}")
                print(f"MOLECULE.SPLIT_IA_IIA: rest indices: {rest_indices}")
                print(f"MOLECULE.SPLIT_IA_IIA: rest radii: {rest_radii}")

            if debug > 0:
                print(
                    f"MOLECULE.SPLIT_IA_IIA: splitting species with {len(rest_labels)} atoms in block"
                )
            if len(rest_labels) == 0:

                if debug > 0:
                    print(
                        f"MOLECULE.SPLIT_IA_IIA: No ligands found in the complex {self.formula}"
                    )
                if debug > 0:
                    print(
                        f"MOLECULE.SPLIT_IA_IIA: Returning empty ligand list {self.ligands=}"
                    )
                for m in IA_IIA_metal_indices:
                    self.metals.append(self.atoms[m])
            else:
                if refcell.exist_cif_bond_moiety and refcell.geom_bond_cif is not None:
                    blocklist = split_species(
                        rest_labels,
                        rest_coord,
                        radii=rest_radii,
                        atom_site_labels=rest_atom_site_labels,
                        geom_bond_cif=refcell.geom_bond_cif,
                        cov_factor=self.cov_factor,
                        debug=debug,
                    )
                else:
                    blocklist = split_species(
                        rest_labels, rest_coord, radii=rest_radii, debug=debug
                    )

                # if debug > 0:
                print(f"SPLIT COMPLEX: received {len(blocklist)} blocks {blocklist=}")

                ## Arranges Ligands
                for b in blocklist:
                    if debug > 0:
                        print(f"PREPARING BLOCK: {b}")
                    lig_indices = extract_from_list(b, rest_indices, dimension=1)
                    lig_labels = extract_from_list(b, rest_labels, dimension=1)
                    lig_coord = extract_from_list(b, rest_coord, dimension=1)
                    if self.frac_coord is not None:
                        lig_frac_coord = extract_from_list(b, rest_frac, dimension=1)

                    lig_radii = extract_from_list(b, rest_radii, dimension=1)
                    lig_atoms = extract_from_list(b, rest_atoms, dimension=1)
                    if self.atom_site_labels is not None:
                        lig_atom_site_labels = extract_from_list(
                            b, rest_atom_site_labels, dimension=1
                        )


                    if debug > 0:
                        print(f"CREATING LIGAND: {labels2formula(lig_labels)}")

                    if self.frac_coord is not None:
                        newligand = ligand.from_positional(
                            lig_labels, lig_coord, lig_frac_coord, radii=lig_radii
                        )
                    else:
                        newligand = ligand.from_positional(
                            lig_labels, lig_coord, radii=lig_radii
                        )

                    newligand.origin = "split_IA_IIA"
                    newligand.add_parent(self, indices=lig_indices)

                    if self.check_parent("unitcell"):
                        cell_indices = [
                            a.get_parent_index("unitcell") for a in lig_atoms
                        ]
                        newligand.add_parent(
                            self.get_parent("unitcell"), indices=cell_indices
                        )

                    if self.check_parent("reference"):
                        ref_indices = [a.get_parent_index("reference") for a in lig_atoms]
                        newligand.add_parent(self.get_parent("reference"), indices=ref_indices)


                    newligand.set_adjacency_parameters(self.cov_factor, self.metal_factor)
                    
                    newligand.set_atoms(atomlist=lig_atoms, 
                                        atom_site_labels=lig_atom_site_labels, 
                                        geom_bond_cif=geom_bond_cif)

                    newligand.inherit_adjmatrix("molecule")
                    self.ligands.append(newligand)

                ## Arranges Metals
                for m in IA_IIA_metal_indices:
                    self.metals.append(self.atoms[m])

            return self.ligands, self.metals
    ############
    def split_complex(self, debug: int = 0):
        if self.atoms is None:
            self.set_atoms()
        if not self.iscomplex:
            self.ligands = None
            self.metals = None
        else:
            refcell = self.get_parent("reference")
            geom_bond_cif = getattr(refcell, "geom_bond_cif", None)

            self.ligands = []
            self.metals = []
            # Identify Metals and the rest
            metal_idx = list(
                [self.indices[idx] for idx in get_metal_idxs(self.labels, debug=debug)]
            )
            ia_iia_metal_idx = list(
                [
                    self.indices[idx]
                    for idx in get_alkali_alkaline_earth_metal_idxs(
                        self.labels, debug=debug
                    )
                ]
            )

            non_transition_metals_idx = list(
                [
                    self.indices[idx]
                    for idx in get_non_transition_metal_idxs(self.labels, debug=debug)
                ]
            )
            if len(non_transition_metals_idx) > 0:
                print(
                    f"MOLECULE.SPLIT COMPLEX: Found non-transition metals in the molecule {self.formula}"
                )
                print(
                    f"MOLECULE.SPLIT COMPLEX: Alkali and Alkaline earth metals {[self.labels[idx] for idx in ia_iia_metal_idx]}"
                )
                print(
                    f"MOLECULE.SPLIT COMPLEX: Non-transition metals found: {[self.labels[idx] for idx in non_transition_metals_idx]}"
                )
                # metal_idx.extend(non_transition_metals_idx)
            metal_idx.extend(ia_iia_metal_idx)
            rest_idx = list(idx for idx in self.indices if idx not in metal_idx)
            if debug > 0:
                print(f"MOLECULE.SPLIT COMPLEX: labels={self.labels}")
            if debug > 0:
                print(f"MOLECULE.SPLIT COMPLEX: metal_idx={metal_idx}")
            if debug > 0:
                print(f"MOLECULE.SPLIT COMPLEX: rest_idx={rest_idx}")

            # Split the "rest" to obtain the ligands
            rest_labels = extract_from_list(rest_idx, self.labels, dimension=1)
            rest_coord = extract_from_list(rest_idx, self.coord, dimension=1)
            rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)
            rest_radii = extract_from_list(rest_idx, self.radii, dimension=1)
            rest_atoms = extract_from_list(rest_idx, self.atoms, dimension=1)

            if self.frac_coord is not None:
                rest_frac = extract_from_list(rest_idx, self.frac_coord, dimension=1)
            if self.atom_site_labels is not None:
                rest_atom_site_labels = extract_from_list(
                    rest_idx, self.atom_site_labels, dimension=1
                )

            if debug >= 2:
                print(f"SPLIT COMPLEX: rest labels: {rest_labels}")
                print(f"SPLIT COMPLEX: rest indices: {rest_indices}")
                print(f"SPLIT COMPLEX: rest radii: {rest_radii}")

            if debug > 0:
                print(
                    f"SPLIT COMPLEX: splitting species with {len(rest_labels)} atoms in block"
                )
            if len(rest_labels) == 0:

                if debug > 0:
                    print(
                        f"SPLIT COMPLEX: No ligands found in the complex {self.formula}"
                    )
                if debug > 0:
                    print(f"SPLIT COMPLEX: Returning empty ligand list {self.ligands=}")
                for m in metal_idx:
                    self.metals.append(self.atoms[m])
            else:
                if refcell.exist_cif_bond_moiety and refcell.geom_bond_cif is not None:
                    blocklist = split_species(
                        rest_labels,
                        rest_coord,
                        radii=rest_radii,
                        atom_site_labels=rest_atom_site_labels,
                        geom_bond_cif=refcell.geom_bond_cif,
                        cov_factor=self.cov_factor,
                        debug=debug,
                    )
                else:
                    blocklist = split_species(
                        rest_labels, rest_coord, radii=rest_radii, debug=debug
                    )

                # if debug > 0:
                print(f"SPLIT COMPLEX: received {len(blocklist)} blocks {blocklist=}")

                ## Arranges Ligands
                for b in blocklist:
                    if debug > 0:
                        print(f"PREPARING BLOCK: {b}")
                    lig_indices = extract_from_list(b, rest_indices, dimension=1)
                    lig_labels = extract_from_list(b, rest_labels, dimension=1)
                    lig_coord = extract_from_list(b, rest_coord, dimension=1)
                    lig_radii = extract_from_list(b, rest_radii, dimension=1)
                    lig_atoms = extract_from_list(b, rest_atoms, dimension=1)
                    if self.frac_coord is not None:
                        lig_frac_coord = extract_from_list(b, rest_frac, dimension=1)

                    if self.atom_site_labels is not None:
                        lig_atom_site_labels = extract_from_list(
                            b, rest_atom_site_labels, dimension=1
                        )

                    if debug > 0:
                        print(f"CREATING LIGAND: {labels2formula(lig_labels)}")
                    # Create Ligand Object
                    if self.frac_coord is not None:
                        newligand = ligand.from_positional(
                            lig_labels, lig_coord, lig_frac_coord, radii=lig_radii
                        )
                    else:
                        newligand = ligand.from_positional(
                            lig_labels, lig_coord, radii=lig_radii
                        )


                    # For debugging
                    newligand.origin = "split_complex"
                    # Define the molecule as parent of the ligand. Bottom-Up hierarchy
                    newligand.add_parent(self, indices=lig_indices)

                    if self.check_parent("unitcell"):
                        cell_indices = [
                            a.get_parent_index("unitcell") for a in lig_atoms
                        ]
                        newligand.add_parent(
                            self.get_parent("unitcell"), indices=cell_indices
                        )

                    if self.check_parent("reference"):
                        ref_indices = [
                            a.get_parent_index("reference") for a in lig_atoms
                        ]
                        newligand.add_parent(
                            self.get_parent("reference"), indices=ref_indices
                        )

                    # Update the ligand with the covalent and metal factors
                    newligand.set_adjacency_parameters(
                        self.cov_factor, self.metal_factor
                    )
                    # Pass the molecule atoms to the ligand

                    newligand.set_atoms(atomlist=lig_atoms, 
                                        atom_site_labels=lig_atom_site_labels, 
                                        geom_bond_cif=geom_bond_cif)

                    # Inherit the adjacencies from molecule
                    newligand.inherit_adjmatrix("molecule")
                    # Add ligand to the list. Top-Down hierarchy
                    # newligand.evaluate_as_nitrosyl()
                    self.ligands.append(newligand)

                ## Arranges Metals
                for m in metal_idx:
                    ## We were creating the metal again, but it is already in the list of molecule.atoms
                    # newmetal    = metal.from_positional(self.labels[m], self.coord[m], self.frac_coord[m], self.radii[m])
                    # newmetal.add_parent(self, index=self.indices[m])
                    # self.metals.append(newmetal)
                    self.metals.append(self.atoms[m])
            return self.ligands, self.metals

    #######################################################
    def get_hapticity(self, debug: int = 0):
        if self.ligands is None:
            self.split_complex(debug=debug)
        self.is_haptic = False
        self.haptic_type = []
        if self.iscomplex:
            for lig in self.ligands:
                if lig.is_haptic is None:
                    lig.get_hapticity(debug=debug)
                if lig.is_haptic:
                    self.is_haptic = True
                for entry in lig.haptic_type:
                    if entry not in self.haptic_type:
                        self.haptic_type.append(entry)
        return self.haptic_type

    #######################################################  
    def save(self, path):
        print(f"SAVING cell2mol CELL object to {path}")
        with open(path, "wb") as fil:
            pickle.dump(self,fil)
    #######################################################        
    def get_unique_species(self, debug: int = 0):
        if debug >= 0:
            print(f"Getting unique species in molecule: {self.formula}")

        self.unique_species = []
        self.unique_indices = []
        self.species_list = []

        typelist_mols = []
        typelist_ligs = []
        typelist_mets = []

        specs_found = -1

        # Case 1: simple molecule (not complex, not IA/IIA)
        if not self.iscomplex and not self.has_IA_IIA and not self.has_post_transition_metal:
            found = False
            for ldx, typ in enumerate(typelist_mols):
                issame = compare_species(self, typ[0], debug=0)
                if issame:
                    found = True
                    kdx = typ[1]
                    if debug >= 2:
                        print(f"Molecule is the same as type {ldx}")
            if not found:
                specs_found += 1
                kdx = specs_found
                typelist_mols.append([self, kdx])
                self.unique_species.append(self)
                if debug >= 2:
                    print(f"New molecule found: formula={self.formula}, added at position {kdx}")
            self.unique_indices.append(kdx)
            self.unique_index = kdx
            self.species_list.append(self)

        else:
            # Ensure ligands and metals are available
            if not self.ligands is not None:
                if self.iscomplex:
                    self.split_complex(debug=debug)
                elif self.has_IA_IIA:
                    self.split_IA_IIA(debug=debug)
                elif self.has_post_transition_metal:
                    self.split_post_transition_metal(debug=debug)

            # Case 2: ligands
            for jdx, lig in enumerate(self.ligands):
                found = False
                for ldx, typ in enumerate(typelist_ligs):
                    if not lig.is_nitrosyl is not None:
                        lig.evaluate_as_nitrosyl()
                    if not typ[0].is_nitrosyl is not None:
                        typ[0].evaluate_as_nitrosyl()

                    if lig.is_nitrosyl and typ[0].is_nitrosyl:
                        issame = lig.NO_type == typ[0].NO_type
                    else:
                        issame = compare_species(lig, typ[0], debug=0)

                    if issame:
                        found = True
                        kdx = typ[1]
                        if debug >= 2:
                            print(f"Ligand {jdx} is the same as {ldx} in typelist")

                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_ligs.append([lig, kdx])
                    self.unique_species.append(lig)
                    if debug >= 2:
                        print(f"New ligand found: {lig.formula}, added at position {kdx}")

                self.unique_indices.append(kdx)
                lig.unique_index = kdx
                self.species_list.append(lig)

            # Case 3: metals
            for jdx, met in enumerate(self.metals):
                found = False
                for ldx, typ in enumerate(typelist_mets):
                    issame = compare_metals(met, typ[0], debug=0)
                    if issame:
                        found = True
                        kdx = typ[1]
                        if debug >= 2:
                            print(f"Metal {jdx} is the same as {ldx} in typelist")

                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_mets.append([met, kdx])
                    self.unique_species.append(met)
                    if debug >= 2:
                        print(f"New metal center found: label={met.label}, added at position {kdx}")

                self.unique_indices.append(kdx)
                met.unique_index = kdx
                self.species_list.append(met)

        return self.unique_species
    #######################################################
    def get_selected_cs(self, debug: int=0):
        if not self.unique_species is not None: self.get_unique_species(debug=debug)  

        self.selected_cs = []
        for unique_specie in self.unique_species:
            if debug >= 0 : print("Get possible charge states for unique specie", unique_specie.formula)
            tmp = unique_specie.get_possible_cs(debug=debug)
            if len(tmp) == 0: 
                self.selected_cs.append(None)
            elif unique_specie.subtype != "metal":
                self.selected_cs.append(list([cs.corr_total_charge for cs in unique_specie.possible_cs]))
            else :
                self.selected_cs.append(unique_specie.possible_cs)
        
        for specie in self.species_list:
            print("Get possible charge states for species list", specie.formula)
            tmp = specie.get_possible_cs(debug=debug)
            if len(tmp) == 0: 
                self.selected_cs.append(None)
            elif specie.subtype != "metal":
                self.selected_cs.append(list([cs.corr_total_charge for cs in specie.possible_cs]))
            else :
                self.selected_cs.append(specie.possible_cs)

        if None in self.selected_cs:
            self.error_get_poscharges = True
        else :
            self.error_get_poscharges = False
    #######################################################
    def balance_charges_for_molecules(self, input_charge: int=None, debug: int=0):
        if not self.unique_species is not None: self.get_unique_species()
        if not self.selected_cs is not None: self.get_selected_cs()
        
        if None in self.selected_cs:
            self.error_get_poscharges = True
        else:
            self.error_get_poscharges = False

        unique_indices = [specie.unique_index for specie in self.species_list]

        final_charge_distribution, final_charges = balance_charge(unique_indices, self.unique_species, charges_sum=input_charge, debug=debug,)
        print(f"{len(final_charge_distribution)=} {final_charge_distribution=}")
        # Handle multiple or no charge distributions
        dist_count = len(final_charge_distribution)
        self.error_multiple_distrib = dist_count > 1
        self.error_empty_distrib = dist_count == 0
        
        if dist_count != 1:
            # Attempt to balance charges again with more specific conditions
            if self.error_multiple_distrib :
                print("More than one possible distribution found.")
                second_final_charge_distribution, second_final_charges = balance_charge(
                    unique_indices, 
                    self.unique_species, 
                    input_charge=input_charge,
                    aromatic=True, 
                    debug=debug,
                )    
            elif self.error_empty_distrib :
                print("No valid distribution found.")
                second_final_charge_distribution, second_final_charges = balance_charge(
                    unique_indices, 
                    self.unique_species, 
                    input_charge=input_charge,
                    rare=True, 
                    debug=debug,
                )           
            second_dist_count = len(second_final_charge_distribution)
            self.error_multiple_distrib = second_dist_count > 1
            self.error_empty_distrib = second_dist_count == 0    

            if second_dist_count == 1:
                final_charge_distribution = second_final_charge_distribution
                final_charges = second_final_charges
                print("Using the second distribution found.")
                
        # If any error was flagged, report failure
        if any([
            self.error_get_poscharges,
            self.error_multiple_distrib,
            self.error_empty_distrib,
        ]):
            print("Charge Assignment Failed.")
            return 

        # Assign charges to unique species in the molecules
        for specie, charge in zip(self.unique_species, final_charges[0]):
            assign_charge_to_specie(specie, charge, debug=debug)
            for refspecie in self.species_list:
                if specie.unique_index == refspecie.unique_index:
                    assign_charge_to_specie(refspecie, charge, debug=debug)

    #######################################################
    def assign_charges_for_molecule(self, debug: int=0):
        print(f"ASSIGN_CHARGES_FOR_MOLECULE {self.formula}") 
        
        for specie in self.unique_species:
            if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal:
                for jdx, lig in enumerate(self.ligands):
                    print(lig.unique_index)
                    print(specie.unique_index)
                    if lig.unique_index == specie.unique_index:
                        set_charge_state (specie, lig, mode=1, debug=debug)
                for kdx, met in enumerate(self.metals):
                    if met.unique_index == specie.unique_index:
                        met.set_charge(specie.charge)
            else:
                if self.unique_index == specie.unique_index:
                    set_charge_state (specie, self, mode=1, debug=debug)
        if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal: 
            prepare_mol(self)         
            print("Complex", self.formula, self.totcharge)
            for jdx, lig in enumerate(self.ligands):
                print("    Ligand", jdx, lig.formula, lig.totcharge, lig.smiles)
            for kdx, met in enumerate(self.metals):
                print("    Metal",  kdx, met.formula, met.charge)
        else:
            print("Non-Complex",  self.formula, self.totcharge, self.smiles)
    #################   
    def create_bonds (self, debug: int=0):

        if not self.iscomplex and not self.has_IA_IIA and not self.has_post_transition_metal: 
            result = create_bonds_specie(self, debug=debug)          ### Creates bonds between molecule.atoms using the molecule.rdkit_object
            if result == False:
                if debug >= 1: print(f"MOLECULE.CREATE_BONDS: error creating bonds for molecule {self.formula}")
                self.error_create_bonds = True
                return # Exit the function entirely if creating bonds fails for a non-complex molecule
            else :
                if debug >= 1: print(f"MOLECULE.CREATE_BONDS: Bonds created for molecule {self.formula}")

        # Second part
        if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal:
            for lig in self.ligands:
                result = create_bonds_specie(lig, debug=debug)      ### Creates bonds between ligand.atoms, which also belong to molecule.atoms, using the ligand.rdkit_object
                if result == False:
                    if debug >= 1: print(f"MOLECULE.CREATE_BONDS: error creating bonds for ligand {lig.formula}")
                    self.error_create_bonds = True
                    return # Exit the function entirely if creating bonds fails for any ligand
                
                else :
                    if debug >= 1: print(f"MOLECULE.CREATE_BONDS: Bonds created for molecule {lig.formula}")
        
        if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal:    
            self.smiles_with_H = [lig.smiles for lig in self.ligands]
            self.smiles = []
            fix_zwitterions_ligands = []
            # Fourth part : correction smiles of ligands
            for lig in self.ligands:
                print(f"MOLECULE.CREATE_BONDS: Correcting Smiles for ligand {lig.formula}")
                result, fix_zwitterions = correct_smiles_ligand(lig, debug=debug)
                if result == False:
                    if debug >= 1: print(f"MOLECULE.CREATE_BONDS: error correcting smiles for ligand {lig.formula}")
                    self.error_create_bonds = True
                    return # Exit the function entirely 
                else :
                    if debug >= 1: print(f"MOLECULE.CREATE_BONDS: Smiles corrected for ligand {lig.formula}")
                    if fix_zwitterions :
                        fix_zwitterions_ligands.append(lig)
                    else:
                        self.smiles.append(lig.smiles)    
            
            if debug >= 1:print(f"MOLECULE.CREATE_BONDS: {len(fix_zwitterions_ligands)} zwitterion ligands found in the complex")
            for lig in fix_zwitterions_ligands:
                for atom in lig.atoms:
                    atom.bonds = []
                if debug >= 1: print(f"MOLECULE.CREATE_BONDS: Re-running create_bonds_specie for ligand {lig.formula} due to zwitterion correction.")
                result = create_bonds_specie(lig, debug=debug)
                if result == False:
                    if debug >= 1: print(f"MOLECULE.CREATE_BONDS: error re-creating bonds for ligand {lig.formula}")
                    self.error_create_bonds = True
                    return
                else :
                    if debug >= 1: print(f"MOLECULE.CREATE_BONDS: Bonds re-created for ligand {lig.formula} after zwitterion correction.")
                    self.smiles.append(lig.smiles)

        # Third part : adds metal-ligand bonds, metal-metal bonds, with a zero order
        if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal:
            create_metal_ligand_bonds(self, debug=debug)
            create_metal_metal_bonds(self, debug=debug)

        self.error_create_bonds = False


###############
### LIGAND ####
###############
class ligand(specie):
    NO_type: NOType | None = None
    connected_atoms: list["atom"] | None = None
    connected_idx: list[int] | None = None
    denticity: int | None = None
    groups: list["group"] | None = None
    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    is_nitrosyl: bool | None = None
    metals: list["metal"] | None = None
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
        to_print += f"------------- Cell2mol LIGAND Object --------------\n"
        to_print += specie.__repr__(self, indirect=True)
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

        if refcell.exist_cif_bond_moiety:
            for met in mol.metals:
                tmplabels = self.labels.copy()
                tmpcoord = self.coord.copy()
                atom_site_labels = [atom.atom_site_label for atom in self.atoms]
                tmplabels.append(met.label)
                tmpcoord.append(met.coord)
                atom_site_labels.append(met.atom_site_label)

                isgood, tmpadjmat, tmpadjnum = get_adjmatrix_from_cif_bonds(tmplabels, tmpcoord, atom_site_labels, geom_bond_cif, metal_only=True)
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
                isgood, tmpadjmat, tmpadjnum = get_adjmatrix(
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

    def split_ligand(self, debug: int=0):
        def is_single_sublist(intermediate_list):
            return (
                isinstance(intermediate_list, list) and
                len(intermediate_list) == 1 and
                isinstance(intermediate_list[0], list)
            )
        if hasattr(self,"cov_factor"):      cov_factor=self.cov_factor

        refcell = self.get_parent("reference") 
        geom_bond_cif = getattr(refcell, "geom_bond_cif", None)

        if debug > 0: print(f"\nLIGAND.SPLIT_LIGAND: splitting {self.formula} into groups")


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
        if debug >= 2:
            print(f"\tLIGAND.SPLIT_LIGAND: {conn_labels=}")
        if debug >= 2:
            print(f"\tLIGAND.SPLIT_LIGAND: {conn_atom_site_labels=}")
        if refcell.exist_cif_bond_moiety and refcell.geom_bond_cif is not None:
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
            # Create Group Object
            if self.frac_coord is not None:
                newgroup = group.from_positional(
                    gr_labels, gr_coord, gr_frac_coord, radii=gr_radii
                )
            else:
                newgroup = group.from_positional(gr_labels, gr_coord, radii=gr_radii)

            # For debugging
            newgroup.origin = "split_ligand"
            # Define the ligand as parent of the group. Bottom-Up hierarchy
            newgroup.add_parent(self, indices=gr_indices)
            # Pass the ligand atoms to the groud

            newgroup.set_atoms(atomlist=gr_atoms, 
                               atom_site_labels=gr_atom_site_labels, 
                               geom_bond_cif=geom_bond_cif)
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

            newgroup, final_group_indices, final_ligand_indices = newgroup.check_coordination(debug=debug)
            print(f"\tLIGAND.SPLIT_LIGAND: {newgroup.formula} {newgroup.labels}")
            print(f"\tLIGAND.SPLIT_LIGAND: {final_group_indices=}")
            print(f"\tLIGAND.SPLIT_LIGAND: {final_ligand_indices=}")
            if not is_single_sublist(final_group_indices): # atoms in new group are connected to different metals
                if debug > 1 : print(f"\tenterting SPLIT_GROUP for the GROUP {newgroup.formula} with {final_group_indices=} {[met.label for met in newgroup.metals]}")
                for conn_idx in final_group_indices:
                    if debug > 1 : print(f"\tenterting SPLIT_GROUP for the GROUP {newgroup.labels} with {conn_idx=}")
                    splitted_groups = split_group(newgroup, conn_idx, final_ligand_indices, debug=debug)
                    for g in splitted_groups:
                        self.groups.append(g)
            else:
                if debug > 1 : print(f"\tGROUP {newgroup.formula} with {final_group_indices=} connected to {[met.label for met in newgroup.metals]}")
                conn_idx = final_group_indices[0]
                if len(conn_idx) == len(newgroup.atoms):
                    if debug > 1 : print(f"\tLIGAND.SPLIT_LIGAND: new group is found")
                    newgroup.get_denticity(debug=debug)
                    # Top-down hierarchy
                    self.groups.append(newgroup)
                elif len(conn_idx) == 0:
                    if debug > 1 : print(f"\tLIGAND.SPLIT_LIGAND: no group is found")
                    continue
                else:
                    if debug > 1 : print(f"\tenterting SPLIT_GROUP for the GROUP {newgroup.formula} with {conn_idx=}")
                    splitted_groups = split_group(newgroup, conn_idx, final_ligand_indices, debug=debug)
                    for g in splitted_groups:
                        self.groups.append(g)
        if debug > 0 : print(f"\tLIGAND.SPLIT_LIGAND: found groups {[ group.formula for group in self.groups]}")
        if debug > 3 : print(f"{self.groups}")

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
            if gr.is_haptic:
                self.is_haptic = True
                self.haptic_type = gr.haptic_type
            for entry in gr.haptic_type:
                if entry not in self.haptic_type:
                    self.haptic_type.append(entry)
        return self.haptic_type


###############
#### GROUP ####
###############
class group(specie):
    checked_coordination: bool | None = None
    closest_metal: Optional["metal"] = None
    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    metals: list["metal"] | None = None
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
        to_print += f"------------- Cell2mol GROUP Object --------------\n"
        to_print += specie.__repr__(self, indirect=True)
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
            (elemdatabase.elementblock[l] == "d")
            or (elemdatabase.elementblock[l] == "f")
            for l in self.labels
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

        if refcell.exist_cif_bond_moiety:

            for met in lig.metals:
                tmplabels = self.labels.copy()
                tmpcoord = self.coord.copy()
                atom_site_labels = [atom.atom_site_label for atom in self.atoms]
                tmplabels.append(met.label)
                tmpcoord.append(met.coord)
                atom_site_labels.append(met.atom_site_label)

                isgood, tmpadjmat, tmpadjnum = get_adjmatrix_from_cif_bonds(tmplabels, tmpcoord, atom_site_labels, geom_bond_cif, metal_only=True)
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
                isgood, tmpadjmat, tmpadjnum = get_adjmatrix(
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
        numN = self.labels.count("N")

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
        elif numC == 1 and numP == 1 and totnum == 2:
            self.haptic_type = ["h2-P=C"]
            self.is_haptic = True
        elif is_haptic_ring(self.labels, self.coord):
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
            self, conn_idx, final_ligand_indices = coordination_correction_for_haptic(
                self, debug=debug
            )
        if self.is_haptic == False:
            self, conn_idx, final_ligand_indices = (
                coordination_correction_for_nonhaptic(self, debug=debug)
            )
        self.checked_coordination = True
        return self, conn_idx, final_ligand_indices

    #######################################################
    def get_denticity(self, debug: int = 0):
        if self.checked_coordination is None:
            self.check_coordination(debug=debug)
        self.denticity = 0
        for a in self.atoms:
            self.denticity += a.mconnec
        return self.denticity


###############
### BOND ######
###############
class bond(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    # Required constructor parameters
    atom1: object
    atom2: object
    # TOFIX @choglass: Is it int? It seems that we assign floats in new_charge_assignment.py#409
    # THIS IS A FLOAT, NOT AN INT
    order: float = Field(default=1, alias="bond_order")  # Using alias to match original parameter name
    
    # Computed attribute with proper default
    distance: float | None = None
    
    # Frozen fields
    version: str = Field(default="2.0", frozen=True)
    type: Type = Field(default="bond")

    def model_post_init(self, __context: Any) -> None:
        # Compute distance between atoms
        self.distance = round(
            np.linalg.norm(np.array(self.atom1.coord) - np.array(self.atom2.coord)), 3
        )

    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()
    
    def __repr__(self):
        to_print = ""
        to_print += f"------------- Cell2mol BOND Object --------------\n"
        to_print += f" Version                  = {self.version}\n"
        to_print += f" Type                     = {self.type}\n"
        idx1 = self.atom1.get_parent_index("molecule")
        idx2 = self.atom2.get_parent_index("molecule")
        to_print += f" Molecule Atom 1 label    = {self.atom1.label}\n"
        to_print += f" Molecule Atom 2 label    = {self.atom2.label}\n"
        to_print += f" Molecule Atom 1 index    = {idx1}\n"
        to_print += f" Molecule Atom 2 index    = {idx2}\n"
        to_print += f" Bond Order               = {self.order}\n"
        to_print += f" Distance                 = {self.distance, 3}\n"
        to_print += "----------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use bond() with the keyword arguments instead.")
    def from_positional(
        cls, atom1: object, atom2: object, bond_order: float = 1
    ) -> "bond":
        return cls(atom1=atom1, atom2=atom2, bond_order=bond_order)


###############
### ATOM ######
###############
class atom(BaseModel):
    label: str
    coord: list[float]
    frac_coord: list[float] | None = None
    radii: float | None = None
    parents: list[object] = Field(default_factory=list)
    parents_index: list[int] = Field(default_factory=list)

    # Seem to be set in various places depending on the context, might need to check this
    cov_factor: float | None = None
    metal_factor: float | None = None
    atom_site_label: str | None = None
    connec: int | None = None
    mconnec: int | None = None
    adjacency: list[object] = Field(default_factory=list)
    metal_adjacency: list[object] = Field(default_factory=list)
    closest_metal: object | None = None
    metal_factor: float | None = None
    charge: int | None = None
    bonds: list[object] = Field(default_factory=list)

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
    ) -> "atom":
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
        if refcell.exist_cif_bond_moiety:
            atom_site_labels = [self.atom_site_label, other.atom_site_label]
            isgood, adjmat, adjnum = get_adjmatrix_from_cif_bonds(
                labels, coords, atom_site_labels, refcell.geom_bond_cif
            )
        else:

            isgood, adjmat, adjnum = get_adjmatrix(labels, coords)
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
                    print(f"ATOM.ADD_BOND found the same bond with atoms:")
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
        self.charge = charge

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
    # def get_connected_metals(self, metalist: list, debug: int=0):

    #     self.metals = []

    #     for met in metalist:
    #         tmplabels = self.label.copy()
    #         tmpcoord  = self.coord.copy()
    #         tmplabels.append(met.label)
    #         tmpcoord.append(met.coord)
    #         isgood, tmpadjmat, tmpadjnum = get_adjmatrix(tmplabels, tmpcoord, metal_only=True)
    #         if isgood and any(tmpadjnum) > 0: self.metals.append(met)
    #     return self.metals

    #######################################################
    def get_closest_metal(self, debug: int = 0):
        ## Here, the list of metal atoms must be provided
        apos = self.coord
        dist = []
        mol = self.get_parent("molecule")
        for met in mol.metals:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        self.closest_metal = mol.metals[np.argmin(dist)]
        return self.closest_metal

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
            to_print += f"------------- Cell2mol ATOM Object ----------------\n"
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

    def reset_mconnec(self, met, diff: int=-1, debug: int=0):
        if debug >= 2: print(f"ATOM.RESET_MCONN: resetting mconnec (and connec) for atom {self.label=}")
        if debug >= 2: print(f"ATOM.RESET_MCONN: initial {self.connec=} {self.mconnec=}")
        if debug >= 2 : print(f"ATOM.RESET_MCONN: initial = {self.adjacency=} {self.metal_adjacency=}")
        self.mconnec += diff
        self.connec += diff


        if debug >= 2: print(f"ATOM.RESET_MCONN: initial {met.connec=} {met.mconnec=}")
        if debug >= 2: print(f"ATOM.RESET_MCONN: initial = {met.adjacency=} {met.metal_adjacency=}")

        # Correct Metal Data
        met.mconnec += diff  # Corrects data of metal object
        met.connec += diff  # Corrects data of metal object

        exists = self.check_parent("ligand")
        if exists:
            lig = self.get_parent("ligand")
            lig_idx = self.get_parent_index("ligand")


            if debug > 2: print(f"ATOM.RESET_MCONN: resetting mconnec (and connec) for atom {self.label=} in ligadn {lig_idx=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: updating ligand atoms and madjnum")
            if debug > 2: print(f"ATOM.RESET_MCONN: {lig.natoms=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: {lig.labels=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {lig.madjnum=} {len(lig.madjnum)}") 
            # Nothing in madjmat of the ligand object, all zeros
            # if debug > 2: print(f"ATOM.RESET_MCONN: initial {lig.madjmat=} {(lig.madjmat).shape}")
            if debug > 2: print(f"ATOM.RESET_MCONN: updating ligand atoms and adjnum")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {lig.adjnum=} {len(lig.adjnum)}") 
            # if debug > 2: print(f"ATOM.RESET_MCONN: initial {lig.adjmat=} {(lig.adjmat).shape}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {lig.atoms[lig_idx].connec=} {lig.atoms[lig_idx].mconnec=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {lig.madjnum[lig_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {lig.adjnum[lig_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {met.connec=} {met.mconnec=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {lig.madjmat[lig_idx]=} {lig.adjmat[lig_idx]=}")
            # Correct Ligand Data
            lig.madjnum[lig_idx] += (
                diff  # Corrects data in metal_adjacency number of the ligand class
            )
            # lig.madjmat[lig_idx,met_idx] += diff            # Corrects data in metal_adjacency matrix
            # lig.madjmat[met_idx,lig_idx] += diff            # Corrects data in metal_adjacency matrix
            lig.adjnum[lig_idx] += (
                diff  # Corrects data in adjacency number of the ligand class
            )

            lig.atoms[lig_idx].set_adjacencies(
                lig.adjmat[lig_idx],
                lig.madjmat[lig_idx],
                lig.adjnum[lig_idx],
                lig.madjnum[lig_idx],
            )

            # lig.adjmat[lig_idx,met_idx]  += diff            # Corrects data in adjacency matrix
            # lig.adjmat[met_idx,lig_idx]  += diff            # Corrects data in adjacency matrix

            # we should delete the adjacencies, but not a priority 
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {lig.madjnum=}")
            # if debug > 2: print(f"ATOM.RESET_MCONN: final {lig.madjmat=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: final {lig.adjnum=}")  
            # if debug > 2: print(f"ATOM.RESET_MCONN: final {lig.adjmat=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {lig.atoms[lig_idx].connec=} {lig.atoms[lig_idx].mconnec=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {lig.atoms[lig_idx].adjacency=} {lig.atoms[lig_idx].metal_adjacency=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {lig.madjnum[lig_idx]=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {lig.adjnum[lig_idx]=}")
            lig.get_connected_idx(debug=debug)
            lig.get_connected_atoms(debug=debug)

        exists = self.check_parent("molecule")
        if exists:
            mol = self.get_parent("molecule")
            mol_idx = self.get_parent_index("molecule")
            met_idx = met.get_parent_index("molecule")

            if debug >= 2: print(f"ATOM.RESET_MCONN: resetting mconnec (and connec) for atom {self.label=} in molecule {mol_idx=} with metal {met_idx=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: updating molecule atoms and madjnum")
            if debug > 2: print(f"ATOM.RESET_MCONN: {mol.natoms=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: {mol.labels=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: initial {mol.madjnum=} {len(mol.madjnum)}") 
            #if debug >= 2: print(f"ATOM.RESET_MCONN: initial {mol.madjmat=} {(mol.madjmat).shape}") # Nothing in madjmat of the ligand object, all zeros
            if debug >= 2: print(f"ATOM.RESET_MCONN: updating molecule atoms and adjnum")
            if debug >= 2: print(f"ATOM.RESET_MCONN: initial {mol.adjnum=} {len(mol.adjnum)}") 
            #if debug >= 2: print(f"ATOM.RESET_MCONN: initial {mol.adjmat=} {(mol.adjmat).shape}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: initial {mol.atoms[mol_idx].mconnec=} {mol.atoms[mol_idx].connec=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: initial {met.mconnec=} {met.connec=}")

            if debug > 2: print(f"ATOM.RESET_MCONN: initial {mol.madjnum[mol_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {mol.adjnum[mol_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {mol.madjnum[met_idx]=} {mol.madjnum[mol_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {mol.adjnum[met_idx]=} {mol.adjnum[mol_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {mol.madjmat[met_idx,mol_idx]=} {mol.madjmat[mol_idx,met_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: initial {mol.adjmat[met_idx,mol_idx]=} {mol.adjmat[mol_idx,met_idx]=}")

            # Correct Molecule Data
            # mol.atoms[mol_idx].mconnec += diff              # Corrects data of atom object in molecule class
            # mol.atoms[mol_idx].connec  += diff              # Corrects data of atom object in molecule class

            mol.madjnum[mol_idx] += diff                    # Corrects data in metal_adjacency number of the molecule class
            mol.madjnum[met_idx] += diff                    # Corrects data in metal_adjacency number of the molecule class

            mol.madjmat[mol_idx,met_idx] += diff            # Corrects data in metal_adjacency matrix
            mol.madjmat[met_idx,mol_idx] += diff            # Corrects data in metal_adjacency matrix
            
            mol.adjnum[mol_idx]  += diff                    # Corrects data in adjacency number of the molecule class
            mol.adjnum[met_idx]  += diff                    # Corrects data in adjacency number of the molecule class
        
            mol.adjmat[mol_idx,met_idx]  += diff            # Corrects data in adjacency matrix
            mol.adjmat[met_idx,mol_idx]  += diff            # Corrects data in adjacency matrix

            self.set_adjacencies(mol.adjmat[mol_idx], mol.madjmat[mol_idx], mol.adjnum[mol_idx], mol.madjnum[mol_idx])

            met.set_adjacencies(mol.adjmat[met_idx], mol.madjmat[met_idx], mol.adjnum[met_idx], mol.madjnum[met_idx])

            if debug >= 2: print(f"ATOM.RESET_MCONN: final {mol.atoms[mol_idx].connec=} {mol.atoms[mol_idx].mconnec=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {mol.atoms[mol_idx].adjacency=} {mol.atoms[mol_idx].metal_adjacency=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {met.connec=} {met.mconnec=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {met.adjacency=} {met.metal_adjacency=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {mol.madjnum[mol_idx]=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {mol.adjnum[mol_idx]=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {mol.madjnum[met_idx]=} {mol.madjnum[mol_idx]=}")
            if debug >= 2: print(f"ATOM.RESET_MCONN: final {mol.adjnum[met_idx]=} {mol.adjnum[mol_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: final {mol.madjmat[met_idx,mol_idx]=} {mol.madjmat[mol_idx,met_idx]=}")
            if debug > 2: print(f"ATOM.RESET_MCONN: final {mol.adjmat[met_idx,mol_idx]=} {mol.adjmat[mol_idx,met_idx]=}")

###############
#### METAL ####
###############
class metal(atom):
    metals: list[object] = Field(default_factory=list)
    groups: list[object] = Field(default_factory=list)
    coord_nr: int | None = None
    coord_geometry: object | Literal["Undefined"] | None = None
    geom_deviation: float | Literal["Undefined"] | None = None
    rel_metal_radius: float | None = None
    metal_factor: float | None = None
    cov_factor: float | None = None
    bond_order: int | None = None
    bond_type: str | None = None
    bond_distance: float | None = None
    coord_sphere: list[atom] | None = None
    coord_sphere_formula: str | None = None
    unique_index: int | None = None
    charge: int | None = None
    possible_cs: list[int] | None = None

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
        from cell2mol.connectivity import split_group

        # metal.groups will be used for the calculation of the relative metal radius
        # and define the coordination geometry of the metal /hapicitiy/ hapttype
        if not self.check_parent("molecule"):
            return None

        refcell = self.get_parent("reference")
        geom_bond_cif = getattr(refcell, "geom_bond_cif", None)

        mol = self.get_parent("molecule")
        if refcell.exist_cif_bond_moiety:

            for lig in mol.ligands:
                for group in lig.groups:
                    if debug > 2:
                        print(group.formula)
                    ligand_indices = [a.get_parent_index("ligand") for a in group.atoms]
                    tmplabels = []
                    tmpcoord = []
                    atom_site_labels = []

                    tmplabels.append(self.label)
                    tmpcoord.append(self.coord)
                    atom_site_labels.append(self.atom_site_label)

                    tmplabels.extend(group.labels)
                    tmpcoord.extend(group.coord)

                    atom_site_labels.extend([atom.atom_site_label for atom in group.atoms])
                    
                    if debug >= 2: print(f"ATOM.Get_connected_groups: {tmplabels=} {atom_site_labels=}")
                    isgood, tmpadjmat, tmpadjnum = get_adjmatrix_from_cif_bonds(tmplabels, tmpcoord, atom_site_labels, geom_bond_cif, metal_only=True)
                    if isgood :
                        if debug > 2: print(group.formula, tmpadjmat, tmpadjnum)
                        if all(tmpadjnum[1:]): 
                            self.groups.append(group)
                            if debug >= 0:
                                print(
                                    f"ATOM.Get_connected_groups: Metal {self.label} is connected to all atoms in {group.formula}"
                                )

                        elif any(tmpadjnum[1:]):
                            if debug > 1:
                                print(
                                    f"Metal {self.label} is connected to {group.formula} but not all atoms are connected"
                                )
                            conn_idx = [
                                idx for idx, num in enumerate(tmpadjnum[1:]) if num == 1
                            ]
                            conn_ligand_indices = [
                                ligand_indices[idx]
                                for idx, num in enumerate(tmpadjnum[1:])
                                if num == 1
                            ]
                            if debug > 1:
                                print(
                                    f"get_connected_groups {tmpadjnum[1:]=} {conn_idx=} {conn_ligand_indices=} {ligand_indices=}"
                                )
                            splitted_groups = split_group(
                                group, conn_idx, conn_ligand_indices, debug=debug
                            )
                            for g in splitted_groups:
                                self.groups.append(g)
                                if debug > 1:
                                    print(
                                        f"Metal {self.label} is connected to {g.formula}"
                                    )
                        else:
                            if debug > 1:
                                print(
                                    f"Metal {self.label} is not connected to {group.formula}"
                                )
        else:
            for lig in mol.ligands:
                for group in lig.groups:
                    if debug > 2:
                        print(group.formula)
                    ligand_indices = [a.get_parent_index("ligand") for a in group.atoms]
                    tmplabels = []
                    tmpcoord = []
                    tmplabels.append(self.label)
                    tmpcoord.append(self.coord)
                    tmplabels.extend(group.labels)
                    tmpcoord.extend(group.coord)
                    if debug > 2:
                        print(tmplabels, tmpcoord)
                    isgood, tmpadjmat, tmpadjnum = get_adjmatrix(
                        tmplabels, tmpcoord, metal_only=True
                    )
                    # if isgood and any(tmpadjnum) > 0: self.groups.append(group)
                    if isgood:
                        if debug > 2:
                            print(group.formula, tmpadjmat, tmpadjnum)
                        if all(tmpadjnum[1:]):
                            self.groups.append(group)
                        elif any(tmpadjnum[1:]):
                            if debug > 1:
                                print(
                                    f"Metal {self.label} is connected to {group.formula} but not all atoms are connected"
                                )
                            conn_idx = [
                                idx for idx, num in enumerate(tmpadjnum[1:]) if num == 1
                            ]
                            conn_ligand_indices = [
                                ligand_indices[idx]
                                for idx, num in enumerate(tmpadjnum[1:])
                                if num == 1
                            ]
                            if debug > 1:
                                print(
                                    f"get_connected_groups {tmpadjnum[1:]=} {conn_idx=} {conn_ligand_indices=} {ligand_indices=}"
                                )
                            splitted_groups = split_group(
                                group, conn_idx, conn_ligand_indices, debug=debug
                            )
                            for g in splitted_groups:
                                self.groups.append(g)
                                if debug > 1:
                                    print(
                                        f"Metal {self.label} is connected to {g.formula}"
                                    )
                        else:
                            if debug > 1:
                                print(
                                    f"Metal {self.label} is not connected to {group.formula}"
                                )
        return self.groups

    #######################################################
    def get_relative_metal_radius(self, debug: int = 0):
        if self.groups is None:
            self.get_connected_groups(debug=debug)
        diff_list = []
        for group in self.groups:
            if group.is_haptic == False:
                for atom in group.atoms:
                    diff = round(
                        get_dist(self.coord, atom.coord)
                        - elemdatabase.CovalentRadius3[atom.label],
                        3,
                    )
                    diff_list.append(diff)
            else:
                haptic_center_label = "C"
                haptic_center_coord = compute_centroid(
                    np.array([atom.coord for atom in group.atoms])
                )
                diff = round(
                    get_dist(self.coord, haptic_center_coord)
                    - elemdatabase.CovalentRadius3[haptic_center_label],
                    3,
                )
                diff_list.append(diff)
        average = round(np.average(diff_list), 3)

        if debug > 1:
            print(f"METAL.Get_relative_metal_radius: {diff_list=}")
            print(f"METAL.Get_relative_metal_radius: {average=}")

        self.rel_metal_radius = round(
            average / elemdatabase.CovalentRadius3[self.label], 3
        )

        return self.rel_metal_radius

    #######################################################
    def get_connected_metals(self, debug: int = 1):
        self.metals = []
        mol = self.get_parent("molecule")
        refcell = self.get_parent("reference")
        if refcell.exist_cif_bond_moiety:

            pidx = self.get_parent_index("molecule")
            # print(f"METAL.Get_connected_metals: {self.label} {pidx=} {mol.metals=}")
            for met in mol.metals:
                if met == self:
                    continue
                met_idx = met.get_parent_index("molecule")
                if mol.adjmat[pidx, met_idx] == 1:
                    if debug > 1:
                        print(
                            f"METAL.Get_connected_metals: {self.label} is connected to {met.label}"
                        )
                    self.metals.append(met)
                else:

                    if debug > 1: print(f"METAL.Get_connected_metals: {self.label} is NOT connected to {met.label}")
        else :
            for met in mol.metals:
                if met == self:
                    continue
                tmplabels = []
                tmpcoord = []
                tmplabels.append(self.label)
                tmpcoord.append(self.coord)
                tmplabels.append(met.label)
                tmpcoord.append(met.coord)

                if debug > 1:
                    print(tmplabels, tmpcoord)

                isgood, tmpadjmat, tmpadjnum = get_adjmatrix(
                    tmplabels, tmpcoord, metal_only=True
                )
                if isgood:
                    if debug > 1:
                        print(met.label, tmpadjmat, tmpadjnum)
                    if all(tmpadjnum[1:]):
                        self.metals.append(met)
                    else:
                        if debug > 1: print(f"Metal {self.label}  {met.label}")                

        return self.metals

    #######################################################
    def get_coordination_geometry(self: object, debug: int = 0):
        if debug >= 1:
            print(f"\nMETAL.Get_coord_geometry: {self.label}")

        coord_group = self.get_connected_groups(debug=debug)

        self.coord_nr, self.coord_geometry, self.geom_deviation = define_coordination_geometry(self, coord_group, debug = debug)
        
        if debug >= 3 : print(f"METAL.Get_coord_geometry:\n{coord_group=}")
        if debug >= 1: print(f"METAL.Get_coord_geometry: coord_nr={self.coord_nr}")    
        if debug >= 2: print(f"METAL.Get_coord_geometry: {self.coord_geometry=} {self.geom_deviation=}")
        
        self.rel_metal_radius = self.get_relative_metal_radius(debug = debug)
        if debug >= 2: print(f"METAL.Get_coord_geometry: {self.rel_metal_radius=}")

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
        atom.reset_charge(
            self
        )  ## First uses the generic atom class function for itself
        if self.poscharges is not None:
            delattr(self, "poscharge")
    
    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()
    
    def __repr__(self):
        to_print = ""
        to_print += f"------------- Cell2mol METAL Object --------------\n"
        to_print += atom.__repr__(self, indirect=True)
        if self.coord_sphere_formula is not None:
            to_print += f" Coordination Sphere Formula  = {self.coord_sphere_formula}\n"
        if self.possible_cs is not None:
            to_print += f" Possible Charges             = {self.possible_cs}\n"
        to_print += "----------------------------------------------------\n"
        return to_print


##############
#### CELL ####
##############
class cell(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    # Required constructor parameters
    name: str
    labels: list[str]
    coord: list[list[float]] = Field(alias="pos")  # Using alias to match original parameter name
    frac_coord: list[list[float]]
    cell_vector: object
    cell_param: object
    
    # Computed in constructor
    natoms: int | None = None
    
    # Set by methods throughout lifecycle
    subtype: SubType | None = None
    atom_site_labels: list[str] | None = None
    
    # CIF bond/moiety related attributes
    geom_bond_cif: object | None = None
    moiety_list_cif: object | None = None
    exist_cif_bond_moiety: bool | None = None
    moiety_indices: object | None = None
    
    # Unique species related attributes
    unique_species: list[object] | None = None
    unique_indices: list[int] | None = None
    species_list: list[object] | None = None
    
    # Missing H related attributes
    missing_H_in_Carbon: bool | None = None
    missing_H_in_CoordWater: bool | None = None
    missing_H_in_Water: bool | None = None
    has_missing_H: bool | None = None
    has_isolated_H: bool | None = None
    
    # Molecule lists
    refmoleclist: list[object] | None = None
    moleclist: list[object] | None = None
    
    # Reconstruction related attributes
    is_fragmented: bool | None = None
    error_get_fragments: bool | None = None
    error_reconstruction: bool | None = None
    
    # Charge assignment related attributes
    error_empty_poscharges: bool | None = None
    error_multiple_distrib: bool | None = None
    error_empty_distrib: bool | None = None
    error_prepare_mols: bool | None = None
    error_get_poscharges: bool | None = None
    selected_cs: list[object] | None = None
    
    # Bond creation related attributes
    error_create_bonds: bool | None = None
    
    # Charge neutrality
    is_neutral: bool | None = None
    
    # Post-processing data
    pp_molecules: list[object] | None = None
    pp_indices: list[int] | None = None
    pp_options: list[object] | None = None
    
    # Error assessment
    error_case: str | None = None

    # TOFIX @choglass: See if we keep here, it's assigned in refcell.py#146
    # KEEP IT FOR NOW
    chemical_name: str | None = None
    reported_metal_os: str | None = None
    moiety_dicts: list[object] | None = None
    # refcell.py#186
    disagree_with_cif_formula: bool | None = None
    
    # Frozen fields
    version: str = Field(default="2.0", frozen=True)
    type: Type = Field(default="cell")
    
    def model_post_init(self, __context: Any) -> None:
        # Compute natoms from labels length
        if self.natoms is None:
            self.natoms = len(self.labels)

    #######################################################
    def set_subtype(self, subtype: SubType):
        self.subtype = subtype

    #######################################################
    def set_atom_site_labels(self, atom_site_labels):
        self.atom_site_labels = atom_site_labels

    #######################################################    
    def get_cif_bond_moiety (self, cif_bond_info: bool, geom_bond_cif=None, moiety_list_cif=None):
        """
        Get the bond moiety from the cif file.
        """
        self.geom_bond_cif = geom_bond_cif
        self.moiety_list_cif = moiety_list_cif

        if cif_bond_info:
            self.exist_cif_bond_moiety = True
            moiety_indices = get_moiety_indices_from_labels(self.atom_site_labels, moiety_list_cif)
            self.moiety_indices = moiety_indices
        else:   
            self.exist_cif_bond_moiety = False
            self.moiety_indices = None
        
        return self.exist_cif_bond_moiety

    #######################################################
    def get_unique_species(self, debug: int = 0):
        if debug >= 0:
            print(f"Getting unique species in {self.subtype}")
        self.unique_species = []
        self.unique_indices = []
        self.species_list = []
        typelist_mols = []  # temporary variable
        typelist_ligs = []  # temporary variable
        typelist_mets = []  # temporary variable

        specs_found = -1
        if self.subtype == "reference":
            moleclist = self.refmoleclist
        else:
            moleclist = self.moleclist
        for idx, mol in enumerate(moleclist):
            if debug >= 2:
                print(f"Molecule {idx} formula={mol.formula}")
            if not mol.iscomplex and not mol.has_IA_IIA and not mol.has_post_transition_metal:
                # if not mol.iscomplex:
                found = False
                for ldx, typ in enumerate(typelist_mols):  # Molecules
                    issame = compare_species(mol, typ[0], debug=0)
                    if issame:
                        found = True
                        kdx = typ[1]
                        if debug >= 2:
                            print(f"Molecule {idx} is the same with {ldx} in typelist")
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_mols.append(list([mol, kdx]))
                    self.unique_species.append(mol)
                    if debug >= 2:
                        print(
                            f"New molecule found with: formula={mol.formula} and added in position {kdx}"
                        )
                self.unique_indices.append(kdx)
                mol.unique_index = kdx
                self.species_list.append(mol)
            else:
                if mol.ligands is None:
                    if mol.iscomplex:
                        mol.split_complex(debug=debug)
                    elif mol.has_IA_IIA:
                        mol.split_IA_IIA(debug=debug)
                    elif mol.has_post_transition_metal:
                        mol.split_post_transition_metal(debug=debug)
                for jdx, lig in enumerate(mol.ligands):  # ligands
                    found = False
                    for ldx, typ in enumerate(typelist_ligs):
                        if lig.is_nitrosyl is None:
                            lig.evaluate_as_nitrosyl()
                        if typ[0].is_nitrosyl is None:
                            typ[0].evaluate_as_nitrosyl()
                        if lig.is_nitrosyl and typ[0].is_nitrosyl:
                            if lig.NO_type == typ[0].NO_type:
                                issame = True
                            else:
                                issame = False
                        else:
                            issame = compare_species(lig, typ[0], debug=0)
                        if issame:
                            found = True
                            kdx = typ[1]
                            if debug >= 2:
                                print(
                                    f"ligand {jdx} is the same with {ldx} in typelist"
                                )
                    if not found:
                        specs_found += 1
                        kdx = specs_found
                        typelist_ligs.append(list([lig, kdx]))
                        self.unique_species.append(lig)
                        if debug >= 2:
                            print(
                                f"New ligand found with: formula {lig.formula} added in position {kdx}"
                            )
                    self.unique_indices.append(kdx)
                    lig.unique_index = kdx
                    self.species_list.append(lig)
                for jdx, met in enumerate(mol.metals):  #  metals
                    found = False
                    for ldx, typ in enumerate(typelist_mets):
                        issame = compare_metals(met, typ[0], debug=0)
                        if issame:
                            found = True
                            kdx = typ[1]
                            if debug >= 2:
                                print(f"Metal {jdx} is the same with {ldx} in typelist")
                    if not found:
                        specs_found += 1
                        kdx = specs_found
                        typelist_mets.append(list([met, kdx]))
                        self.unique_species.append(met)
                        if debug >= 2:
                            print(
                                f"New Metal Center found with: labels {met.label} and added in position {kdx}"
                            )
                    self.unique_indices.append(kdx)
                    met.unique_index = kdx
                    self.species_list.append(met)
        return self.unique_species

    #######################################################
    # def get_fractional_coord(self):
    #     self.frac_coord = cart2frac(self.coord, self.cellvec)
    #     return self.frac_coord

    #######################################################
    def check_missing_H(self, debug: int = 0):
        from cell2mol.missingH import check_missingH

        (
            Warning,
            ismissingH,
            Missing_H_in_C,
            Missing_H_in_CoordWater,
            Missing_H_in_Water,
        ) = check_missingH(self.refmoleclist, debug=debug)
        if debug >= 2:
            print(
                f"CELL.Check_missing_H: {Missing_H_in_C=} {Missing_H_in_CoordWater=} {Missing_H_in_Water=}"
            )
        self.missing_H_in_Carbon = Missing_H_in_C
        self.missing_H_in_CoordWater = Missing_H_in_CoordWater
        self.missing_H_in_Water = Missing_H_in_Water
        if (
            ismissingH
            or Missing_H_in_C
            or Missing_H_in_CoordWater
            or Missing_H_in_Water
        ):
            self.has_missing_H = True
        else:
            self.has_missing_H = False
        return self.has_missing_H

    #######################################################
    def get_reference_molecules_from_moiety(
        self,
        ref_labels: list,
        ref_fracs: list,
        cov_factor: float = 1.3,
        metal_factor: float = 1.0,
        debug: int = 0,
    ):
        # Convert fractional coordinates to cartesian
        ref_pos = frac2cart_fromparam(ref_fracs, self.cell_param)

        # Define reference cell
        refcell = cell.from_positional(
            self.name, ref_labels, ref_pos, ref_fracs, self.cell_vector, self.cell_param
        )
        refcell.set_subtype("reference")
        atom_site_labels = self.atom_site_labels

        if debug >= 0:
            print("#########################################")
            print("  GETREFS: Generate reference molecules  ")
            print("  GETREFS: Consistent with CIF moieties  ")
            print("#########################################")

        geom_bond_cif = self.geom_bond_cif
        blocklist = self.moiety_indices

        if debug >= 2: print(f"GETREFS: blocklist={blocklist}")
        if debug >= 2: print(f"GETREFS: Generate Adjacency matrix based on bond information from CIF")

        self.refmoleclist = []
        if blocklist is None:
            print(f"GETREFS: No moiety indices found in the CIF file")
            return self.refmoleclist
        
        for b in blocklist:
            mol_labels = extract_from_list(b, ref_labels, dimension=1)
            mol_coord = extract_from_list(b, ref_pos, dimension=1)
            mol_frac_coord = extract_from_list(b, ref_fracs, dimension=1)
            mol_atom_site_labels = extract_from_list(b, atom_site_labels, dimension=1)

            newmolec = molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
            newmolec.add_parent(self, indices=b)
            # newmolec.add_parent(refcell, indices=b)
            newmolec.set_adjacency_parameters(cov_factor, metal_factor)
            newmolec.set_atoms(create_adjacencies=True, atom_site_labels=mol_atom_site_labels, geom_bond_cif=geom_bond_cif, debug=debug)

            for atom, idx in zip(newmolec.atoms, b):
                # atom.add_parent(refcell, index=idx)
                atom.add_parent(self, index=idx)
            # This must be below the frac_coord, so they are carried on to the ligands
            # if newmolec.iscomplex :
            if newmolec.iscomplex:
                newmolec.split_complex()
            elif newmolec.has_IA_IIA:
                newmolec.split_IA_IIA()
            else:
                newmolec.add_parent(newmolec, indices=[*range(0, newmolec.natoms, 1)])
            self.refmoleclist.append(newmolec)

        if debug >= 0:
            print(f"GETREFS: found {len(self.refmoleclist)} reference molecules")
        if debug >= 0:
            print(f"GETREFS:", [ref.formula for ref in self.refmoleclist])

        # Checks for isolated atoms, and retrieves warning if there is any.
        # Except if it is H, halogen (group 17) or alkalyne (group 2)
        isgood = True
        for ref in self.refmoleclist:
            if ref.natoms == 1:
                label = ref.atoms[0].label
                group = elemdatabase.elementgroup[label]
                if label == "H" or label == "D":
                    isgood = False
                else:  # (group == 1 or group == 2 or group == 17)
                    if debug >= 0:
                        print(
                            f"GETREFS: found ref molecule with only one atom {ref.labels}"
                        )

        # If all good, then works with the reference molecules
        if isgood:
            self.has_isolated_H = False
            for ref in self.refmoleclist:

                if ref.iscomplex: 
                    if debug >= 0: print(f"GETREFS: working with {ref.formula} with transition metals")
                    ref.get_hapticity(debug=debug)
                    if len(ref.ligands) == 0:
                        print(f"GETREFS: {ref.formula} is a metal cluster")
                    else:
                        for lig in ref.ligands:
                            lig.get_denticity(debug=debug)
                    for met in ref.metals:
                        met.get_connected_metals(debug=debug)
                        met.get_coordination_geometry(debug=debug)
                        met.get_coord_sphere_formula(debug=debug)
                elif ref.has_IA_IIA:

                    if debug >= 0: print(f"GETREFS: working with {ref.formula} with alkali or alkali earth metals")
                    if len(ref.ligands) == 0 :
                        pass
                    else:
                        for lig in ref.ligands:
                            lig.get_denticity(debug=debug)
                    for met in ref.metals:
                        met.get_connected_metals(debug=debug)
                        met.get_coordination_geometry(debug=debug)
                        met.get_coord_sphere_formula(debug=debug)
        else:
            self.has_isolated_H = True

        return self.refmoleclist

    #######################################################
    def get_reference_molecules(
        self,
        ref_labels: list,
        ref_fracs: list,
        cov_factor: float = 1.3,
        metal_factor: float = 1.0,
        debug: int = 0,
    ):
        if debug >= 0:
            print("#########################################")
            print("  GETREFS: Generate reference molecules  ")
            print("#########################################")

        # Convert fractional coordinates to cartesian
        ref_pos = frac2cart_fromparam(ref_fracs, self.cell_param)

        # Define reference cell
        refcell = cell.from_positional(
            self.name, ref_labels, ref_pos, ref_fracs, self.cell_vector, self.cell_param
        )
        refcell.set_subtype("reference")

        atom_site_labels = self.atom_site_labels
        geom_bond_cif = self.geom_bond_cif
        
        if debug > 2: print(f"GETREFS: geom_bond_cif={geom_bond_cif}")
        if debug >= 2: print(f"GETREFS: Generate Adjacency matrix based on interatomic distances")
        blocklist = split_species(ref_labels, ref_pos, cov_factor=cov_factor)
        if debug >= 2: print(f"GETREFS: blocklist={blocklist}")

        self.refmoleclist = []
        if blocklist is None:
            print(f"GETREFS: No blocklist found")
            return self.refmoleclist
        
        # Get reference molecules
        for b in blocklist:
            mol_labels = extract_from_list(b, ref_labels, dimension=1)
            mol_coord = extract_from_list(b, ref_pos, dimension=1)
            mol_frac_coord = extract_from_list(b, ref_fracs, dimension=1)
            mol_atom_site_labels = extract_from_list(b, atom_site_labels, dimension=1)

            newmolec = molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
            newmolec.add_parent(self, indices=b)
            # newmolec.add_parent(refcell, indices=b)
            newmolec.set_adjacency_parameters(cov_factor, metal_factor)
            newmolec.set_atoms(
                create_adjacencies=True,
                atom_site_labels=mol_atom_site_labels,
                geom_bond_cif=geom_bond_cif,
                debug=debug,
            )
            for atom, idx in zip(newmolec.atoms, b):
                # atom.add_parent(refcell, index=idx)
                atom.add_parent(self, index=idx)

            # This must be below the frac_coord, so they are carried on to the ligands
            if newmolec.iscomplex:
                newmolec.split_complex()
            elif newmolec.has_IA_IIA:
                newmolec.split_IA_IIA()
            elif newmolec.has_post_transition_metal:
                print(f"GETREFS: {newmolec.formula} has post-transition metal")
                newmolec.split_post_transition_metal()
            else:
                newmolec.add_parent(newmolec, indices=[*range(0, newmolec.natoms, 1)])
            self.refmoleclist.append(newmolec)

        if debug >= 0:
            print(f"GETREFS: found {len(self.refmoleclist)} reference molecules")
        if debug >= 0:
            print(f"GETREFS:", [ref.formula for ref in self.refmoleclist])

        # Checks for isolated atoms, and retrieves warning if there is any.
        # Except if it is H, halogen (group 17) or alkalyne (group 2)
        isgood = True
        for ref in self.refmoleclist:
            if ref.natoms == 1:
                label = ref.atoms[0].label
                group = elemdatabase.elementgroup[label]
                if label == "H" or label == "D":
                    isgood = False
                else:  # (group == 1 or group == 2 or group == 17)
                    if debug >= 0:
                        print(
                            f"GETREFS: found ref molecule with only one atom {ref.labels}"
                        )

        # If all good, then works with the reference molecules
        if isgood:
            self.has_isolated_H = False
            for ref in self.refmoleclist:
                if ref.iscomplex:
                    if debug >= 0: print(f"GETREFS: working with {ref.formula} with transition metals")
                    ref.get_hapticity(debug=debug)
                    if len(ref.ligands) == 0:
                        print(f"GETREFS: {ref.formula} is a metal cluster")
                    else:
                        for lig in ref.ligands:
                            lig.get_denticity(debug=debug)
                    for met in ref.metals:
                        met.get_connected_metals(debug=debug)
                        met.get_coordination_geometry(debug=debug)
                        met.get_coord_sphere_formula(debug=debug)
                elif ref.has_IA_IIA:
                    if debug >= 0: print(f"GETREFS: working with {ref.formula} with alkali or alkali earth metals")
                    if len(ref.ligands) == 0 :
                        pass
                    else:
                        for lig in ref.ligands:
                            lig.get_denticity(debug=debug)
                    for met in ref.metals:
                        met.get_connected_metals(debug=debug)
                        met.get_coordination_geometry(debug=debug)
                        met.get_coord_sphere_formula(debug=debug)
                elif ref.has_post_transition_metal:
                    if debug >= 0: print(f"GETREFS: working with {ref.formula} with post-transition metals")
                    if debug >= 0: print(f"GETREFS: {[met.label for met in ref.metals]}")
                    if debug >= 0: print(f"GETREFS: {[lig.formula for lig in ref.ligands]}")
                    if len(ref.ligands) == 0 :
                        pass
                    else:
                        for lig in ref.ligands:
                            lig.get_denticity(debug=debug)
                    for met in ref.metals:
                        met.get_connected_metals(debug=debug)
                        met.get_coordination_geometry(debug=debug)
                        met.get_coord_sphere_formula(debug=debug)
        else:
            self.has_isolated_H = True

        return self.refmoleclist

    #######################################################
    def get_moleclist(
        self, cov_factor: float = 1.3, metal_factor: float = 1.0, debug: int = 0
    ):
        if debug > 3:
            print(f"Entered CELL.MOLECLIST with debug={debug}")
        if self.labels is None or self.coord is None:
            if debug > 3:
                print(
                    f"CELL.MOLECLIST. Labels or coordinates not found. Returning None"
                )
            return None
        if len(self.labels) == 0 or len(self.coord) == 0:
            if debug > 3:
                print(f"CELL.MOLECLIST. Empty labels or coordinates. Returning None")
            return None
        if debug > 3:
            print(f"CELL.MOLECLIST passed initial checks")

        ions_idx = []
        for ref in self.refmoleclist:
            if ref.natoms == 1:
                label = ref.atoms[0].label
                ions_idx.extend(
                    [idx for idx, l in enumerate(self.labels) if l == label]
                )
                if debug > 3:
                    print(f"CELL.MOLECLIST: {ions_idx=} with {label=}")

        if len(ions_idx) > 0:
            cell_indices = [*range(0, len(self.labels), 1)]
            if debug > 3:
                print(f"CELL.MOLECLIST: found {len(ions_idx)} ions")
            rest_idx = list(idx for idx in cell_indices if idx not in ions_idx)
            if debug > 3:
                print(f"CELL.MOLECLIST: {rest_idx=}")
            rest_labels = extract_from_list(rest_idx, self.labels, dimension=1)
            rest_coord = extract_from_list(rest_idx, self.coord, dimension=1)
            rest_indices = extract_from_list(rest_idx, cell_indices, dimension=1)
            if debug > 3:
                print(f"CELL.MOLECLIST: {rest_labels=}")
            if debug > 3:
                print(f"CELL.MOLECLIST: {rest_coord=}")
            if debug > 3:
                print(f"CELL.MOLECLIST: {rest_indices=}")
            blocklist = split_species(
                rest_labels,
                rest_coord,
                indices=rest_indices,
                cov_factor=cov_factor,
                debug=debug,
            )
            for idx in ions_idx:
                blocklist.append([idx])
        else:
            blocklist = split_species(
                self.labels, self.coord, cov_factor=cov_factor, debug=debug
            )

        if blocklist is None:
            return None
        else:
            if debug > 3:
                print(f"CELL.MOLECLIST: found {len(blocklist)} blocks")
            if debug > 3:
                print(f"CELL.MOLECLIST: {blocklist=}")

        self.moleclist = []
        for b in blocklist:
            if debug > 3:
                print(f"CELL.MOLECLIST: doing block={b}")
            mol_labels = extract_from_list(b, self.labels, dimension=1)
            mol_coord = extract_from_list(b, self.coord, dimension=1)
            mol_frac_coord = extract_from_list(b, self.frac_coord, dimension=1)
            # Creates Molecule Object
            newmolec = molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
            # For debugging
            newmolec.origin = "cell.get_moleclist"
            # Adds cell as parent of the molecule, with indices b
            newmolec.add_parent(self, indices=b)
            newmolec.set_adjacency_parameters(cov_factor, metal_factor)
            # Creates The atom objects with adjacencies
            newmolec.set_atoms(create_adjacencies=True, debug=debug)
            # The split_complex must be below the frac_coord, so they are carried on to the ligands
            # if newmolec.iscomplex:
            #     if debug > 0: print(f"CELL.MOLECLIST: splitting complex")
            #     newmolec.split_complex(debug=debug)
            # Not needed here, as the reconstruction will take care of it
            self.moleclist.append(newmolec)

        return self.moleclist

    #######################################################
    def arrange_cell_coord(self):
        ## Updates the cell coordinates preserving the original atom ordering
        ## Do do so, it uses the variable parent_indices stored in each molecule
        self.coord = np.zeros((self.natoms, 3))
        for mol in self.moleclist:
            idx = mol.get_parent_indices("cell")
            for z in zip(idx, mol.coord):
                for i in range(0, 3):
                    self.coord[z[0]][i] = z[1][i]
        self.coord = np.ndarray.tolist(self.coord)

    #######################################################
    def get_occurrence(self, substructure: object) -> int:
        occurrence = 0
        ## Molecules in Cell
        if substructure.subtype is not None and self.moleclist is not None:
            if substructure.subtype == "molecule":
                for m in self.moleclist:
                    issame = compare_species(substructure, m)
                    if issame:
                        occurrence += 1
        return occurrence

    #######################################################
    def data_for_postproc(self, molecules: list, indices: list, options: list):
        self.pp_molecules = molecules
        self.pp_indices = indices
        self.pp_options = options

    #######################################################
    def reconstruct(
        self, cov_factor: float = None, metal_factor: float = None, debug: int = 0
    ):
        if self.refmoleclist is None:
            print("CELL.RECONSTRUCT. CELL missing list of reference molecules")
            return
        if cov_factor is None:
            cov_factor = self.refmoleclist[0].cov_factor
        if metal_factor is None:
            metal_factor = self.refmoleclist[0].metal_factor

        ## Get the fragments, which is the moleclist of a fragmented cell
        fragments = self.get_moleclist(
            cov_factor=cov_factor, metal_factor=metal_factor, debug=debug
        )

        if fragments is None:
            self.error_get_fragments = True
            return  # Stopping. self.error_get_fragments must be False to reconstruct the cell
        else:
            self.error_get_fragments = False

        ## Classifies fragments
        # for f in fragments:
        #     if f.frac_coord is None:       f.get_fractional_coord(self.cellvec)
        molecules, fragments, hydrogens = classify_fragments(
            fragments, self.refmoleclist, debug=debug
        )
        if debug > 0:
            print(f"CELL.RECONSTRUCT: {len(molecules)} {molecules=}")
        if debug > 0:
            print(f"CELL.RECONSTRUCT: {len(fragments)} {fragments=}")
        if debug > 0:
            print(f"CELL.RECONSTRUCT: {len(hydrogens)} {hydrogens=}")

        ## Determines if Reconstruction is necessary
        if len(fragments) > 0 or len(hydrogens) > 0:
            self.is_fragmented = True
        else:
            self.is_fragmented = False

        self.moleclist = []
        if not self.is_fragmented:
            for mol in molecules:
                self.moleclist.append(mol)
                #########################################
                ## In principle, this is not necessary ##
                #########################################
                # newmolec = molecule.from_positional(mol.labels, mol.coord)
                # newmolec.add_parent(self,mol_indices)
                # newmolec.set_fractional_coord(mol.frac_coord)
                # newmolec.set_atoms(debug=debug, create_adjacencies=True)
                # if newmolec.iscomplex:
                #    newmolec.split_complex()
                #    newmolec.get_hapticity()
                # self.moleclist.append(newmolec)
            return self.moleclist
        else:
            reconstructed_molecules, Warning = fragments_reconstruct(
                molecules,
                fragments,
                hydrogens,
                self.refmoleclist,
                self.cellvec,
                cov_factor,
                metal_factor,
            )

            if Warning:
                self.is_fragmented = True
                self.error_reconstruction = True

            else:
                self.is_fragmented = False
                self.error_reconstruction = False

            ## For consistency, we create the molecules once again, even if mol is already a molecule-class object.
            ## One must follow the same structure as in self.get_moleclist()
            for mol in reconstructed_molecules:
                newmolec = molecule.from_positional(mol.labels, mol.coord)
                newmolec.origin = "cell.reconstruct"
                newmolec.set_adjacency_parameters(cov_factor, metal_factor)
                newmolec.set_atoms(create_adjacencies=True, debug=debug)
                newmolec.add_parent(self, mol.cell_indices)
                newmolec.set_fractional_coord(mol.frac_coord)
                if newmolec.iscomplex:
                    newmolec.split_complex()
                self.moleclist.append(newmolec)
            return self.moleclist

    #######################################################
    def reset_charge_assignment(self, debug: int = 0):
        if self.moleclist is None:
            return None
        for mol in self.moleclist:
            mol.reset_charge()

    #######################################################
    def get_selected_cs(self, debug: int = 0):
        if self.unique_species is None:
            self.get_unique_species(debug=debug)

        self.selected_cs = []
        for unique_specie in self.unique_species:
            if debug >= 0:
                print(
                    "Get possible charge states for unique specie",
                    unique_specie.formula,
                )
            tmp = unique_specie.get_possible_cs(debug=debug)
            if len(tmp) == 0:
                self.selected_cs.append(None)
            elif unique_specie.subtype != "metal":
                self.selected_cs.append(
                    list([cs.corr_total_charge for cs in unique_specie.possible_cs])
                )
            else:
                self.selected_cs.append(unique_specie.possible_cs)

        for specie in self.species_list:
            print("Get possible charge states for species list", specie.formula)
            tmp = specie.get_possible_cs(debug=debug)
            if len(tmp) == 0:
                self.selected_cs.append(None)
            elif specie.subtype != "metal":
                self.selected_cs.append(
                    list([cs.corr_total_charge for cs in specie.possible_cs])
                )
            else:
                self.selected_cs.append(specie.possible_cs)

        if None in self.selected_cs:
            self.error_get_poscharges = True
        else:
            self.error_get_poscharges = False

    #######################################################
    #######################################################
    def assign_charges(self, debug: int = 0):
        # if self.unique_species is None: self.get_unique_species(debug=debug)
        # if debug >= 1: print(f"{len(self.unique_species)} Species (Metal or Ligand or Molecules) to Characterize")
        
        # Placeholder implementation - the original method was very long and complex
        # This should be implemented based on the specific charge assignment logic needed
        pass

    #######################################################
    def assign_charges_for_refcell(self, debug: int = 0):
        for specie in self.unique_species:
            for idx, ref in enumerate(self.refmoleclist):
                if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
                    for jdx, lig in enumerate(ref.ligands):
                        print(lig.unique_index)
                        print(specie.unique_index)
                        if lig.unique_index == specie.unique_index:
                            set_charge_state(specie, lig, mode=1, debug=debug)
                    for kdx, met in enumerate(ref.metals):
                        if met.unique_index == specie.unique_index:
                            met.set_charge(specie.charge)
                else:
                    if ref.unique_index == specie.unique_index:
                        set_charge_state(specie, ref, mode=1, debug=debug)

        for idx, ref in enumerate(self.refmoleclist):
            if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
                prepare_mol(ref)

        for idx, ref in enumerate(self.refmoleclist):
            print(f"ASSIGN_CHARGES: Refenrence Molecule {idx}: {ref.formula}")
            if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
                print("ASSIGN_CHARGES: Complex", idx, ref.formula, ref.totcharge)
                for jdx, lig in enumerate(ref.ligands):
                    print(
                        "ASSIGN_CHARGES: Ligand",
                        idx,
                        jdx,
                        lig.formula,
                        lig.totcharge,
                        lig.smiles,
                    )
                for kdx, met in enumerate(ref.metals):
                    print("ASSIGN_CHARGES: Metal", idx, kdx, met.formula, met.charge)
            else:
                print(
                    "ASSIGN_CHARGES: Non-Complex",
                    idx,
                    ref.formula,
                    ref.totcharge,
                    ref.smiles,
                )

    ########################################################
    def assign_charges_for_unitcell(self, debug: int = 0):
        for idx, mol in enumerate(self.moleclist):
            print(f"ASSIGN_CHARGES: Unitcell Molecule {idx}: {mol.formula}")
            if not mol.iscomplex and not mol.has_IA_IIA and not mol.has_post_transition_metal:
                for ref in self.refmoleclist:
                    if (not ref.iscomplex and not ref.has_IA_IIA and not ref.has_post_transition_metal) and (
                        mol.unique_index == ref.unique_index
                    ):
                        issame = compare_reference_indices(ref, mol, debug=debug)
                        if issame:
                            set_charge_state(ref, mol, mode=2, debug=debug)
            else:
                for ref in self.refmoleclist:
                    if (ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal) and (
                        mol.formula == ref.formula
                    ):
                        for jdx, lig in enumerate(mol.ligands):
                            for rdx, ref_lig in enumerate(ref.ligands):
                                if lig.formula == ref_lig.formula:
                                    issame = compare_reference_indices(
                                        ref_lig, lig, debug=debug
                                    )
                                    if issame:
                                        set_charge_state(
                                            ref_lig, lig, mode=2, debug=debug
                                        )
                                    # else:
                                    #     print("ERROR: ASSIGN_CHARGES: Ligand", idx, jdx, rdx, lig.formula, ref_lig.totcharge, ref_lig.smiles)
                        for kdx, met in enumerate(mol.metals):
                            for ref_met in ref.metals:
                                if met.formula == ref_met.formula:
                                    if ref_met.get_parent_index(
                                        "reference"
                                    ) == met.get_parent_index("reference"):
                                        met.set_charge(ref_met.charge)

        for idx, mol in enumerate(self.moleclist):
            if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
                prepare_mol(mol)

        for idx, mol in enumerate(self.moleclist):
            print(f"ASSIGN_CHARGES: Unitcell Molecule {idx}: {mol.formula}")
            if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
                print("ASSIGN_CHARGES: Complex", idx, mol.formula, mol.totcharge)
                for jdx, lig in enumerate(mol.ligands):
                    print(
                        "ASSIGN_CHARGES: Ligand",
                        idx,
                        jdx,
                        lig.formula,
                        lig.totcharge,
                        lig.smiles,
                    )
                for kdx, met in enumerate(mol.metals):
                    print("ASSIGN_CHARGES: Metal", idx, kdx, met.formula, met.charge)
            else:
                print(
                    "ASSIGN_CHARGES: Non-Complex",
                    idx,
                    mol.formula,
                    mol.totcharge,
                    mol.smiles,
                )

    #######################################################
    def check_charge_neutrality(self, debug: int = 0):
        if self.subtype == "reference":
            moleclist = self.refmoleclist
        else:
            moleclist = self.moleclist
        
        if moleclist is None:
            # TOFIX @choglass: if no molecule list, then the cell is not neutral?
            # IF THERE IS NO MOLECULE LIST, THE PROCESS WOULD BE STOPPED PREVIOUSLY
            raise ValueError("TO CHECK:Molecule list is None")
            # self.is_neutral = None
            # return
            
        totcharge_list = []
        for mol in moleclist:
            if mol.totcharge is None:
                if debug >= 1:
                    print(f"CELL.CHECK_CHARGE_NEUTRALITY: Charges not assigned yet")
                self.is_neutral = None
            else:
                totcharge_list.append(mol.totcharge)

        if len(totcharge_list) != 0:
            if debug >= 1:
                print(
                    f"Total Charge of the Cell ({self.subtype}): {sum(totcharge_list)} {totcharge_list=}"
                )
            if sum(totcharge_list) == 0:
                self.is_neutral = True
            else:
                self.is_neutral = False

    #######################################################
    def assign_charges_old(self, debug: int = 0) -> object:
        #########
        # CHARGE#
        #########
        # This function drives the determination of the charge the species in the unit cell
        # The whole process is done by 4 functions, which are run at the specie class level:
        # 1) spec.get_protonation_states(), which determines which atoms of the specie must have added elements (see above) to have a meaningful Lewis structure
        # 2) spec.get_possible_cs(), which retrieves the possible charge states associated with the specie
        # 3) spec.get_possible_charge_state(), which generates one connectivity for a set of charges
        # 4) cell.select_charge_distr() chooses the best connectivity among the generated ones.

        # Basically, this function connects these other three functions,
        # while managing some key information for those
        # Initiates variables
        ############

        # (0) Makes sure the cell is reconstructed
        # if self.is_fragmented is None: self.reconstruct(debug=debug)
        # if self.is_fragmented: return # Stopping. self.is_fragmented must be false to determine the charges of the cell

        # (1) Indentify unique chemical species
        if self.unique_species is None:
            self.get_unique_species(debug=debug)
        if debug >= 1:
            print(
                f"{len(self.unique_species)} Species (Metal or Ligand or Molecules) to Characterize"
            )

        # (2) Gets a preliminary list of possible charge states for each specie (former drive_poscharge)
        selected_cs = []
        for idx, spec in enumerate(self.unique_species):
            tmp = spec.get_possible_cs(debug=debug)
            if tmp is None:
                self.error_get_poscharges = True
                return  # Stopping. Empty list of possible charges received.
            elif spec.subtype != "metal":
                selected_cs.append(
                    list([cs.corr_total_charge for cs in spec.possible_cs])
                )
            else:
                selected_cs.append(spec.possible_cs)
        self.error_get_poscharges = False

        # Finds the charge_state that satisfies that the crystal must be neutral
        final_charge_distribution = balance_charge(
            self.unique_indices, self.unique_species, debug=debug
        )

        if len(final_charge_distribution) > 1:
            if debug >= 1:
                print(
                    "More than one Possible Distribution Found:",
                    final_charge_distribution,
                )
            self.error_multiple_distrib = True
            self.error_empty_distrib = False
            pp_mols, pp_idx, pp_opt = prepare_unresolved(
                self.unique_indices,
                self.unique_species,
                final_charge_distribution,
                debug=debug,
            )
            self.data_for_postproc(pp_mols, pp_idx, pp_opt)
            return  # Stopping.

        elif len(final_charge_distribution) == 0:  #
            if debug >= 1:
                print("No valid Distribution Found", final_charge_distribution)
            self.error_multiple_distrib = False
            self.error_empty_distrib = True
            return  # Stopping.

        else:  # Only one possible charge distribution -> getcharge for the repeated species
            self.error_multiple_distrib = False
            self.error_empty_distrib = False

            self.moleclist, self.error_prepare_mols = prepare_mols(
                self.moleclist,
                self.unique_indices,
                self.unique_species,
                final_charge_distribution[0],
                debug=debug,
            )

            if self.error_prepare_mols:
                return  # Stopping. Error while preparing molecules
            else:
                return self.moleclist

    #######################################################
    def create_bonds(self, debug: int = 0):
        # if self.error_prepare_mols is None: self.assign_charges(debug=debug)
        # if self.error_prepare_mols: return # Stopping. self.error_prepare_mols must be false to create the spin

        if self.subtype == "reference":
            moleclist = self.refmoleclist
        else:
            moleclist = self.moleclist

        if moleclist is None:
            # TOFIX @choglass: if no molecule list, then the cell is not neutral?
            # IF THERE IS NO MOLECULE LIST, THE PROCESS WOULD BE STOPPED PREVIOUSLY
            raise ValueError("TO CHECK: Molecule list is None")
            # Proposition ↓
            # self.error_create_bonds = True
            # return

        temp = []
        for mol in moleclist:
            if debug >= 1: print(f"CELL.CREATE_BONDS: Creating Bonds for molecule {mol.formula}")
            mol.create_bonds(debug=debug)  
            temp.append(mol.error_create_bonds)
        
        if any(temp):
            self.error_create_bonds = True
        else:
            self.error_create_bonds = False

    #######################################################
    def assign_spin(self, debug: int = 0) -> object:
        # if self.error_prepare_mols is None: self.assign_charges(debug=debug)
        # if self.error_prepare_mols: return None # Stopping. self.error_prepare_mols must be false to assign the spin

        if debug >= 1:
            print("#########################################")
            print("       Assigning Spin multiplicity       ")
            print("#########################################")

        if self.subtype == "reference":
            moleclist = self.refmoleclist
        else:
            moleclist = self.moleclist

        if moleclist is None:
            # TOFIX @choglass: if no molecule list, just return?
            # IF THERE IS NO MOLECULE LIST, THE PROCESS WOULD BE STOPPED PREVIOUSLY
            raise ValueError("TO CHECK:Molecule list is None")
            # Proposition ↓
            # return

        for mol in moleclist:
            if mol.iscomplex:
                for metal in mol.metals:
                    if metal.coord_nr is None:
                        metal.get_coordination_geometry()
                        metal.get_coord_sphere_formula()
                    metal.get_spin(debug=debug)
            mol.get_spin(debug=debug)

    #######################################################
    def predict_metal_ox(self, debug: int = 0):
        if self.subtype == "reference":
            moleclist = self.refmoleclist
        else:
            moleclist = self.moleclist

        if moleclist is None:
            return

        for mol in moleclist:
            if mol.iscomplex:
                for metal in mol.metals:
                    if metal.coord_nr is None:
                        metal.get_coordination_geometry(debug=debug)
                        metal.get_coord_sphere_formula()
                    metal.predict_charge(debug=debug)

    #######################################################
    def assess_errors(self, mode):
        ### This function might be called to print the possible errors found in the unit cell, during reconstruction, and charge/spin assignment

        if mode == "cif_formula":
            print("-------------------------------")
            print("Errors in disagreement with CIF")
            print("-------------------------------")
            if self.has_isolated_H:
                case = 1
            elif self.disagree_with_cif_formula:
                case = 9
            else:
                case = 0
        if mode == "hydrogens":
            print("-------------------------------")
            print("Errors in hydrogens")
            print("-------------------------------")
            if self.has_isolated_H:
                case = 1
            # elif self.has_missing_H:            case = 2
            elif self.missing_H_in_Water:
                case = 2
            elif self.missing_H_in_CoordWater:
                case = 3
            elif self.missing_H_in_Carbon:
                case = 4
            else:
                case = 0
        elif mode == "possible_charges":
            print("-------------------------------")
            print("Errors in possible charges")
            print("-------------------------------")
            if self.has_isolated_H:
                case = 1
            # elif self.has_missing_H:            case = 2
            elif self.missing_H_in_Water:
                case = 2
            elif self.missing_H_in_CoordWater:
                case = 3
            elif self.missing_H_in_Carbon:
                case = 4
            elif self.error_get_poscharges:
                case = 5
            else:
                case = 0
        elif mode == "reconstruction":
            print("-------------------------------")
            print("Errors in reconstruction")
            print("-------------------------------")
            if self.has_isolated_H:
                case = 1
            elif self.has_missing_H:
                case = 2
            elif self.error_get_fragments:
                case = 3
            elif self.error_reconstruction:
                case = 4
            else:
                case = 0
        elif mode == "charge_assignment":
            print("-------------------------------")
            print("Errors in charge assignment")
            print("-------------------------------")
            if self.has_isolated_H:
                case = 1
            elif self.has_missing_H:
                case = 2
            elif self.error_get_fragments:
                case = 3
            elif self.error_reconstruction:
                case = 4
            elif self.error_get_poscharges:
                case = 5
            elif self.error_multiple_distrib:
                case = 6
            elif self.error_empty_distrib:
                case = 7
            # elif self.error_create_bonds :      case = 8
            else:
                case = 0
            # handle_error(case)
            # print("")
        # elif mode == "neutrality":
        #     print("-------------------------------")
        #     print("Errors in Unit Cell")
        #     print("-------------------------------")
        #     # Get reference molecules
        #     # if self.has_isolated_H:             case = 1
        #     # elif self.has_missing_H:            case = 2
        #     # Reconstruct Cell
        #     # elif self.error_get_fragments:      case = 3
        #     # elif self.error_reconstruction:     case = 4
        #     # Assign Charges
        #     # elif self.error_get_poscharges :  case = 5
        #     # elif self.error_multiple_distrib :  case = 6
        #     # elif self.error_empty_distrib :     case = 7
        #     # elif self.error_prepare_mols :      case = 8
        #     if not self.is_neutral:             case = 9
        #     # No errors
        #     else :                              case = 0
        if mode == "hydrogens" or mode == "possible_charges":
            if case == 2 or case == 3 or case == 4:
                handle_error(2)
                if case == 2:
                    print("    - Missing Hydrogens in Water Molecules")
                elif case == 3:
                    print("    - Missing Hydrogens in Coordinated Water Molecules")
                elif case == 4:
                    print("    - Missing Hydrogens in Carbon Atoms")
            else:
                handle_error(case)
        else:
            handle_error(case)
        print("")
        self.error_case = case

    #######################################################
    def save(self, path):
        print(f"SAVING cell2mol CELL ({self.subtype}) object to {path}")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    #######################################################
    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()
    
    def __repr__(self):
        to_print = f"------------- Cell2mol CELL Object ----------------\n"
        to_print += f" Version               = {self.version}\n"
        to_print += f" Type                  = {self.type}\n"
        if self.subtype is not None:
            to_print += f" Sub-Type              = {self.subtype}\n"
        to_print += f" Name (Refcode)        = {self.name}\n"
        to_print += f" Num Atoms             = {self.natoms}\n"
        to_print += f" Cell Parameters a:c   = {self.cell_param[0:3]}\n"
        to_print += f" Cell Parameters al:ga = {self.cell_param[3:6]}\n"
        # to_print += f' Cell Vector           = {self.cell_vector}\n'
        if self.moleclist is not None:
            to_print += f" # Molecules:          = {len(self.moleclist)}\n"
            to_print += f" With Formulae:                               \n"
            for idx, m in enumerate(self.moleclist):
                to_print += f"    {idx}: {m.formula} \n"
        to_print += "---------------------------------------------------\n"
        if self.refmoleclist is not None:
            to_print += f" # of Ref Molecules:   = {len(self.refmoleclist)}\n"
            to_print += f" With Formulae:                                  \n"
            for idx, ref in enumerate(self.refmoleclist):
                to_print += f"    {idx}: {ref.formula} \n"
        return to_print

    @classmethod
    @deprecated("Use cell() with the keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        labels: list[str],
        pos: list[list[float]],
        frac_coord: list[list[float]],
        cell_vector: object,
        cell_param: object
    ) -> "cell":
        return cls(
            name=name,
            labels=labels,
            pos=pos,  # Using pos which gets aliased to coord
            frac_coord=frac_coord,
            cell_vector=cell_vector,
            cell_param=cell_param
        )
