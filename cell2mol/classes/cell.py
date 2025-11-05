from __future__ import annotations
import numpy as np
import pickle
from typing import Any
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.molecule import Molecule
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import (
    compare_species,
    compare_metals,
    compare_reference_indices,
    split_species,
)
from cell2mol.cell_operations import frac2cart_fromparam
from cell2mol.new_charge_assignment import (
    set_charge_state,
    prepare_mol,
)

from cell2mol.other import extract_from_list, handle_error
from cell2mol.elementdata import ElementData
from cell2mol.read_write import get_moiety_indices_from_labels
from cell2mol.utils import BaseModel
from cell2mol.my_types import (
    NDArray,
    Type,
    SubType,
)

elemdatabase = ElementData()


Labels = list[str]


##############
#### CELL ####
##############
class Cell(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Required constructor parameters
    name: str
    labels: Labels
    coord: list[float] | NDArray = Field(
        alias="pos"
    )  # Using alias to match original parameter name
    frac_coord: list[float] | NDArray
    cell_vector: NDArray
    cell_param: NDArray

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
    unique_species: list[Specie] | None = None
    unique_indices: list[int] | None = None
    species_list: list[Specie] | None = None

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
    def get_cif_bond_moiety(
        self, cif_bond_info: bool, geom_bond_cif=None, moiety_list_cif=None
    ):
        """
        Get the bond moiety from the cif file.
        """
        self.geom_bond_cif = geom_bond_cif
        self.moiety_list_cif = moiety_list_cif

        if cif_bond_info:
            self.exist_cif_bond_moiety = True
            moiety_indices = get_moiety_indices_from_labels(
                self.atom_site_labels, moiety_list_cif
            )
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
            if (
                not mol.iscomplex
                and not mol.has_IA_IIA
                and not mol.has_post_transition_metal
            ):
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
                        if lig.haptic_type is None:
                            lig.get_hapticity(debug=debug)
                        if typ[0].haptic_type is None:
                            typ[0].get_hapticity(debug=debug)
                        # if lig.metals is None:
                        #     lig.get_connected_metals(debug=debug)
                        # if typ[0].metals is None:
                        #     typ[0].get_connected_metals(debug=debug)
                        lig_groups_labels = [g.labels for g in lig.groups]
                        typ_groups_labels = [g.labels for g in typ[0].groups]

                        if lig.is_nitrosyl and typ[0].is_nitrosyl:
                            if lig.NO_type == typ[0].NO_type:
                                issame = True
                            else:
                                issame = False
                        else:
                            if (
                                len(lig_groups_labels) == len(typ_groups_labels)
                                and sorted(lig_groups_labels)
                                == sorted(typ_groups_labels)
                                and lig.haptic_type == typ[0].haptic_type
                            ):
                                # if lig.haptic_type == typ[0].haptic_type:
                                issame = compare_species(lig, typ[0], debug=0)
                            else:
                                issame = False
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
        ref_labels: Labels,
        ref_fracs: list,
        cov_factor: float = 1.3,
        metal_factor: float = 1.0,
        debug: int = 0,
    ):
        # Convert fractional coordinates to cartesian
        ref_pos = frac2cart_fromparam(ref_fracs, self.cell_param)

        # Define reference cell
        refcell = Cell.from_positional(
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

        if debug >= 2:
            print(f"GETREFS: blocklist={blocklist}")
        if debug >= 2:
            print(
                "GETREFS: Generate Adjacency matrix based on bond information from CIF"
            )

        self.refmoleclist = []
        if blocklist is None:
            print("GETREFS: No moiety indices found in the CIF file")
            return self.refmoleclist

        for b in blocklist:
            mol_labels = extract_from_list(b, ref_labels, dimension=1)
            mol_coord = extract_from_list(b, ref_pos, dimension=1)
            mol_frac_coord = extract_from_list(b, ref_fracs, dimension=1)
            mol_atom_site_labels = extract_from_list(b, atom_site_labels, dimension=1)

            newmolec = Molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
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
            # if newmolec.iscomplex :
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
            print("GETREFS:", [ref.formula for ref in self.refmoleclist])

        # Checks for isolated atoms, and retrieves warning if there is any.
        # Except if it is H, halogen (group 17) or alkalyne (group 2)
        isgood = True
        for ref in self.refmoleclist:
            if ref.natoms == 1:
                label = ref.atoms[0].label
                # group = elemdatabase.elementgroup[label]
                if label == "H" or label == "D":
                    isgood = False
                else:  # (group == 1 or group == 2 or group == 17)
                    if debug >= 0:
                        print(
                            f"GETREFS: found ref molecule with only one atom {ref.labels}"
                        )
        if debug >= 0:
            print(f"GETREFS: isgood={isgood}")
        # If all good, then works with the reference molecules
        if isgood:
            self.has_isolated_H = False
            for ref in self.refmoleclist:
                if ref.iscomplex:
                    if debug >= 0:
                        print(
                            f"GETREFS: working with {ref.formula} with transition metals"
                        )
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
                    if debug >= 0:
                        print(
                            f"GETREFS: working with {ref.formula} with alkali or alkali earth metals"
                        )
                    if len(ref.ligands) == 0:
                        pass
                    else:
                        for lig in ref.ligands:
                            lig.get_denticity(debug=debug)
                    for met in ref.metals:
                        met.get_connected_metals(debug=debug)
                        met.get_coordination_geometry(debug=debug)
                        met.get_coord_sphere_formula(debug=debug)
                elif ref.has_post_transition_metal:
                    if debug >= 0:
                        print(
                            f"GETREFS: working with {ref.formula} with post-transition metals"
                        )
                    if debug >= 0:
                        print(f"GETREFS: {[met.label for met in ref.metals]}")
                    if debug >= 0:
                        print(f"GETREFS: {[lig.formula for lig in ref.ligands]}")
                    if len(ref.ligands) == 0:
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
        refcell = Cell.from_positional(
            self.name, ref_labels, ref_pos, ref_fracs, self.cell_vector, self.cell_param
        )
        refcell.set_subtype("reference")

        atom_site_labels = self.atom_site_labels
        geom_bond_cif = self.geom_bond_cif

        if debug > 2:
            print(f"GETREFS: geom_bond_cif={geom_bond_cif}")
        if debug >= 2:
            print("GETREFS: Generate Adjacency matrix based on interatomic distances")
        blocklist = split_species(ref_labels, ref_pos, cov_factor=cov_factor)
        if debug >= 2:
            print(f"GETREFS: blocklist={blocklist}")

        self.refmoleclist = []
        if blocklist is None:
            print("GETREFS: No blocklist found")
            return self.refmoleclist

        # Get reference molecules
        for b in blocklist:
            mol_labels = extract_from_list(b, ref_labels, dimension=1)
            mol_coord = extract_from_list(b, ref_pos, dimension=1)
            mol_frac_coord = extract_from_list(b, ref_fracs, dimension=1)
            mol_atom_site_labels = extract_from_list(b, atom_site_labels, dimension=1)

            newmolec = Molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
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
            print("GETREFS:", [ref.formula for ref in self.refmoleclist])

        # Checks for isolated atoms, and retrieves warning if there is any.
        # Except if it is H, halogen (group 17) or alkalyne (group 2)
        isgood = True
        for ref in self.refmoleclist:
            if ref.natoms == 1:
                label = ref.atoms[0].label
                # group = elemdatabase.elementgroup[label]
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
        else:
            self.has_isolated_H = True
        if debug >= 0:
            print(f"GETREFS: has_isolated_H={self.has_isolated_H}")

        for ref in self.refmoleclist:
            if ref.iscomplex:
                if debug >= 0:
                    print(f"GETREFS: working with {ref.formula} with transition metals")
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
                if debug >= 0:
                    print(
                        f"GETREFS: working with {ref.formula} with alkali or alkali earth metals"
                    )
                if len(ref.ligands) == 0:
                    pass
                else:
                    for lig in ref.ligands:
                        lig.get_denticity(debug=debug)
                for met in ref.metals:
                    met.get_connected_metals(debug=debug)
                    met.get_coordination_geometry(debug=debug)
                    met.get_coord_sphere_formula(debug=debug)
            elif ref.has_post_transition_metal:
                if debug >= 0:
                    print(
                        f"GETREFS: working with {ref.formula} with post-transition metals"
                    )
                if debug >= 0:
                    print(f"GETREFS: {[met.label for met in ref.metals]}")
                if debug >= 0:
                    print(f"GETREFS: {[lig.formula for lig in ref.ligands]}")
                if len(ref.ligands) == 0:
                    pass
                else:
                    for lig in ref.ligands:
                        lig.get_denticity(debug=debug)
                for met in ref.metals:
                    met.get_connected_metals(debug=debug)
                    met.get_coordination_geometry(debug=debug)
                    met.get_coord_sphere_formula(debug=debug)

        return self.refmoleclist

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
            if tmp is None:
                self.selected_cs.append(None)
            elif len(tmp) == 0:
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
            if tmp is None:
                self.selected_cs.append(None)
            elif len(tmp) == 0:
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
    def assign_charges_for_refcell(self, debug: int = 0):
        for specie in self.unique_species:
            for idx, ref in enumerate(self.refmoleclist):
                if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
                    for jdx, lig in enumerate(ref.ligands):
                        if lig.unique_index == specie.unique_index:
                            set_charge_state(specie, lig, mode=1, debug=debug)
                    for kdx, met in enumerate(ref.metals):
                        if met.unique_index == specie.unique_index:
                            met.set_charge(specie.charge)
                else:
                    if ref.unique_index == specie.unique_index:
                        set_charge_state(specie, ref, mode=1, debug=debug)
        temp = []
        for idx, ref in enumerate(self.refmoleclist):
            ref.create_bonds(debug=debug)
            temp.append(ref.error_create_bonds)
            if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
                prepare_mol(ref, debug=debug)

        if any(temp):
            self.error_create_bonds = True
        else:
            self.error_create_bonds = False

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
            if (
                not mol.iscomplex
                and not mol.has_IA_IIA
                and not mol.has_post_transition_metal
            ):
                for ref in self.refmoleclist:
                    if (
                        not ref.iscomplex
                        and not ref.has_IA_IIA
                        and not ref.has_post_transition_metal
                    ) and (mol.unique_index == ref.unique_index):
                        issame = compare_reference_indices(ref, mol, debug=debug)
                        if issame:
                            set_charge_state(ref, mol, mode=2, debug=debug)
            else:
                for ref in self.refmoleclist:
                    if (
                        ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal
                    ) and (mol.formula == ref.formula):
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

        temp = []
        for idx, mol in enumerate(self.moleclist):
            mol.create_bonds(debug=debug)
            temp.append(mol.error_create_bonds)
            if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
                prepare_mol(mol, debug=debug)

        if any(temp):
            self.error_create_bonds = True
        else:
            self.error_create_bonds = False

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
                    print("CELL.CHECK_CHARGE_NEUTRALITY: Charges not assigned yet")
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
            if debug >= 1:
                print(f"\nCELL.CREATE_BONDS: Creating Bonds for molecule {mol.formula}")
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
        to_print = "------------- Cell2mol CELL Object ----------------\n"
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
            to_print += " with Formula:                               \n"
            for idx, m in enumerate(self.moleclist):
                to_print += f"    {idx}: {m.formula} \n"
        to_print += "---------------------------------------------------\n"
        if self.refmoleclist is not None:
            to_print += f" # of Ref Molecules:   = {len(self.refmoleclist)}\n"
            to_print += " with Formula:                                  \n"
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
        cell_param: object,
    ) -> "Cell":
        return cls(
            name=name,
            labels=labels,
            pos=pos,  # Using pos which gets aliased to coord
            frac_coord=frac_coord,
            cell_vector=cell_vector,
            cell_param=cell_param,
        )
