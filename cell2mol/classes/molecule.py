from __future__ import annotations
from typing_extensions import deprecated

from pydantic import Field
from cell2mol.classes.atom import Atom
from cell2mol.classes.ligand import Ligand
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import (
    labels2formula,
)
from cell2mol.connectivity import (
    get_metal_idxs,
    get_non_transition_metal_idxs,
    split_species,
    get_alkali_alkaline_earth_metal_idxs,
)
from cell2mol.connectivity import (
    compare_species,
    compare_metals,
)

from cell2mol.charge_assignment import (
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

from cell2mol.spin import assign_spin_complexes
from cell2mol.other import extract_from_list
from cell2mol.elementdata import ElementData
from cell2mol.my_types import (
    Spin,
    HapticType,
    SubType,
)

elemdatabase = ElementData()
import pickle


###############
### MOLECULE ##
###############
class Molecule(Specie):
    """
    A molecule is a specie that contains other specie objects.
    """

    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    ligands: list | None = None
    metals: list[Atom] | None = None
    spin: Spin | None = None
    ref_indices: list[int] | None = None
    cell_indices: list[int] | None = None

    unique_index: int | None = None
    totcharge_cif: int | None = None

    ligand_smiles: str | list[str] | None = None
    ligand_smiles_with_H: list[str] | None = None
    error_create_bonds: bool = False

    subtype: SubType = Field(default="molecule")

    # Needed in get_molecule in xyz_molecule.py
    input_charge: int | None = None
    unique_species: list[object] | None = None
    unique_indices: list[int] | None = None
    species_list: list[object] | None = None
    selected_cs: list[object] | None = None
    error_get_poscharges: bool = False
    error_multiple_distrib: bool = False
    error_empty_distrib: bool = False
    error_create_bonds: bool = False

    @classmethod
    @deprecated("Use molecule() with the keyword arguments instead.")
    def from_positional(
        cls, labels: list, coord: list, frac_coord: list = None, radii: list = None
    ) -> "Molecule":
        return cls(labels=labels, coord=coord, frac_coord=frac_coord, radii=radii)

    def __repr__(self):
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

    ############
    def get_spin(self, debug: int = 0):
        if self.iscomplex:
            self.spin = assign_spin_complexes(self)
        else:
            if (self.eleccount - self.totcharge) % 2 == 0:
                self.spin = 1
            else:
                self.spin = 2
        if debug >= 1:
            print(
                f"GET_SPIN: Spin multiplicity of the complex {self.formula} is assigned as {self.spin}\n"
            )
        return self.spin

    ############
    def reset_charge(self):
        Specie.reset_charge(
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
                print(
                    f"MOLECULE.SPLIT_POST_TRANSITION_METAL: metal_idx={post_transition_metal_indices}"
                )
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
                print(
                    f"MOLECULE.SPLIT_POST_TRANSITION_METAL: rest labels: {rest_labels}"
                )
                print(
                    f"MOLECULE.SPLIT_POST_TRANSITION_METAL: rest indices: {rest_indices}"
                )
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
                if (
                    refcell is not None
                    and refcell.exist_cif_bond_moiety
                    and refcell.geom_bond_cif is not None
                ):
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
                print(
                    f"MOLECULE.SPLIT_POST_TRANSITION_METAL: received {len(blocklist)} blocks {blocklist=}"
                )

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
                        newligand = Ligand.from_positional(
                            lig_labels, lig_coord, lig_frac_coord, radii=lig_radii
                        )
                    else:
                        newligand = Ligand.from_positional(
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
                        ref_indices = [
                            a.get_parent_index("reference") for a in lig_atoms
                        ]
                        newligand.add_parent(
                            self.get_parent("reference"), indices=ref_indices
                        )

                    newligand.set_adjacency_parameters(
                        self.cov_factor, self.metal_factor
                    )

                    newligand.set_atoms(
                        atomlist=lig_atoms,
                        atom_site_labels=lig_atom_site_labels,
                        geom_bond_cif=geom_bond_cif,
                    )

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
                if (
                    refcell is not None
                    and refcell.exist_cif_bond_moiety
                    and refcell.geom_bond_cif is not None
                ):
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
                        newligand = Ligand.from_positional(
                            lig_labels, lig_coord, lig_frac_coord, radii=lig_radii
                        )
                    else:
                        newligand = Ligand.from_positional(
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
                        ref_indices = [
                            a.get_parent_index("reference") for a in lig_atoms
                        ]
                        newligand.add_parent(
                            self.get_parent("reference"), indices=ref_indices
                        )

                    newligand.set_adjacency_parameters(
                        self.cov_factor, self.metal_factor
                    )

                    newligand.set_atoms(
                        atomlist=lig_atoms,
                        atom_site_labels=lig_atom_site_labels,
                        geom_bond_cif=geom_bond_cif,
                    )

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
                if (
                    refcell is not None
                    and refcell.exist_cif_bond_moiety
                    and refcell.geom_bond_cif is not None
                ):
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
                    else:
                        lig_atom_site_labels = None
                    if debug > 0:
                        print(f"CREATING LIGAND: {labels2formula(lig_labels)}")
                    # Create Ligand Object
                    if self.frac_coord is not None:
                        newligand = Ligand.from_positional(
                            lig_labels, lig_coord, lig_frac_coord, radii=lig_radii
                        )
                    else:
                        newligand = Ligand.from_positional(
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

                    newligand.set_atoms(
                        atomlist=lig_atoms,
                        atom_site_labels=lig_atom_site_labels,
                        geom_bond_cif=geom_bond_cif,
                    )

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
                    self.haptic_type.append(entry)
                    # if entry not in self.haptic_type:
                    #     self.haptic_type.append(entry)
        return self.haptic_type

    #######################################################
    def save(self, path):
        print(f"SAVING cell2mol CELL object to {path}")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

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
        if (
            not self.iscomplex
            and not self.has_IA_IIA
            and not self.has_post_transition_metal
        ):
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
                    print(
                        f"New molecule found: formula={self.formula}, added at position {kdx}"
                    )
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
                        issame = lig.NO_type == typ[0].NO_type
                    else:
                        if (
                            len(lig_groups_labels) == len(typ_groups_labels)
                            and sorted(lig_groups_labels) == sorted(typ_groups_labels)
                            and lig.haptic_type == typ[0].haptic_type
                        ):
                            # if lig.haptic_type == typ[0].haptic_type:
                            issame = compare_species(lig, typ[0], debug=0)
                        else:
                            issame = False

                    if issame:
                        found = True
                        kdx = typ[1]
                        if debug >= 0:
                            print(f"Ligand {jdx} is the same as {ldx} in typelist")

                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_ligs.append([lig, kdx])
                    self.unique_species.append(lig)
                    if debug >= 2:
                        print(
                            f"New ligand found: {lig.formula}, added at position {kdx}"
                        )

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
                        print(
                            f"New metal center found: label={met.label}, added at position {kdx}"
                        )

                self.unique_indices.append(kdx)
                met.unique_index = kdx
                self.species_list.append(met)

        return self.unique_species

    #######################################################
    def get_selected_cs(self, debug: int = 0):
        if not self.unique_species is not None:
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
    def balance_charges_for_molecules(
        self, input_charge: int = None, second_try: bool = False, debug: int = 0
    ):
        if not self.unique_species is not None:
            self.get_unique_species()
        if not self.selected_cs is not None:
            self.get_selected_cs()

        if None in self.selected_cs:
            self.error_get_poscharges = True
        else:
            self.error_get_poscharges = False

        unique_indices = [specie.unique_index for specie in self.species_list]

        final_charge_distribution, final_charges = balance_charge(
            unique_indices,
            self.unique_species,
            input_charge=input_charge,
            debug=debug,
        )
        print(f"{len(final_charge_distribution)=} {final_charge_distribution=}")
        # Handle multiple or no charge distributions
        dist_count = len(final_charge_distribution)
        self.error_multiple_distrib = dist_count > 1
        self.error_empty_distrib = dist_count == 0

        if dist_count != 1 and second_try:
            # Attempt to balance charges again with more specific conditions
            if self.error_multiple_distrib:
                print("More than one possible distribution found.")
                second_final_charge_distribution, second_final_charges = balance_charge(
                    unique_indices,
                    self.unique_species,
                    input_charge=input_charge,
                    aromatic=True,
                    debug=debug,
                )
            if self.error_empty_distrib:
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
        if any(
            [
                self.error_get_poscharges,
                self.error_multiple_distrib,
                self.error_empty_distrib,
            ]
        ):
            print("Charge Assignment Failed.")
            return

        # Assign charges to unique species in the molecules
        for specie, charge in zip(self.unique_species, final_charges[0]):
            assign_charge_to_specie(specie, charge, debug=debug)
            for refspecie in self.species_list:
                if specie.unique_index == refspecie.unique_index:
                    assign_charge_to_specie(refspecie, charge, debug=debug)

    #######################################################
    def assign_charges_for_molecule(self, debug: int = 0):
        print(f"ASSIGN_CHARGES_FOR_MOLECULE {self.formula}")

        for specie in self.unique_species:
            if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal:
                for jdx, lig in enumerate(self.ligands):
                    print(lig.unique_index)
                    print(specie.unique_index)
                    if lig.unique_index == specie.unique_index:
                        set_charge_state(specie, lig, mode=1, debug=debug)
                for kdx, met in enumerate(self.metals):
                    if met.unique_index == specie.unique_index:
                        met.set_charge(specie.charge)
            else:
                if self.unique_index == specie.unique_index:
                    set_charge_state(specie, self, mode=1, debug=debug)
        if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal:
            prepare_mol(self, debug=debug)
            print("Complex", self.formula, self.totcharge)
            for jdx, lig in enumerate(self.ligands):
                print("    Ligand", jdx, lig.formula, lig.totcharge, lig.smiles)
            for kdx, met in enumerate(self.metals):
                print("    Metal", kdx, met.formula, met.charge)
        else:
            print("Non-Complex", self.formula, self.totcharge, self.smiles)

    #######################################################
    def create_bonds(self, debug: int = 0):
        # First part: Non-complex molecule
        if (
            not self.iscomplex
            and not self.has_IA_IIA
            and not self.has_post_transition_metal
        ):
            # Creates bonds between molecule.atoms using the molecule.rdkit_object
            result = create_bonds_specie(self, debug=debug)
            if result == False:
                if debug >= 1:
                    print(
                        f"MOLECULE.CREATE_BONDS: error creating bonds for non-complex molecule {self.formula}"
                    )
                self.error_create_bonds = True
                return  # Exit the function entirely if creating bonds fails
            else:
                if debug > 2:
                    print(
                        f"MOLECULE.CREATE_BONDS: Bonds created for non-complex molecule {self.formula}"
                    )

        # Second part: Complex molecule, add bonds for ligands
        if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal:
            self.ligand_smiles_with_H = [lig.smiles for lig in self.ligands]
            self.ligand_smiles = []
            fix_zwitterions_ligands = []

            for lig in self.ligands:
                # Creates bonds between ligand.atoms, using the ligand.rdkit_object
                result = create_bonds_specie(lig, debug=debug)
                if result == False:
                    if debug >= 1:
                        print(
                            f"MOLECULE.CREATE_BONDS: error creating bonds for ligand {lig.formula}"
                        )
                    self.error_create_bonds = True
                    return  # Exit the function entirely if creating bonds fails for any ligand

                if debug > 1:
                    print(
                        f"MOLECULE.CREATE_BONDS: Bonds created for ligand {lig.formula}"
                    )
                if debug > 1:
                    print(
                        f"MOLECULE.CREATE_BONDS: Correcting Smiles for ligand {lig.formula}"
                    )
                result, fix_zwitterions = correct_smiles_ligand(lig, debug=debug)
                if result == False:
                    if debug > 1:
                        print(
                            f"MOLECULE.CREATE_BONDS: error correcting smiles for ligand {lig.formula}"
                        )
                    self.error_create_bonds = True
                    return  # Exit the function entirely

                if debug > 1:
                    print(
                        f"MOLECULE.CREATE_BONDS: Smiles corrected for ligand {lig.formula}"
                    )
                if fix_zwitterions:
                    fix_zwitterions_ligands.append(lig)
                else:
                    self.ligand_smiles.append(lig.smiles)

            for lig in fix_zwitterions_ligands:
                for atom in lig.atoms:
                    atom.bonds = []
                if debug >= 1:
                    print(
                        f"MOLECULE.CREATE_BONDS: Re-running create_bonds_specie for ligand {lig.formula} due to zwitterion correction."
                    )
                result = create_bonds_specie(lig, debug=debug)
                if result == False:
                    if debug >= 1:
                        print(
                            f"MOLECULE.CREATE_BONDS: error re-creating bonds for ligand {lig.formula}"
                        )
                    self.error_create_bonds = True
                    return
                if debug >= 1:
                    print(
                        f"MOLECULE.CREATE_BONDS: Bonds re-created for ligand {lig.formula} after zwitterion correction."
                    )
                self.ligand_smiles.append(lig.smiles)

        # Third part : adds metal-ligand bonds, metal-metal bonds, with a zero order
        if self.iscomplex or self.has_IA_IIA or self.has_post_transition_metal:
            create_metal_ligand_bonds(self, debug=debug)
            create_metal_metal_bonds(self, debug=debug)

        self.error_create_bonds = False
