from __future__ import annotations
import pickle
import logging
from typing import Any
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.molecule import Molecule
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import split_species
from cell2mol.compare import (
    compare_species,
    compare_metals,
    compare_reference_indices,
)
from cell2mol.operations import extract_from_list
from cell2mol.new_charge_assignment import (
    set_charge_state,
    prepare_mol,
)
from cell2mol.elementdata import ElementData
from cell2mol.read_cif import get_moiety_indices_from_labels
from cell2mol.write_results import handle_error
from cell2mol.utils import BaseModel, config
from cell2mol.my_types import NDArray, Type, SubType

logger = logging.getLogger(__name__)

elemdatabase = ElementData()


Labels = list[str]


class Cell(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    # Required constructor parameters
    name: str
    labels: Labels
    coord: NDArray = Field(alias="pos")  # Using alias to match original parameter name
    frac_coord: NDArray
    cell_vector: NDArray
    cell_param: NDArray

    # Computed in constructor
    natoms: int | None = None

    # Set by methods throughout lifecycle
    subtype: SubType | None = None
    atom_site_labels: list[str] | None = None

    # CIF bond/moiety related attributes
    geom_bond_cif: list[tuple] | None = None
    moiety_list_cif: list[list[str]] | None = None
    has_cif_bond_moiety: bool | None = None
    moiety_indices: list[list[int]] | None = None

    # Unique species related attributes (Specie for molecules/ligands, Metal for metals)
    unique_species: list[Specie | Metal] | None = None
    unique_indices: list[int] | None = None
    species_list: list[Specie | Metal] | None = None

    # Missing H related attributes
    missing_H_in_Carbon: bool | None = None
    missing_H_in_CoordWater: bool | None = None
    missing_H_in_Water: bool | None = None
    has_missing_H: bool | None = None
    has_isolated_H: bool | None = None

    # Molecule lists
    refmoleclist: list[Molecule] | None = None
    moleclist: list[Molecule] | None = None

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
    selected_cs: list[list[int]] | None = None

    # Bond creation related attributes
    error_create_bonds: bool | None = None

    # Charge neutrality
    is_neutral: bool | None = None

    # Error assessment
    error_case: int | None = None

    # TOFIX @choglass: See if we keep here, it's assigned in refcell.py#146
    # KEEP IT FOR NOW
    chemical_name: str | None = None
    reported_metal_os: list[tuple[str, int]] | None = None
    moiety_dicts: list[dict] | None = None
    disagree_with_cif_formula: bool | None = None

    # Frozen fields
    version: str = Field(default="2.0", frozen=True)
    type: Type = Field(default="cell")

    def model_post_init(self, __context: Any) -> None:
        # Compute natoms from labels length
        if self.natoms is None:
            self.natoms = len(self.labels)

    def set_subtype(self, subtype: SubType):
        self.subtype = subtype

    def set_atom_site_labels(self, atom_site_labels):
        """Set the atom site labels (e.g. Fe1, O2, C3, etc. from CIF file)."""
        self.atom_site_labels = atom_site_labels

    def set_cif_bond_moiety(self, geom_bond_cif=None, moiety_list_cif=None):
        """Set and process bond moiety information from a CIF file."""
        self.geom_bond_cif = geom_bond_cif
        self.moiety_list_cif = moiety_list_cif

        self.has_cif_bond_moiety = False
        self.moiety_indices = None

        if geom_bond_cif is None:
            return

        self.has_cif_bond_moiety = True
        self.moiety_indices = get_moiety_indices_from_labels(
            self.atom_site_labels, moiety_list_cif
        )

    def set_additional_cif_info(self, chemical_name, reported_metal_os, moiety_dicts):
        """Set additional CIF information such as chemical name, reported metal oxidation states, and moiety dictionaries."""
        self.chemical_name = chemical_name
        self.reported_metal_os = reported_metal_os
        self.moiety_dicts = moiety_dicts

    def get_unique_species(self):
        """Get unique species in the cell."""
        logger.info("Getting unique species in %s", self.subtype)

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
            logger.debug("Molecule %d formula=%s", idx, mol.formula)
            if (
                not mol.iscomplex
                and not mol.has_IA_IIA
                and not mol.has_post_transition_metal
            ):  # Non-complex molecules
                found = False
                for ldx, typ in enumerate(typelist_mols):
                    issame = compare_species(mol, typ[0])
                    if issame:
                        found = True
                        kdx = typ[1]
                        logger.debug(
                            "Molecule %d is the same with %d in typelist", idx, ldx
                        )
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_mols.append(list([mol, kdx]))
                    self.unique_species.append(mol)
                    logger.debug(
                        "New molecule found with: formula=%s and added in position %d",
                        mol.formula,
                        kdx,
                    )
                self.unique_indices.append(kdx)
                mol.unique_index = kdx
                self.species_list.append(mol)
            else:  # Complex molecules
                if mol.ligands is None:
                    if mol.iscomplex:
                        mol.split_complex()
                    elif mol.has_IA_IIA:
                        mol.split_IA_IIA()
                    elif mol.has_post_transition_metal:
                        mol.split_post_transition_metal()
                # ligands
                for jdx, lig in enumerate(mol.ligands):
                    found = False
                    for ldx, typ in enumerate(typelist_ligs):
                        if lig.is_nitrosyl is None:
                            lig.evaluate_as_nitrosyl()
                        if typ[0].is_nitrosyl is None:
                            typ[0].evaluate_as_nitrosyl()
                        if lig.haptic_type is None:
                            lig.get_hapticity()
                        if typ[0].haptic_type is None:
                            typ[0].get_hapticity()

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
                                issame = compare_species(lig, typ[0])
                            else:
                                issame = False
                        if issame:
                            found = True
                            kdx = typ[1]
                            logger.debug(
                                "ligand %d is the same with %d in typelist", jdx, ldx
                            )
                    if not found:
                        specs_found += 1
                        kdx = specs_found
                        typelist_ligs.append(list([lig, kdx]))
                        self.unique_species.append(lig)
                        logger.debug(
                            "New ligand found with: formula=%s added in position %d",
                            lig.formula,
                            kdx,
                        )
                    self.unique_indices.append(kdx)
                    lig.unique_index = kdx
                    self.species_list.append(lig)
                # metals
                for jdx, met in enumerate(mol.metals):
                    found = False
                    for ldx, typ in enumerate(typelist_mets):
                        issame = compare_metals(met, typ[0])
                        if issame:
                            found = True
                            kdx = typ[1]
                            logger.debug(
                                "Metal %d is the same with %d in typelist", jdx, ldx
                            )
                    if not found:
                        specs_found += 1
                        kdx = specs_found
                        typelist_mets.append(list([met, kdx]))
                        self.unique_species.append(met)
                        logger.debug(
                            "New Metal Center found with: labels %s and added in position %d",
                            met.label,
                            kdx,
                        )
                    self.unique_indices.append(kdx)
                    met.unique_index = kdx
                    self.species_list.append(met)

        logger.info("Unique species: %s", [s.formula for s in self.unique_species])
        logger.info("Species list: %s", [s.formula for s in self.species_list])

        return self.unique_species

    def check_hydrogens(self):
        from cell2mol.hydrogen import check_missing_hydrogens

        (
            has_missing_h,
            missing_h_in_carbon,
            missing_h_in_coordinated_water,
            missing_h_in_water,
        ) = check_missing_hydrogens(self.refmoleclist)
        if has_missing_h:
            logger.info(
                "Missing hydrogens | carbon=%d, coordinated_water=%d, water=%d",
                missing_h_in_carbon,
                missing_h_in_coordinated_water,
                missing_h_in_water,
            )
        self.has_missing_H = has_missing_h
        self.missing_H_in_Carbon = missing_h_in_carbon
        self.missing_H_in_CoordWater = missing_h_in_coordinated_water
        self.missing_H_in_Water = missing_h_in_water

        return self.has_missing_H

    def get_reference_molecules(
        self,
        cov_factor: float | None = None,
        metal_factor: float | None = None,
        use_bond_info: bool | None = None,
    ):
        """
        Generate reference molecules from cell fractional coordinates and labels.
        Args:
            cov_factor (float): covalent radius scaling factor for adjacency
            metal_factor (float): additional scaling factor for metals
            use_bond_info (bool): whether to use CIF bond information
        Returns:
            list: list of reference molecule objects
        """
        if cov_factor is None:
            cov_factor = config.COV_FACTOR
        if metal_factor is None:
            metal_factor = config.METAL_FACTOR
        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        logger.info("#########################################")
        logger.info("    Generate reference molecules    ")
        if use_bond_info:
            logger.info("  Consistent with CIF moieties  ")
        logger.info("#########################################")

        if self.subtype != "reference":
            logger.error("Cell subtype is not 'reference'")
            return []

        ref_labels = self.labels
        ref_fracs = self.frac_coord
        ref_pos = self.coord
        atom_site_labels = self.atom_site_labels

        # Determine blocklist
        if use_bond_info:
            blocklist = self.moiety_indices
            logger.info("Using CIF bond/moiety information for blocklist")
            if self.moiety_indices is None:
                logger.error("CIF moiety indices are not available")
                return []
        else:
            blocklist = split_species(ref_labels, ref_pos, cov_factor=cov_factor)
            logger.info("Using distance-based species splitting for blocklist")
            if self.moiety_indices is not None:
                logger.info("CIF bond/moiety information is available but not used")

        logger.debug(f"blocklist={blocklist}")
        if blocklist is None:
            logger.warning("No blocklist found")
            return []

        self.refmoleclist = []
        # Build reference molecules
        for b in blocklist:
            mol_labels = extract_from_list(b, ref_labels, dimension=1)
            mol_coord = extract_from_list(b, ref_pos, dimension=1)
            mol_frac_coord = extract_from_list(b, ref_fracs, dimension=1)
            mol_atom_site_labels = extract_from_list(b, atom_site_labels, dimension=1)

            newmolec = Molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
            newmolec.add_parent(self, indices=b)
            newmolec.set_adjacency_parameters(cov_factor, metal_factor)
            newmolec.set_atoms(
                create_adjacencies=True,
                atom_site_labels=mol_atom_site_labels,
                use_bond_info=use_bond_info,
            )

            for atom, idx in zip(newmolec.atoms, b):
                atom.add_parent(self, index=idx)

            if newmolec.iscomplex or newmolec.has_IA_IIA:
                logger.debug("Is complex: %s", newmolec.formula)
                logger.debug("Splitting complex: %s", newmolec.formula)
                newmolec.split_complex()
            elif newmolec.has_post_transition_metal:
                logger.debug("Has post-transition metal: %s", newmolec.formula)
                newmolec.split_complex(post_tms=True)
            else:
                newmolec.add_parent(newmolec, indices=list(range(newmolec.natoms)))

            self.refmoleclist.append(newmolec)

        logger.info("Found %d reference molecules", len(self.refmoleclist))
        logger.info("Formulas: %s", [ref.formula for ref in self.refmoleclist])

        # Check for isolated atoms
        has_isolated_h = False
        for ref in self.refmoleclist:
            if ref.natoms == 1:
                label = ref.atoms[0].label
                if label in {"H", "D"}:
                    has_isolated_h = True
                else:
                    logger.warning("Isolated atom found %s", ref.labels)

        self.has_isolated_H = has_isolated_h
        logger.info("Has isolated hydrogen: %s", self.has_isolated_H)

        # Post-processing: coordination analysis
        for ref in self.refmoleclist:
            if ref.iscomplex:
                logger.info("Working with transition metals %s", ref.formula)
                ref.get_hapticity()
                if len(ref.ligands) == 0:
                    logger.debug("A metal cluster found")
                else:
                    for lig in ref.ligands:
                        lig.get_denticity()
                for met in ref.metals:
                    met.get_connected_metals()
                    met.get_coordination_geometry()
                    met.get_coord_sphere_formula()

            elif ref.has_IA_IIA:
                logger.info(
                    "Working with alkali or alkali earth metals: %s", ref.formula
                )
                for lig in ref.ligands:
                    lig.get_denticity()
                for met in ref.metals:
                    met.get_connected_metals()
                    met.get_coordination_geometry()
                    met.get_coord_sphere_formula()

            elif ref.has_post_transition_metal:
                logger.info("Working with post transition metals: %s", ref.formula)
                logger.debug("metals=%s", [met.label for met in ref.metals])
                logger.debug("ligands=%s", [lig.formula for lig in ref.ligands])
                for lig in ref.ligands:
                    lig.get_denticity()
                for met in ref.metals:
                    met.get_connected_metals()
                    met.get_coordination_geometry()
                    met.get_coord_sphere_formula()

        return self.refmoleclist

    def get_selected_cs(self, debug: int = 0):
        """Get selected charge states for unique species and species list."""
        if self.unique_species is None:
            self.get_unique_species()

        self.selected_cs = []
        for unique_specie in self.unique_species:
            logger.info(
                "Get possible charge states for unique specie %s",
                unique_specie.formula,
            )
            tmp = unique_specie.get_possible_cs()
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

    #
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
                        issame = compare_reference_indices(ref, mol)
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
                                    issame = compare_reference_indices(ref_lig, lig)
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

    def check_charge_neutrality(self):
        """Check if the total charge of the cell is neutral."""
        if self.subtype == "reference":
            moleclist = self.refmoleclist
        else:
            moleclist = self.moleclist

        if moleclist is None:
            raise ValueError("TO CHECK: Molecule list is None")

        totcharge_list = []
        for mol in moleclist:
            if mol.totcharge is None:
                logger.warning("Charges not assigned yet")
                self.is_neutral = None
            else:
                totcharge_list.append(mol.totcharge)

        if len(totcharge_list) != 0:
            logger.info(
                f"Total Charge of the Cell ({self.subtype}): {sum(totcharge_list)} {totcharge_list=}"
            )
            if sum(totcharge_list) == 0:
                self.is_neutral = True
            else:
                self.is_neutral = False

    def create_bonds(self):
        """Create bonds for all molecules in the cell."""
        if self.subtype == "reference":
            moleclist = self.refmoleclist
        else:
            moleclist = self.moleclist

        if moleclist is None:
            raise ValueError("TO CHECK: Molecule list is None")

        temp = []
        for mol in moleclist:
            logger.info(f"Creating Bonds for molecule {mol.formula}")
            mol.create_bonds()
            temp.append(mol.error_create_bonds)

        if any(temp):
            self.error_create_bonds = True
        else:
            self.error_create_bonds = False

    def assign_spin(self):
        """Assign spin multiplicity for all molecules in the cell."""
        logger.info("#########################################")
        logger.info("       Assigning Spin multiplicity       ")
        logger.info("#########################################")

        if self.subtype == "reference":
            moleclist = self.refmoleclist
        else:
            moleclist = self.moleclist

        if moleclist is None:
            raise ValueError("TO CHECK:Molecule list is None")

        for mol in moleclist:
            if mol.iscomplex:
                for metal in mol.metals:
                    if metal.coord_nr is None:
                        metal.get_coordination_geometry()
                        metal.get_coord_sphere_formula()
                    metal.get_spin()
            mol.get_spin()

    def predict_metal_ox(self):
        """Predict oxidation states for metals in all molecules in the cell."""
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
                        metal.get_coordination_geometry()
                        metal.get_coord_sphere_formula()
                    metal.predict_charge()

    def assess_errors(self, mode):
        ### This function might be called to print the possible errors found in the unit cell, during reconstruction, and charge/spin assignment

        if mode == "hydrogens":
            logger.info("Check Errors in hydrogens")
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
            logger.info("Check Errors in possible charges")
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
            logger.info("Check Errors in reconstruction")
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
            logger.info("Check Errors in charge assignment")
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

    def save(self, path):
        """Save the Cell object to a file using pickle."""
        logger.info(f"SAVING cell2mol CELL ({self.subtype}) object to {path}")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    def __str__(self):
        """Return a string representation of the Cell object."""
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
