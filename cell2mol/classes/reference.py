from __future__ import annotations
import logging
from cell2mol.classes.cell import Cell
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.molecule import Molecule
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import split_species
from cell2mol.compare import compare_species, compare_metals
from cell2mol.operations import extract_from_list, get_moiety_indices_from_labels
from cell2mol.elementdata import ElementData
from cell2mol.utils import config
from cell2mol.my_types import SubType
from cell2mol.charge.specie_assigner import set_charge_state

logger = logging.getLogger(__name__)

elemdatabase = ElementData()


Labels = list[str]


class Reference(Cell):
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    # Reference molecule lists
    refmoleclist: list[Molecule] | None = None

    atom_site_labels: list[str] | None = None

    # CIF bond/moiety related attributes
    geom_bond_cif: list[tuple] | None = None
    moiety_list_cif: list[list[str]] | None = None
    exist_cif_bond_moiety: bool | None = None
    moiety_indices: list[list[int]] | None = None

    # Unique species related attributes
    unique_species: list[Specie | Metal] | None = None

    # Missing H related attributes
    has_isolated_H: bool | None = None
    missing_H_in_Carbon: bool | None = None
    missing_H_in_CoordWater: bool | None = None
    missing_H_in_Water: bool | None = None
    has_missing_H: bool | None = None

    # Retrieving possible charges related attributes
    selected_cs: list[list[int]] | None = None
    error_get_poscharges: bool | None = None

    # Additional CIF related attributes
    chemical_name: str | None = None
    reported_metal_os: list[tuple[str, int]] | None = None
    moiety_dicts: list[dict] | None = None

    # Potential warning flags
    disagree_with_cif_formula: bool | None = None
    is_polynuclear_over_limit: bool | None = None
    has_mixed_metal_types: bool | None = None
    is_mismatch_adjacency: bool | None = None

    # Frozen fields
    subtype: SubType = Field(default="reference")

    def set_atom_site_labels(self, atom_site_labels):
        """Set the atom site labels (e.g. Fe1, O2, C3, etc. from CIF file)."""
        self.atom_site_labels = atom_site_labels

    def set_cif_bond_moiety(self, geom_bond_cif=None, moiety_list_cif=None):
        """Set and process bond moiety information from a CIF file."""
        self.geom_bond_cif = geom_bond_cif
        self.moiety_list_cif = moiety_list_cif

        self.exist_cif_bond_moiety = False
        self.moiety_indices = None

        if geom_bond_cif is None:
            return

        self.exist_cif_bond_moiety = True
        self.moiety_indices = get_moiety_indices_from_labels(
            self.atom_site_labels, moiety_list_cif
        )

    def set_additional_cif_info(self, chemical_name, reported_metal_os, moiety_dicts):
        """Set additional CIF information such as chemical name, reported metal oxidation states, and moiety dictionaries."""
        self.chemical_name = chemical_name
        self.reported_metal_os = reported_metal_os
        self.moiety_dicts = moiety_dicts

    def set_potential_warning(
        self, cif_mismatch, over_polynuclear_limit, mixed_metals, is_mismatch_adj
    ):
        """
        Set potential warning flags based on CIF mismatch, polynuclear limit,
        mixed metals, and adjacency mismatch.
        """
        self.disagree_with_cif_formula = cif_mismatch
        self.is_polynuclear_over_limit = over_polynuclear_limit
        self.has_mixed_metal_types = mixed_metals
        self.is_mismatch_adjacency = is_mismatch_adj

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

        logger.info("=" * 40)
        logger.info("    Generate reference molecules    ")
        if use_bond_info:
            logger.info("    Consistent with CIF moieties  ")
        logger.info("=" * 40)

        if self.subtype != "reference":
            logger.error("Cell subtype is not 'reference'")
            return []

        ref_labels = self.labels
        ref_fracs = self.frac_coord
        ref_pos = self.coord
        atom_site_labels = self.atom_site_labels
        bond_data = getattr(self, "geom_bond_cif", None)

        # Determine blocklist
        if use_bond_info:
            blocklist = self.moiety_indices
            logger.info("Using CIF bond/moiety information for blocklist")
            if self.moiety_indices is None:
                logger.error("CIF moiety indices are not available")
                return []
        else:
            blocklist = split_species(
                ref_labels,
                ref_pos,
                atom_site_labels=atom_site_labels,
                bond_data=bond_data,
                cov_factor=cov_factor,
                metal_factor=metal_factor,
                warn_on_mismatch=True,
            )
            logger.info("Using distance-based species splitting for blocklist")
            if self.moiety_indices is not None:
                logger.info("CIF bond/moiety information is available but not used")

        if not blocklist:
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

            if newmolec.iscomplex or newmolec.has_ia_iia:
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
                    logger.warning(
                        "  Isolated hydrogen found %s (%s)",
                        ref.labels[0],
                        ref.atom_site_labels[0] if ref.atom_site_labels else "N/A",
                    )
                else:
                    logger.warning(
                        "  Isolated atom found %s (%s)",
                        ref.labels[0],
                        ref.atom_site_labels[0] if ref.atom_site_labels else "N/A",
                    )

        self.has_isolated_H = has_isolated_h
        logger.info("Has isolated hydrogen: %s", self.has_isolated_H)

        # Post-processing: coordination analysis
        for ref in self.refmoleclist:
            if ref.iscomplex:
                logger.info("Has transition metals %s", ref.formula)
                ref.get_hapticity()
                if not ref.ligands:
                    logger.debug("A metal cluster found")

            elif ref.has_ia_iia:
                logger.info("Has alkali or alkaline earth metals: %s", ref.formula)

            elif ref.has_post_transition_metal:
                logger.info("Has post transition metals: %s", ref.formula)
                logger.debug("metals=%s", [met.label for met in ref.metals])
                logger.debug("ligands=%s", [lig.formula for lig in ref.ligands])

            else:
                continue

            # Common ligand analysis
            for lig in ref.ligands:
                lig.get_denticity()
            # Common metal analysis
            for met in ref.metals:
                met.get_connected_metals()
                met.get_coordination_geometry()
                met.get_coord_sphere_formula()

        return self.refmoleclist

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

    def get_unique_species(self):
        """Get unique species, unique indices, and species list in the Reference cell."""
        logger.info("Getting unique species in %s", self.subtype)

        self.unique_species = []
        self.unique_indices = []
        self.species_list = []
        typelist_mols = []  # temporary variable
        typelist_ligs = []  # temporary variable
        typelist_mets = []  # temporary variable

        specs_found = -1
        if not self.refmoleclist:  # None or empty
            logger.error("Reference molecule list is None")
            return
        moleclist = self.refmoleclist

        for idx, mol in enumerate(moleclist):
            logger.debug("Molecule (%d) formula=%s", idx, mol.formula)
            if mol.is_non_complex_molecule:  # Non-complex molecules
                found = False
                for ldx, typ in enumerate(typelist_mols):
                    issame = compare_species(mol, typ[0])
                    if issame:
                        found = True
                        kdx = typ[1]
                        logger.debug(
                            "molecule %s (%d) is the same with type %d in type list",
                            mol.formula,
                            idx,
                            ldx,
                        )
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_mols.append(list([mol, kdx]))
                    self.unique_species.append(mol)
                    logger.debug(
                        "New molecule found with: formula=%s and added in specie type %d",
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
                    elif mol.has_ia_iia:
                        mol.split_ia_iia()
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
                                "ligand %s (%d) is the same with type %d in type list",
                                lig.formula,
                                jdx,
                                ldx,
                            )
                    if not found:
                        specs_found += 1
                        kdx = specs_found
                        typelist_ligs.append(list([lig, kdx]))
                        self.unique_species.append(lig)
                        logger.debug(
                            "New ligand found with: formula=%s added in specie type %d",
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
                                "Metal %s (%d) is the same with type %d in type list",
                                met.formula,
                                jdx,
                                ldx,
                            )
                    if not found:
                        specs_found += 1
                        kdx = specs_found
                        typelist_mets.append(list([met, kdx]))
                        self.unique_species.append(met)
                        logger.debug(
                            "New metal found with: formula=%s and added in specie type %d",
                            met.formula,
                            kdx,
                        )
                    self.unique_indices.append(kdx)
                    met.unique_index = kdx
                    self.species_list.append(met)

        logger.info("Unique species: %s", [s.formula for s in self.unique_species])
        logger.info("Unique indices: %s", self.unique_indices)
        logger.info("Species list: %s", [s.formula for s in self.species_list])

    def get_selected_cs(self) -> None:
        """
        Get selected (valid) charge states for unique species and species list.
        Updates self.selected_cs and sets error flags if any None is found.
        """
        assert self.subtype == "reference", (
            "get_selected_cs should only be called on reference"
        )

        if self.unique_species is None:
            self.get_unique_species()

        self.selected_cs = []

        # Process unique_species first, then the full species_list
        all_targets = [(specie, "unique specie") for specie in self.unique_species]
        all_targets.extend([(specie, "species list") for specie in self.species_list])

        for specie, context_label in all_targets:
            logger.info(
                "Get possible charge states for %s: %s",
                context_label,
                specie.formula,
            )
            possible_cs = specie.get_possible_cs()

            if not possible_cs:
                # Appending None indicates a failure to find options for this species
                self.selected_cs.append(None)
                continue

            if specie.subtype != "metal":
                charges = [cs.corr_total_charge for cs in specie.possible_cs]
                self.selected_cs.append(charges)
            else:
                self.selected_cs.append(specie.possible_cs)

        # Update error flag
        self.error_get_poscharges = None in self.selected_cs

    def map_charges_to_reference(self):
        """Logic: Propagate charges from Unique Species to Reference Molecules."""
        self.error_assign_charge = False

        for specie in self.unique_species:
            for ref in self.refmoleclist:
                try:
                    if ref.is_non_complex_molecule:
                        # Direct match for simple molecules
                        if ref.unique_index == specie.unique_index:
                            set_charge_state(specie, ref, mode=1)
                    else:
                        # Attempt to match all Ligands
                        for lig in ref.ligands:
                            if lig.unique_index == specie.unique_index:
                                try:
                                    set_charge_state(specie, lig, mode=1)
                                except Exception:
                                    self.error_assign_charge = True

                        # Attempt to match all Metals
                        for met in ref.metals:
                            if met.unique_index == specie.unique_index:
                                try:
                                    met.set_charge(specie.charge)
                                except Exception:
                                    self.error_assign_charge = True

                except Exception as e:
                    # Catch-all for unexpected logic errors in the outer ref loop
                    self.error_assign_charge = True
                    logger.error(
                        "Error mapping charge for Specie %s: %s", specie.unique_index, e
                    )

    def __repr__(self):
        to_print = ""
        to_print += "------------- Cell2mol Reference Object --------------\n"
        to_print += Cell.__repr__(self, indirect=True)
        if self.refmoleclist is not None:
            to_print += f" # of Ref Molecules:   = {len(self.refmoleclist)}\n"
            to_print += " with Formula:                                  \n"
            for idx, ref in enumerate(self.refmoleclist):
                to_print += f"    {idx}: {ref.formula} \n"
            to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use reference() with the keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        labels: list[str],
        pos: list[list[float]],
        frac_coord: list[list[float]],
        cell_vector: object,
        cell_param: object,
    ) -> "Reference":
        return cls(
            name=name,
            labels=labels,
            pos=pos,  # Using pos which gets aliased to coord
            frac_coord=frac_coord,
            cell_vector=cell_vector,
            cell_param=cell_param,
        )
