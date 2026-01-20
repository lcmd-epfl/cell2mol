from __future__ import annotations
import pickle
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.ligand import Ligand
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import split_species
from cell2mol.element_utils import (
    labels2formula,
    get_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
    POST_TRANSITION_METALS,
    METALLOIDS,
)
from cell2mol.compare import compare_species, compare_metals
from cell2mol.charge.specie_assigner import set_charge_state, prepare_mol
from cell2mol.charge.smiles_handler import (
    create_bonds_specie,
    create_metal_ligand_bonds,
    create_metal_metal_bonds,
    correct_smiles_ligand,
)
from cell2mol.spin import assign_spin_complexes
from cell2mol.operations import extract_from_list
from cell2mol.elementdata import ElementData
from cell2mol.my_types import Spin, HapticType, SubType
from cell2mol.utils import config
import logging

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


class Molecule(Specie):
    """
    A molecule is a specie that contains other specie objects.
    """

    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    ligands: list[Ligand] | None = Field(default=None)
    metals: list[Metal] | None = Field(default=None)
    spin: Spin | None = None
    ref_indices: list[int] | None = None
    cell_indices: list[int] | None = None

    unique_index: int | None = None
    totcharge_cif: int | None = None

    ligand_smiles: str | list[str] | None = None
    ligand_smiles_with_H: list[str] | None = None
    subtype: SubType = Field(default="molecule")

    # Needed in get_molecule in xyz_molecule.py
    input_charge: int | None = None
    unique_species: list[Specie] | None = None
    unique_indices: list[int] | None = None
    species_list: list[Specie] | None = None
    selected_cs: list[object] | None = None

    # Error assessment
    error_get_poscharges: bool | None = None
    error_multiple_distrib: bool | None = None
    error_empty_distrib: bool | None = None
    error_assign_charge: bool | None = None
    error_create_bonds: bool | None = None
    error_get_spin: bool | None = None
    error_case: int | None = None

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

    def get_spin(self):
        """
        Assign and return spin multiplicity for this molecule.
        Sets mol.error_get_spin on failure and raises an exception.
        """
        self.error_get_spin = False

        try:
            if self.iscomplex:
                self.spin = assign_spin_complexes(self)
            else:
                if (self.eleccount - self.totcharge) % 2 == 0:
                    self.spin = 1
                else:
                    self.spin = 2

        except Exception as e:
            self.error_get_spin = True
            raise RuntimeError(
                f"Spin assignment failed for molecule {self.formula}"
            ) from e

        logger.debug(
            "Assigned spin multiplicity | molecule=%s | spin=%d",
            self.formula,
            self.spin,
        )

        return self.spin

    def reset_charge(self):
        # First, uses the generic specie class function for itself and its atoms
        Specie.reset_charge(self)
        ## Then reset charge for the child classes
        if self.ligands is not None:
            for lig in self.ligands:
                lig.reset_charge()
        if self.metals is not None:
            for met in self.metals:
                met.reset_charge()

    def split_complex(self, use_bond_info: bool | None = None, post_tms: bool = False):
        """
        Split a complex into metals and ligands.

        Args:
        ----------
        use_bond_info : bool or None, optional
            Whether to use explicit bond information.
            If None, defaults to config.USE_BOND_INFO.
        post_tms : bool, optional
            If True, ONLY post-transition metals are treated as metals.
        """

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        # Ensure atoms exist
        if self.atoms is None:
            self.set_atoms()

        # Non-complex shortcut
        # If the molecule is not a complex and does not contain IA/IIA metals,
        # there is nothing to split.
        if self.is_non_complex_molecule:
            logger.debug(
                "Molecule %s is not a complex. No splitting needed.", self.formula
            )
            self.ligands = None
            self.metals = None
            return self.ligands, self.metals

        # Reference / bond information
        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None

        cov_factor = getattr(self, "cov_factor", config.COV_FACTOR)
        metal_factor = getattr(self, "metal_factor", config.METAL_FACTOR)

        self.ligands = []
        self.metals = []

        # ============================================================
        # Identify metal indices (GLOBAL index space: self.indices)
        # ============================================================
        if post_tms:
            logger.info(
                "post_tms enabled: ONLY post-transition metals exist as metals."
            )
            metal_idx: set[int] = {
                self.indices[i] for i in get_post_transition_metal_idxs(self.labels)
            }
        else:
            metal_idx: set[int] = {self.indices[i] for i in get_metal_idxs(self.labels)}
            metal_idx.update(
                self.indices[i]
                for i in get_alkali_alkaline_earth_metal_idxs(self.labels)
            )

        post_tm_found = [
            label for label in self.labels if label in POST_TRANSITION_METALS
        ]
        metalloid_found = [label for label in self.labels if label in METALLOIDS]
        if post_tm_found:
            logger.info(
                "Post-transition metals found: %s %s ", self.formula, post_tm_found
            )
        if metalloid_found:
            logger.info("Metalloids found: %s %s", self.formula, metalloid_found)

        # ============================================================
        # Remaining atoms → ligands
        # ============================================================

        rest_idx = [i for i in self.indices if i not in metal_idx]

        rest_labels = extract_from_list(rest_idx, self.labels, dimension=1)
        rest_coord = extract_from_list(rest_idx, self.coord, dimension=1)
        rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)
        rest_radii = extract_from_list(rest_idx, self.radii, dimension=1)
        rest_atoms = extract_from_list(rest_idx, self.atoms, dimension=1)
        # logger.debug("Remaining atom indices: %s", rest_idx)

        rest_frac = (
            extract_from_list(rest_idx, self.frac_coord, dimension=1)
            if self.frac_coord is not None
            else None
        )

        rest_atom_site_labels = (
            extract_from_list(rest_idx, self.atom_site_labels, dimension=1)
            if self.atom_site_labels is not None
            else None
        )

        # No ligands case
        if not rest_labels:
            logger.debug(
                "No ligands found in complex %s. Assigning metals only.",
                self.formula,
            )
            self.metals.extend(self.atoms[i] for i in metal_idx)
            return self.ligands, self.metals

        # ============================================================
        # Split ligands
        # ============================================================

        blocklist = split_species(
            labels=rest_labels,
            positions=rest_coord,
            radii=rest_radii,
            indices=None,  # rest_indices
            atom_site_labels=rest_atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
        )

        logger.info("Received %d ligand blocks", len(blocklist))
        # logger.debug("Blocks: %s", blocklist)

        for block in blocklist:
            lig_indices = extract_from_list(block, rest_indices, dimension=1)
            lig_labels = extract_from_list(block, rest_labels, dimension=1)
            lig_coord = extract_from_list(block, rest_coord, dimension=1)
            lig_radii = extract_from_list(block, rest_radii, dimension=1)
            lig_atoms = extract_from_list(block, rest_atoms, dimension=1)

            lig_frac_coord = (
                extract_from_list(block, rest_frac, dimension=1)
                if rest_frac is not None
                else None
            )

            lig_atom_site_labels = (
                extract_from_list(block, rest_atom_site_labels, dimension=1)
                if rest_atom_site_labels is not None
                else None
            )

            logger.info("  Ligand: %s", labels2formula(lig_labels))

            # --- create ligand ---
            if lig_frac_coord is not None:
                ligand = Ligand.from_positional(
                    lig_labels, lig_coord, lig_frac_coord, radii=lig_radii
                )
            else:
                ligand = Ligand.from_positional(lig_labels, lig_coord, radii=lig_radii)

            ligand.set_origin("split_complex")
            ligand.add_parent(self, indices=lig_indices)

            if self.check_parent("unitcell"):
                ligand.add_parent(
                    self.get_parent("unitcell"),
                    indices=[a.get_parent_index("unitcell") for a in lig_atoms],
                )

            if self.check_parent("reference"):
                ligand.add_parent(
                    self.get_parent("reference"),
                    indices=[a.get_parent_index("reference") for a in lig_atoms],
                )

            ligand.set_adjacency_parameters(cov_factor, metal_factor)
            ligand.set_atoms(
                atomlist=lig_atoms,
                create_adjacencies=False,
                atom_site_labels=lig_atom_site_labels,
                use_bond_info=use_bond_info,
            )
            ligand.set_inherit_adjmatrix("molecule")

            self.ligands.append(ligand)

        # ============================================================
        # Metals
        # ============================================================

        self.metals.extend(self.atoms[i] for i in metal_idx)

        return self.ligands, self.metals

    def get_hapticity(self):
        if self.ligands is None:
            self.split_complex()
        self.is_haptic = False
        self.haptic_type = []
        if self.iscomplex:
            for lig in self.ligands:
                if lig.is_haptic is None:
                    lig.get_hapticity()
                if lig.is_haptic:
                    self.is_haptic = True
                for entry in lig.haptic_type:
                    self.haptic_type.append(entry)

        return self.haptic_type

    def save(self, path):
        logger.info(f"SAVING cell2mol MOLECULE object to {path}")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    def get_unique_species(self):
        logger.info("Getting unique species in molecule: %s", self.formula)

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
            and not self.has_ia_iia
            and not self.has_post_transition_metal
        ):
            found = False
            for ldx, typ in enumerate(typelist_mols):
                issame = compare_species(self, typ[0])
                if issame:
                    found = True
                    kdx = typ[1]
                    logger.debug("molecule is the same as type %s", ldx)

            if not found:
                specs_found += 1
                kdx = specs_found
                typelist_mols.append([self, kdx])
                self.unique_species.append(self)
                logger.debug(
                    "New molecule found: formula=%s, added at specie type %d",
                    self.formula,
                    kdx,
                )

            self.unique_indices.append(kdx)
            self.unique_index = kdx
            self.species_list.append(self)

        else:
            # Ensure ligands and metals are available
            if not self.ligands is not None:
                if self.iscomplex or self.has_ia_iia:
                    self.split_complex()
                elif self.has_post_transition_metal:
                    self.split_complex(post_tms=True)
            # Case 2: ligands
            for jdx, lig in enumerate(self.ligands):
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
                        issame = lig.NO_type == typ[0].NO_type
                    else:
                        if (
                            len(lig_groups_labels) == len(typ_groups_labels)
                            and sorted(lig_groups_labels) == sorted(typ_groups_labels)
                            and lig.haptic_type == typ[0].haptic_type
                        ):
                            # if lig.haptic_type == typ[0].haptic_type:
                            issame = compare_species(lig, typ[0])
                        else:
                            issame = False

                    if issame:
                        found = True
                        kdx = typ[1]
                        logger.debug(
                            "ligand %s (%d) is the same with type %d in typelist",
                            lig.formula,
                            jdx,
                            ldx,
                        )
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_ligs.append([lig, kdx])
                    self.unique_species.append(lig)
                    logger.debug(
                        "New ligand found: %s, added at specie type %d",
                        lig.formula,
                        kdx,
                    )

                self.unique_indices.append(kdx)
                lig.unique_index = kdx
                self.species_list.append(lig)

            # Case 3: metals
            for jdx, met in enumerate(self.metals):
                found = False
                for ldx, typ in enumerate(typelist_mets):
                    issame = compare_metals(met, typ[0])
                    if issame:
                        found = True
                        kdx = typ[1]
                        logger.debug(
                            "metal %s (%d) is the same with type %d in typelist",
                            met.formula,
                            jdx,
                            ldx,
                        )
                if not found:
                    specs_found += 1
                    kdx = specs_found
                    typelist_mets.append([met, kdx])
                    self.unique_species.append(met)
                    logger.debug(
                        "New metal found: formula=%s, added at specie type %d",
                        met.formula,
                        kdx,
                    )

                self.unique_indices.append(kdx)
                met.unique_index = kdx
                self.species_list.append(met)

        return self.unique_species

    def get_selected_cs(self):
        if not self.unique_species is not None:
            self.get_unique_species()

        self.selected_cs = []
        for unique_specie in self.unique_species:
            logger.info(
                "Get possible charge states for unique specie %s", unique_specie.formula
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
            logger.info(
                "Get possible charge states for species list %s", specie.formula
            )
            tmp = specie.get_possible_cs()
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

    def assign_charges(self):
        logger.info("Assigning charges for molecule: %s", self.formula)
        for specie in self.unique_species:
            if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
                for jdx, lig in enumerate(self.ligands):
                    if lig.unique_index == specie.unique_index:
                        set_charge_state(specie, lig, mode=1)
                for kdx, met in enumerate(self.metals):
                    if met.unique_index == specie.unique_index:
                        met.set_charge(specie.charge)
            else:
                if self.unique_index == specie.unique_index:
                    set_charge_state(specie, self, mode=1)
        temp = []
        self.create_bonds()
        temp.append(self.error_create_bonds)
        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            prepare_mol(self)

        if any(temp):
            self.error_create_bonds = True
        else:
            self.error_create_bonds = False

        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            prepare_mol(self)
            logger.info("Complex %s %s", self.formula, self.totcharge)
            for jdx, lig in enumerate(self.ligands):
                logger.info(
                    "    Ligand %d %s %s %s",
                    jdx,
                    lig.formula,
                    lig.totcharge,
                    lig.smiles,
                )
            for kdx, met in enumerate(self.metals):
                logger.info("    Metal %d %s %s", kdx, met.formula, met.charge)
        else:
            logger.info(
                "Non-Complex %s %s %s", self.formula, self.totcharge, self.smiles
            )

    def create_bonds(self):
        # First part: Non-complex molecule
        if self.is_non_complex_molecule:
            # Creates bonds between molecule.atoms using the molecule.rdkit_object
            result = create_bonds_specie(self)
            if not result:
                logger.error("Error for non-complex molecule %s", self.formula)
                self.error_create_bonds = True
                return  # Exit the function entirely if creating bonds fails
            else:
                logger.debug("Bonds created for non-complex molecule %s", self.formula)

        # Second part: Complex molecule, add bonds for ligands
        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            self.ligand_smiles_with_H = [lig.smiles for lig in self.ligands]
            self.ligand_smiles = []
            fix_zwitterions_ligands = []

            for lig in self.ligands:
                # Creates bonds between ligand.atoms, using the ligand.rdkit_object
                result = create_bonds_specie(lig)
                if not result:
                    logger.error("Error for ligand %s", lig.formula)

                    self.error_create_bonds = True
                    return  # Exit the function entirely if creating bonds fails for any ligand

                logger.debug("Bonds created for ligand %s", lig.formula)
                logger.debug("Correcting Smiles for ligand %s", lig.formula)
                result, fix_zwitterions = correct_smiles_ligand(lig)
                if not result:
                    logger.error(
                        "Error for ligand %s in correcting smiles", lig.formula
                    )
                    self.error_create_bonds = True
                    return  # Exit the function entirely

                logger.debug("Smiles corrected for ligand %s", lig.formula)
                if fix_zwitterions:
                    fix_zwitterions_ligands.append(lig)
                else:
                    self.ligand_smiles.append(lig.smiles)

            for lig in fix_zwitterions_ligands:
                for atom in lig.atoms:
                    atom.bonds = []

                    logger.debug(
                        "Re-running create_bonds_specie for ligand %s due to zwitterion correction.",
                        lig.formula,
                    )
                result = create_bonds_specie(lig)
                if not result:
                    logger.error(
                        "Error for ligand %s in re-creating bonds", lig.formula
                    )
                    self.error_create_bonds = True
                    return

                logger.debug(
                    "Bonds re-created for ligand %s after zwitterion correction.",
                    lig.formula,
                )
                self.ligand_smiles.append(lig.smiles)

        # Third part : adds metal-ligand bonds, metal-metal bonds, with a zero order
        if self.iscomplex or self.has_ia_iia or self.has_post_transition_metal:
            create_metal_ligand_bonds(self)
            create_metal_metal_bonds(self)

        self.error_create_bonds = False

    def assess_errors(self):
        if self.error_get_poscharges:
            case = 5
        elif self.error_multiple_distrib:
            case = 6
        elif self.error_empty_distrib:
            case = 7
        elif self.error_assign_charge:
            case = 8
        elif self.error_create_bonds:
            case = 9
        elif self.error_get_spin:
            case = 10
        else:
            case = 0

        self.error_case = case
