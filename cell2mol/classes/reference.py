from __future__ import annotations
import logging
from typing import Any, cast

import numpy as np
from cell2mol.classes.cell import Cell
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.molecule import Molecule
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import split_species
from cell2mol.species_collection import (
    collect_missing_hydrogens,
    collect_plausible_charges,
    collect_unique_species,
    map_charges_to_molecules,
)
from cell2mol.operations import extract_from_list, get_moiety_indices_from_labels
from cell2mol.elementdata import ElementData
from cell2mol.utils import config
from cell2mol.my_types import SubType, NDArray

logger = logging.getLogger(__name__)

elemdatabase = ElementData()


Labels = list[str]


class Reference(Cell):
    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    # Reference molecule lists
    refmoleclist: list[Molecule] | None = None

    atom_site_labels: list[str] | None = None

    # CIF bond/moiety related attributes
    geom_bond_cif: list[tuple[str, str, float]] | None = None
    moiety_list_cif: list[list[str]] | None = None
    exist_cif_bond_moiety: bool | None = None
    moiety_indices: list[list[int]] | None = None

    # Unique species related attributes
    unique_species: list[Specie | Metal] | None = None

    # Missing H related attributes
    has_isolated_H: bool | None = None
    missing_H_in_Carbon: bool | None = None
    missing_H_on_CoordDonor: bool | None = None
    missing_H_in_Water: bool | None = None
    has_missing_H: bool | None = None

    # Retrieving plausible charges related attributes
    # Plausible integer charges per specie: metal oxidation states, or a
    # ligand/molecule's total charges. NOT one entry per specie -- it is
    # unique_species first, then all of species_list, so species recur at
    # two indices. None marks a specie whose charges could not be found.
    plausible_charges: list[list[int] | None] | None = None
    error_plausible_charges: bool | None = None
    # Set when two entries sharing a unique_index -- i.e. copies of the SAME
    # specie -- enumerated to different charges. Populated by
    # get_plausible_charges; see inconsistent_plausible_charges for the detail.
    error_inconsistent_plausible_charges: bool | None = None
    # {unique_index: sorted list of the distinct charge sets that were found},
    # for reporting. Empty/None when every copy agreed.
    inconsistent_plausible_charges: dict[int, list[list[int] | None]] | None = None

    # Additional CIF related attributes
    chemical_name: str | None = None
    reported_metal_os: list[tuple[str, int]] | None = None
    # Reported metal OS matched to Metal_state tokens (e.g. ["Ni_2"]), produced
    # by standardize_reported_metal_os during compare_metal_oxidation_states.
    # The confidence is the fraction of reported names that matched a metal
    # symbol -- i.e. how reliable the matching step was, not a judgement of the
    # reported values.
    reported_metal_os_matched: list[str] | None = None
    metal_os_match_confidence: float | None = None
    moiety_dicts: list[dict[str, Any]] | None = None

    # Potential warning flags dictionary allowing True, False, or None
    potential_warnings: dict[str, bool | None] | None = None

    # Frozen fields
    subtype: SubType | None = Field(default="reference")

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
        """
        Set additional CIF information such as chemical name, reported metal
        oxidation states, and moiety dictionaries.
        """
        self.chemical_name = chemical_name
        self.reported_metal_os = reported_metal_os
        self.moiety_dicts = moiety_dicts

    def set_potential_warning(self, warnings_dict: dict[str, bool | None]):
        """
        Directly stores the warnings dictionary and logs any active flags.
        Based on CIF mismatch, polynuclear limit,
        mixed metals, and adjacency mismatch, metal coordination differences.
        """
        self.potential_warnings = warnings_dict

        # Log only actual warnings to the system console
        active = [k for k, v in warnings_dict.items() if v is True]
        if active:
            logger.warning(f"Reference Warnings: {active}")

        # Log skipped checks as debug info
        skipped = [k for k, v in warnings_dict.items() if v is None]
        if skipped:
            logger.debug(f"Comparisons with CIF Skipped: {skipped}")

    def get_reference_molecules(
        self,
        cov_factor: float | None = None,
        metal_factor: float | None = None,
        use_bond_info: bool | None = None,
    ):
        """
        Generate reference molecules from atomic fractional coordinates and labels.
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
        bond_data = self.geom_bond_cif

        # Determine blocklist
        blocklist: list[list[int]]
        if use_bond_info:
            logger.info("Using CIF bond/moiety information for blocklist")
            if self.moiety_indices is None:
                logger.error("CIF moiety indices are not available")
                return []
            blocklist = self.moiety_indices
        else:
            blocklist = cast(
                "list[list[int]]",
                split_species(
                    ref_labels,
                    ref_pos,
                    atom_site_labels=atom_site_labels,
                    bond_data=bond_data,
                    cov_factor=cov_factor,
                    metal_factor=metal_factor,
                    warn_on_mismatch=True,
                    detail=True,
                ),
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
            mol_coord = extract_from_list(b, ref_pos.tolist(), dimension=1)
            mol_frac_coord = extract_from_list(b, ref_fracs.tolist(), dimension=1)
            mol_atom_site_labels = (
                extract_from_list(b, atom_site_labels, dimension=1)
                if atom_site_labels is not None
                else None
            )

            newmolec = Molecule.from_positional(mol_labels, mol_coord, mol_frac_coord)
            newmolec.add_parent(self, indices=b)
            newmolec.set_adjacency_parameters(cov_factor, metal_factor)
            newmolec.set_atoms(
                create_adjacencies=True,
                atom_site_labels=mol_atom_site_labels,
                use_bond_info=use_bond_info,
            )

            for atom, idx in zip(newmolec.atoms or [], b):
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

        # Post-processing: coordination analysis and structural-motif screening
        for ref in self.refmoleclist:
            ref.analyze_coordination()
            ref.detect_special_moieties()

        return self.refmoleclist

    def check_hydrogens(self):
        """Check every specie for missing hydrogens and aggregate the flags.

        Feeds ``assess_errors(mode="hydrogens")``. Call after
        ``get_unique_species``.
        """
        if self.species_list is None:
            self.get_unique_species()

        flags = collect_missing_hydrogens(self.refmoleclist, self.species_list)
        self.has_isolated_H = flags["has_isolated_H"]
        self.missing_H_in_Carbon = flags["missing_H_in_Carbon"]
        self.missing_H_on_CoordDonor = flags["missing_H_on_CoordDonor"]
        self.missing_H_in_Water = flags["missing_H_in_Water"]
        self.has_missing_H = flags["has_missing_H"]

        return self.has_missing_H

    def get_unique_species(self):
        """Get unique species, unique indices, and species list in the Reference Cell."""
        logger.info("Getting unique species in %s", self.subtype)

        if not self.refmoleclist:
            logger.error("Reference molecule list is None")
            return

        (
            self.unique_species,
            self.unique_indices,
            self.species_list,
        ) = collect_unique_species(self.refmoleclist)

    def get_plausible_charges(self) -> None:
        """
        Collect the plausible integer charges for every unique specie and every
        entry of the species list. Nothing is selected here -- the final choice
        is made later by the charge balancer.

        A specie with no options records None. Species skipped up front because
        they are missing hydrogens record None too (to keep the list positionally
        aligned with unique_species + species_list) but do NOT set
        error_plausible_charges -- that failure is already reported by
        assess_errors(mode="hydrogens"), and double-reporting it hides which
        structures failed on charge perception alone.
        """
        assert self.subtype == "reference", (
            "get_plausible_charges should only be called on reference"
        )

        if self.unique_species is None:
            self.get_unique_species()

        (
            self.plausible_charges,
            self.error_plausible_charges,
            self.inconsistent_plausible_charges,
        ) = collect_plausible_charges(
            self.unique_species, self.species_list, skip_missing_h=True
        )
        self.error_inconsistent_plausible_charges = bool(
            self.inconsistent_plausible_charges
        )

    def map_charges_to_reference(self):
        """Logic: Propagate charges from Unique Species to Reference Molecules."""
        self.error_assign_charge = map_charges_to_molecules(
            self.unique_species, self.refmoleclist
        )

    def __repr__(self, indirect: bool = False):
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
        pos: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]],
        cell_vector: NDArray | list[list[float]],
        cell_param: NDArray | list[float],
    ) -> "Reference":
        return cls(
            name=name,
            labels=labels,
            pos=np.asarray(pos),  # Using pos which gets aliased to coord
            frac_coord=np.asarray(frac_coord),
            cell_vector=np.asarray(cell_vector),
            cell_param=np.asarray(cell_param),
        )
