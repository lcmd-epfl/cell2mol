from __future__ import annotations
import logging
from typing import Any, cast

import numpy as np
from cell2mol.classes.cell import Cell
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.classes.metal import Metal
from cell2mol.classes.molecule import Molecule
from cell2mol.classes.species import Species
from cell2mol.connectivity import split_species
from cell2mol.species_collection import (
    collect_hydrogen_flags,
    update_plausible_charges,
    collect_unique_species,
    map_charges_to_molecules,
)
from cell2mol.operations import extract_from_list, get_moiety_indices_from_labels
from cell2mol.utils import config
from cell2mol.my_types import SubType, NDArray

logger = logging.getLogger(__name__)


class Reference(Cell):
    """CIF reference molecules, species, reported metadata, and processing flags."""

    # Reference molecules and CIF atom labels
    refmoleclist: list[Molecule] | None = None

    atom_site_labels: list[str] | None = None

    # CIF bonds and moiety partitions
    geom_bond_cif: list[tuple[str, str, float]] | None = None
    moiety_list_cif: list[list[str]] | None = None
    moiety_indices: list[list[int]] | None = None

    # Unique species used for charge balancing
    unique_species: list[Species | Metal] | None = None

    # Aggregated hydrogen flags
    has_isolated_H: bool | None = None
    missing_H_in_Carbon: bool | None = None
    missing_H_on_CoordDonor: bool | None = None
    missing_H_in_Water: bool | None = None
    has_missing_H: bool | None = None

    # Charge candidates for unique_species followed by species_list.
    # None marks failed enumeration; copies can occur in both lists.
    plausible_charges: list[list[int] | None] | None = None
    error_plausible_charges: bool | None = None
    # Different charge candidates for copies sharing a unique_index.
    error_inconsistent_plausible_charges: bool | None = None
    inconsistent_plausible_charges: dict[int, list[list[int] | None]] | None = None

    # Reported CIF metadata
    chemical_name: str | None = None
    reported_metal_os: list[tuple[str, int | None]] | None = None
    # Matched metal-state tokens (e.g. Ni_2) and the fraction with known states.
    reported_metal_os_matched: list[str] | None = None
    metal_os_match_confidence: float | None = None
    moiety_dicts: list[dict[str, Any]] | None = None

    # Warning flags: True=active, False=clear, None=check skipped.
    potential_warnings: dict[str, bool | None] | None = None

    subtype: SubType | None = Field(default="reference")

    def set_atom_site_labels(self, atom_site_labels):
        """Store CIF atom-site labels, such as Fe1 or O2."""
        self.atom_site_labels = atom_site_labels

    def set_cif_bond_moiety(self, geom_bond_cif=None, moiety_list_cif=None):
        """Store CIF bonds and moieties, then map moieties to atom indices."""
        self.geom_bond_cif = geom_bond_cif
        self.moiety_list_cif = moiety_list_cif

        self.moiety_indices = None

        if geom_bond_cif is None:
            return

        self.moiety_indices = get_moiety_indices_from_labels(
            self.atom_site_labels, moiety_list_cif
        )

    def set_cif_metadata(self, chemical_name, reported_metal_os, moiety_dicts):
        """Store the chemical name, reported oxidation states, and moiety metadata."""
        self.chemical_name = chemical_name
        self.reported_metal_os = reported_metal_os
        self.moiety_dicts = moiety_dicts

    def set_potential_warnings(self, warnings_dict: dict[str, bool | None]):
        """Store warning flags and log active warnings and skipped checks."""
        self.potential_warnings = warnings_dict

        active = [k for k, v in warnings_dict.items() if v is True]
        if active:
            logger.warning(f"Reference Warnings: {active}")

        skipped = [k for k, v in warnings_dict.items() if v is None]
        if skipped:
            logger.debug(f"Comparisons with CIF Skipped: {skipped}")

    def build_reference_molecules(
        self,
        cov_factor: float | None = None,
        metal_factor: float | None = None,
        use_bond_info: bool | None = None,
    ):
        """Build reference molecules, split complexes, and analyze coordination.

        None options use the corresponding config defaults. Molecules are
        partitioned by CIF moieties or distance-based connectivity."""
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

        # Partition atoms into molecule blocks.
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

            if newmolec.is_metal_complex or newmolec.has_s_block_metal:
                logger.debug("Is Metal complex: %s", newmolec.formula)
                logger.debug("Splitting Metal complex: %s", newmolec.formula)
                newmolec.split_complex()
            elif newmolec.is_post_transition_metal_complex:
                logger.debug(
                    "Splitting post-transition metal complex: %s", newmolec.formula
                )
                newmolec.split_complex(post_tms=True)
            else:
                newmolec.add_parent(newmolec, indices=list(range(newmolec.natoms)))

            self.refmoleclist.append(newmolec)

        logger.info("Found %d reference molecules", len(self.refmoleclist))
        logger.info("Formulas: %s", [ref.formula for ref in self.refmoleclist])

        # Analyze coordination and screen special structural motifs.
        for ref in self.refmoleclist:
            ref.analyze_coordination()
            ref.detect_special_moieties()

        return self.refmoleclist

    def check_hydrogens(self):
        """Check for isolated hydrogen atoms and missing hydrogens."""
        if self.species_list is None:
            self.collect_unique_species()

        flags = collect_hydrogen_flags(self.refmoleclist, self.species_list)
        self.has_isolated_H = flags["has_isolated_H"]
        self.missing_H_in_Carbon = flags["missing_H_in_Carbon"]
        self.missing_H_on_CoordDonor = flags["missing_H_on_CoordDonor"]
        self.missing_H_in_Water = flags["missing_H_in_Water"]
        self.has_missing_H = flags["has_missing_H"]

        return self.has_missing_H

    def collect_unique_species(self):
        """Collect unique species and map all reference species to them."""
        logger.info("Getting unique species in %s", self.subtype)

        if not self.refmoleclist:
            logger.error("Reference molecule list is None")
            return

        (
            self.unique_species,
            self.unique_indices,
            self.species_list,
        ) = collect_unique_species(self.refmoleclist)

    def get_plausible_charges(self) -> list[list[int] | None] | None:
        """Refresh and return charge candidates, preserving missing-H entries as None."""
        assert self.subtype == "reference", (
            "get_plausible_charges should only be called on reference"
        )

        if self.unique_species is None:
            self.collect_unique_species()

        update_plausible_charges(self)
        return self.plausible_charges

    def map_charges_to_reference(self):
        """Map balanced unique-species charges to reference molecules."""
        self.error_assign_charge = map_charges_to_molecules(
            self.unique_species, self.refmoleclist
        )

    def __repr__(self, indirect: bool = False):
        """Return cell details and reference molecule formulas."""
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
    @deprecated("Use Reference() with keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        labels: list[str],
        pos: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]],
        cell_vector: NDArray | list[list[float]],
        cell_param: NDArray | list[float],
    ) -> "Reference":
        """Construct a reference using the legacy positional interface."""
        return cls(
            name=name,
            labels=labels,
            pos=np.asarray(pos),
            frac_coord=np.asarray(frac_coord),
            cell_vector=np.asarray(cell_vector),
            cell_param=np.asarray(cell_param),
        )
