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
from cell2mol.compare import compare_species, compare_metals
from cell2mol.operations import extract_from_list, get_moiety_indices_from_labels
from cell2mol.elementdata import ElementData
from cell2mol.utils import config
from cell2mol.my_types import SubType, NDArray
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
        Scans the reference molecules for isolated hydrogens (dangling H/D), then
        runs ``Specie.check_hydrogens`` per specie (metals skipped) and OR-folds
        their flags into the reference-level ``has_missing_H`` / ``missing_H_*``
        used by ``assess_errors(mode="hydrogens")``. Call after
        ``get_unique_species``.
        """
        if self.species_list is None:
            self.get_unique_species()

        # Isolated single-atom species. A lone H/D is a dangling hydrogen
        # (error code 1); any other lone atom is only reported.
        has_isolated_h = False
        for ref in self.refmoleclist or []:
            if ref.natoms != 1:
                continue
            label = (ref.atoms or [])[0].label
            site_label = ref.atom_site_labels[0] if ref.atom_site_labels else "N/A"
            if label in {"H", "D"}:
                has_isolated_h = True
                logger.warning(
                    "  Isolated hydrogen found %s (%s)", ref.labels[0], site_label
                )
            else:
                logger.warning(
                    "  Isolated atom found %s (%s)", ref.labels[0], site_label
                )

        self.has_isolated_H = has_isolated_h
        logger.info("Has isolated hydrogen: %s", self.has_isolated_H)

        missing = []
        for specie in self.species_list or []:
            # Metals subclass Atom (not Specie) and carry no hydrogens, so they
            # have no check_hydrogens() and are always clean -- skip them.
            if specie.subtype == "metal":
                continue
            if specie.check_hydrogens():
                missing.append(specie)

        logger.info(
            "Missing hydrogens found in %d/%d species: %s",
            len(missing),
            len(self.species_list or []),
            [s.formula for s in missing],
        )

        species = self.species_list or []
        self.missing_H_in_Carbon = any(
            getattr(s, "missing_H_in_Carbon", False) for s in species
        )
        self.missing_H_on_CoordDonor = any(
            getattr(s, "missing_H_on_CoordDonor", False) for s in species
        )
        self.missing_H_in_Water = any(
            getattr(s, "missing_H_in_Water", False) for s in species
        )
        self.has_missing_H = (
            self.missing_H_in_Carbon
            or self.missing_H_on_CoordDonor
            or self.missing_H_in_Water
        )

        if self.has_missing_H:
            logger.info(
                "Missing hydrogens | check_hydrogens=%s carbon=%s, coordinated_donor=%s, water=%s",
                self.has_missing_H,
                self.missing_H_in_Carbon,
                self.missing_H_on_CoordDonor,
                self.missing_H_in_Water,
            )

        return self.has_missing_H

    def get_unique_species(self):
        """Get unique species, unique indices, and species list in the Reference Cell."""
        logger.info("Getting unique species in %s", self.subtype)

        self.unique_species = []
        self.unique_indices = []
        self.species_list = []
        typelist_mols = []  # temporary variable
        typelist_ligs = []  # temporary variable
        typelist_mets = []  # temporary variable

        specs_found = -1
        if not self.refmoleclist:
            logger.error("Reference molecule list is None")
            return
        moleclist = self.refmoleclist

        for idx, mol in enumerate(moleclist):
            logger.debug("Molecule (%d) formula=%s", idx, mol.formula)
            if mol.is_non_complex_molecule:  # Non-complex molecules
                found = False
                kdx = None
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
                assert kdx is not None
                self.unique_indices.append(kdx)
                mol.unique_index = kdx
                self.species_list.append(mol)
            else:  # Complex molecules
                # metals
                for jdx, met in enumerate(mol.metals or []):
                    found = False
                    kdx: int | None = None
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
                    assert kdx is not None
                    self.unique_indices.append(kdx)
                    met.unique_index = kdx
                    self.species_list.append(met)

                if mol.ligands is None:
                    if mol.iscomplex or mol.has_ia_iia:
                        mol.split_complex()
                    elif mol.has_post_transition_metal:
                        mol.split_complex(post_tms=True)
                # ligands
                for jdx, lig in enumerate(mol.ligands or []):
                    found = False
                    kdx = None
                    for ldx, typ in enumerate(typelist_ligs):
                        if lig.is_nitrosyl is None:
                            lig.evaluate_as_nitrosyl()
                        if typ[0].is_nitrosyl is None:
                            typ[0].evaluate_as_nitrosyl()
                        if lig.haptic_type is None:
                            lig.get_hapticity()
                        if typ[0].haptic_type is None:
                            typ[0].get_hapticity()

                        lig_groups_labels = [g.labels for g in lig.groups or []]
                        typ_groups_labels = [g.labels for g in typ[0].groups or []]

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
                    assert kdx is not None
                    self.unique_indices.append(kdx)
                    lig.unique_index = kdx
                    self.species_list.append(lig)

        logger.info("Unique species: %s", [s.formula for s in self.unique_species])
        logger.info("Unique indices: %s", self.unique_indices)
        logger.info("Species list: %s", [s.formula for s in self.species_list])

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

        self.plausible_charges = []

        # Process unique_species first, then the full species_list
        all_targets = [
            (specie, "unique specie") for specie in self.unique_species or []
        ]
        all_targets.extend(
            [(specie, "species list") for specie in self.species_list or []]
        )

        # Positions of entries that are None because enumeration was never
        # attempted, as opposed to attempted and failed.
        skipped_for_missing_h: set[int] = set()

        for idx, (specie, context_label) in enumerate(all_targets):
            logger.info(
                "Get plausible charge states for %s: %s",
                context_label,
                specie.formula,
            )

            if specie.subtype != "metal":
                if specie.has_missing_H is None:
                    specie.check_hydrogens()
                if specie.has_missing_H:
                    logger.warning(
                        "Specie %s has missing hydrogens; skipping charge-state "
                        "enumeration and recording None",
                        specie.formula,
                    )
                    self.plausible_charges.append(None)
                    skipped_for_missing_h.add(idx)
                    continue

            if specie.subtype == "metal":
                plausible = specie.get_plausible_os()
                if not plausible:
                    # Appending None indicates a failure to find options for this species
                    self.plausible_charges.append(None)
                    continue
                self.plausible_charges.append(plausible)
            else:
                plausible = specie.get_plausible_charge_states()
                if not plausible:
                    self.plausible_charges.append(None)
                    continue
                self.plausible_charges.append(
                    [cs.specie_total_charge for cs in plausible]
                )

        # Error flag covers only genuine enumeration failures -- a specie skipped
        # for missing hydrogens is reported by the "hydrogens" mode instead.
        unexplained = [
            idx
            for idx, charges in enumerate(self.plausible_charges)
            if charges is None and idx not in skipped_for_missing_h
        ]
        self.error_plausible_charges = bool(unexplained)

        if skipped_for_missing_h:
            logger.info(
                "Plausible charges: %d specie entr(ies) skipped for missing "
                "hydrogens (not counted as a charge error)",
                len(skipped_for_missing_h),
            )
        if unexplained:
            logger.error(
                "Plausible charges: %d specie entr(ies) with no charges found: %s",
                len(unexplained),
                [all_targets[i][0].formula for i in unexplained],
            )

        self._check_plausible_charges_consistency(all_targets, skipped_for_missing_h)

    def _check_plausible_charges_consistency(
        self, all_targets, skipped_for_missing_h: set[int]
    ) -> None:
        """Flag unique species whose copies did not enumerate to the same charges.

        Entries sharing a ``unique_index`` are, by ``compare_species``, the same
        specie -- symmetry-related copies differing only in atom ordering. They
        must therefore yield the same charge options. When they do not, the
        selected charge depends on which copy happened to be stored first (the
        balancer reads ``unique_species`` only), which is never a property of
        the chemistry.

        The usual cause is that bond perception is not atom-order invariant, so
        one copy finds a Lewis structure the other misses. Recorded rather than
        raised: a disagreement means the answer is order-dependent, not
        necessarily wrong.
        """
        self.error_inconsistent_plausible_charges = False
        self.inconsistent_plausible_charges = {}

        if not self.plausible_charges:
            return

        # Charges by unique_index, skipping entries that were never attempted.
        by_unique_index: dict[int, list[list[int] | None]] = {}
        for idx, (specie, _context) in enumerate(all_targets):
            if idx in skipped_for_missing_h:
                continue
            unique_index = getattr(specie, "unique_index", None)
            if unique_index is None:
                continue
            charges = self.plausible_charges[idx]
            by_unique_index.setdefault(unique_index, []).append(charges)

        for unique_index, recorded in sorted(by_unique_index.items()):
            # Order within a charge list carries no meaning, so compare as sets.
            distinct = {
                None if charges is None else tuple(sorted(set(charges)))
                for charges in recorded
            }
            if len(distinct) <= 1:
                continue

            self.error_inconsistent_plausible_charges = True
            self.inconsistent_plausible_charges[unique_index] = [
                None if entry is None else list(entry)
                for entry in sorted(distinct, key=lambda e: (e is None, e or ()))
            ]
            formula = next(
                (
                    spec.formula
                    for spec, _ in all_targets
                    if getattr(spec, "unique_index", None) == unique_index
                ),
                "?",
            )
            logger.warning(
                "Plausible charges disagree between copies of the same specie "
                "%s (unique_index=%d): %s. The charge finally assigned depends "
                "on which copy is stored first.",
                formula,
                unique_index,
                self.inconsistent_plausible_charges[unique_index],
            )

    def map_charges_to_reference(self):
        """Logic: Propagate charges from Unique Species to Reference Molecules."""
        self.error_assign_charge = False

        for specie in self.unique_species or []:
            specie_unique_index = getattr(specie, "unique_index", None)
            for ref in self.refmoleclist or []:
                try:
                    if ref.is_non_complex_molecule:
                        # Direct match for simple molecules
                        if ref.unique_index == specie_unique_index:
                            set_charge_state(specie, ref, mode=1)
                    else:
                        # Attempt to match all Ligands
                        for lig in ref.ligands or []:
                            if lig.unique_index == specie_unique_index:
                                try:
                                    set_charge_state(specie, lig, mode=1)
                                except Exception:
                                    self.error_assign_charge = True
                                if lig.totcharge is None:
                                    logger.warning(
                                        "Ligand %s charge not set from Specie %s",
                                        lig.formula,
                                        specie_unique_index,
                                    )
                                    self.error_assign_charge = True

                        # Attempt to match all Metals
                        for met in ref.metals or []:
                            if met.unique_index == specie_unique_index:
                                try:
                                    specie_charge = getattr(specie, "charge", None)
                                    if specie_charge is not None:
                                        met.set_charge(specie_charge)
                                except Exception:
                                    self.error_assign_charge = True

                except Exception as e:
                    # Catch-all for unexpected logic errors in the outer ref loop
                    self.error_assign_charge = True
                    logger.error(
                        "Error mapping charge for Specie %s: %s",
                        specie_unique_index,
                        e,
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
