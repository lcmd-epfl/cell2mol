from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import numpy as np
from pydantic import Field
from typing_extensions import deprecated

from cell2mol.classes.atom import Atom
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.metal import Metal
from cell2mol.classes.protonation import Protonation
from cell2mol.element_utils import (
    get_df_block_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
    get_element_count,
    get_radii,
    labels2electrons,
    labels2formula,
)
from cell2mol.connectivity import build_adjacency, count_bonds_by_element_pair
from cell2mol.elementdata import ElementData
from cell2mol.my_types import NDArray, RDKitObject, SubType
from cell2mol.operations import compute_centroid, extract_from_list
from cell2mol.utils import BaseModel, config
import logging

if TYPE_CHECKING:
    from cell2mol.classes.ligand import Ligand
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.cell import Cell

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


class Specie(BaseModel):
    """Shared structure, connectivity, and charge-state data for molecular species."""

    # Structural data
    labels: list[str]
    coord: NDArray
    frac_coord: NDArray | None = None
    radii: NDArray | None = None
    atnums: list[int] | None = None
    atom_site_labels: list[str] | None = None
    atoms: list[Atom] | None = None

    # Parent relationships
    parents: list["Specie | Cell"] = Field(default_factory=list)
    parents_indices: list[list[int]] = Field(default_factory=list)

    # Connectivity and geometry
    cov_factor: float = Field(default=config.COV_FACTOR)
    metal_factor: float = Field(default=config.METAL_FACTOR)
    adj_types: NDArray | None = None
    adjmat: NDArray | None = None
    adjnum: NDArray | None = None
    madjmat: NDArray | None = None
    madjnum: NDArray | None = None
    centroid: NDArray | None = None
    frac_centroid: NDArray | None = None
    element_count: NDArray | None = None

    # Charge states
    totcharge: int | None = None
    atomic_charges: list[int] | NDArray | None = None
    rdkit_obj: RDKitObject | None = Field(default=None)
    smiles: str | None = None
    protonation_states: list[Protonation] | None = None
    charge_state: ChargeState | None = None
    plausible_charge_states: list[ChargeState] | None = Field(default=None)

    # Structural motifs, including cages embedded in larger derivatives.
    has_porphyrin: bool | None = None
    has_fullerene: bool | None = None
    has_borane: bool | None = None

    # Reasons protonation or bond perception could not complete.
    protonation_warning: str | None = None
    valence_search_too_large: bool | None = None
    bond_perception_capped: bool | None = None

    # Missing hydrogens on carbon, coordinated donors, or water.
    has_missing_H: bool | None = None
    missing_H_in_Carbon: bool | None = None
    missing_H_on_CoordDonor: bool | None = None
    missing_H_in_Water: bool | None = None

    # Type metadata
    subtype: SubType | None = None
    origin: str | None = None
    type: str = Field(default="specie", frozen=True)

    def model_post_init(self, context: Any) -> None:
        """Validate coordinate lengths and initialize missing radii."""
        super().model_post_init(context)
        assert len(self.labels) == len(self.coord)
        if self.frac_coord is not None:
            assert len(self.coord) == len(self.frac_coord)
        if self.radii is not None:
            # Legacy data may contain list or scalar radii.
            if isinstance(self.radii, (list, tuple)):
                assert len(self.labels) == len(self.radii)
            elif len(self.labels) > 1:
                self.radii = np.asarray(get_radii(self.labels))
        else:
            self.radii = np.asarray(get_radii(self.labels))

    @property
    def formula(self) -> str:
        """Return the formula built from element labels."""
        return labels2formula(self.labels)

    @property
    def eleccount(self) -> int:
        """Return the electron count for a neutral structure."""
        return labels2electrons(self.labels)

    @property
    def natoms(self) -> int:
        """Return the number of atoms."""
        return len(self.labels)

    @property
    def is_metal_complex(self) -> bool:
        """Return whether d- or f-block metals are present."""
        return bool(get_df_block_metal_idxs(self.labels))

    @property
    def has_s_block_metal(self) -> bool:
        """Return whether alkali or alkaline-earth metals are present."""
        return bool(get_alkali_alkaline_earth_metal_idxs(self.labels))

    @property
    def is_post_transition_metal_complex(self) -> bool:
        """Detect post-transition metals only when no d/f- or s-block metals are present."""
        return (
            not self.is_metal_complex
            and not self.has_s_block_metal
            and bool(get_post_transition_metal_idxs(self.labels))
        )

    @property
    def is_metal_free_molecule(self):
        """Return whether this is a molecule without any supported metal category."""
        return (
            self.subtype == "molecule"
            and not self.is_metal_complex
            and not self.has_s_block_metal
            and not self.is_post_transition_metal_complex
        )

    @property
    def indices(self) -> list[int]:
        """Return local atom indices in structure order."""
        return list(range(self.natoms))

    def check_parent(self, subtype: str):
        """Return whether a parent with this subtype exists."""
        for p in self.parents:
            if p.subtype == subtype:
                return True
        return False

    def get_parent(self, subtype: str):
        """Return the parent with this subtype, or None."""
        for p in self.parents:
            if p.subtype == subtype:
                return p
        return None

    def get_parent_indices(self, subtype: str):
        """Return atom indices in the requested parent, or None."""
        for idx, p in enumerate(self.parents):
            if p.subtype == subtype:
                return self.parents_indices[idx]
        return None

    def get_adjacency_parameters(self) -> tuple[float, float]:
        """Return covalent-radius and metal-radius scaling factors."""
        return self.cov_factor, self.metal_factor

    def set_subtype(self, subtype: SubType):
        """Store the species subtype."""
        self.subtype = subtype

    def set_origin(self, origin: str):
        """Store the structure creation source."""
        self.origin = origin

    def _register_parent(
        self, parent: "Specie | Cell", indices: list[int], overwrite: bool
    ) -> None:
        """Store one parent and its atom map, keeping the serialized lists aligned."""
        for position, existing in enumerate(self.parents):
            if existing.subtype == parent.subtype:
                if overwrite:
                    self.parents[position] = parent
                    self.parents_indices[position] = indices.copy()
                return
        self.parents.append(parent)
        self.parents_indices.append(indices.copy())

    def add_parent(
        self, parent: "Specie | Cell", indices: list[int], overwrite: bool = True
    ) -> None:
        """Register a parent and compose atom maps for its ancestors."""
        if len(indices) != self.natoms:
            raise ValueError("Parent index map must contain one index per atom")
        if any(not isinstance(i, (int, np.integer)) or i < 0 for i in indices):
            raise ValueError("Parent indices must be non-negative integers")
        if isinstance(parent, Specie) and any(i >= parent.natoms for i in indices):
            raise ValueError("Parent index out of range")
        if len(self.parents) != len(self.parents_indices):
            raise ValueError("Parent objects and index maps are inconsistent")
        if not overwrite and self.check_parent(parent.subtype or ""):
            return

        ancestors = []
        if isinstance(parent, Specie):
            if len(parent.parents) != len(parent.parents_indices):
                raise ValueError(
                    "Parent objects and ancestor index maps are inconsistent"
                )
            for ancestor, parent_indices in zip(parent.parents, parent.parents_indices):
                if len(parent_indices) != parent.natoms:
                    raise ValueError(
                        "Ancestor index map must contain one index per parent atom"
                    )
                ancestors.append((ancestor, [parent_indices[i] for i in indices]))
        self._register_parent(parent, indices, overwrite)
        for ancestor, own_indices in ancestors:
            self._register_parent(ancestor, own_indices, overwrite)

    def compute_centroid(self):
        """Compute and store Cartesian and available fractional centroids."""
        self.centroid = np.asarray(compute_centroid(np.array(self.coord)))
        if self.frac_coord is not None:
            self.frac_centroid = np.asarray(compute_centroid(np.array(self.frac_coord)))
        return self.centroid

    def set_fractional_coordinates(
        self, frac_coord: NDArray | list[list[float]]
    ) -> None:
        """Store fractional coordinates with one row per atom."""
        assert len(frac_coord) == len(self.coord)
        self.frac_coord = np.asarray(frac_coord)

    def get_atomic_numbers(self):
        """Build atom objects if needed, then return their atomic numbers."""
        if self.atoms is None:
            self.set_atoms()
        self.atnums = []
        for at in self.atoms or []:
            self.atnums.append(at.atnum)
        return self.atnums

    def detect_porphyrin(self) -> bool:
        """Detect and store whether a porphyrin-family N4 macrocycle is present."""
        from cell2mol.charge.special_cases import is_porphyrin_macrocycle

        has_porphyrin, _ = is_porphyrin_macrocycle(
            self.get_atomic_numbers(), self.adjmat
        )
        self.has_porphyrin = has_porphyrin
        return self.has_porphyrin

    def detect_fullerene(self) -> bool:
        """Detect and store whether a fullerene cage is present."""
        from cell2mol.charge.special_cases import has_fullerene

        is_cage, _ = has_fullerene(self.get_atomic_numbers(), self.adjmat)
        self.has_fullerene = is_cage
        return self.has_fullerene

    def detect_borane(self) -> bool:
        """Detect and store whether a borane or carborane cage is present."""
        from cell2mol.charge.special_cases import is_borane_cage

        is_cage, _ = is_borane_cage(self.get_atomic_numbers(), self.adjmat)
        self.has_borane = is_cage
        return self.has_borane

    def detect_special_moieties(self):
        """Detect fullerene, porphyrin, and borane motifs and store their flags."""
        self.detect_fullerene()
        if self.has_fullerene:
            logger.debug("%s %s has a fullerene", self.subtype, self.formula)

        self.detect_porphyrin()
        if self.has_porphyrin:
            logger.debug("%s %s has a porphyrin", self.subtype, self.formula)

        self.detect_borane()
        if self.has_borane:
            logger.debug("%s %s has a borane/carborane", self.subtype, self.formula)

    def compute_element_count(self, heavy_only: bool = False):
        """Compute and store element counts, optionally excluding hydrogen."""
        self.element_count = get_element_count(self.labels, heavy_only=heavy_only)
        return self.element_count

    def build_adjacency_matrix(
        self,
        use_bond_info: bool | None = None,
        metal_only: bool = False,
    ):
        """Build and store adjacency and neighbor counts, optionally for metal links only."""
        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
        cov_factor = self.cov_factor
        metal_factor = self.metal_factor

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        adjmat = build_adjacency(
            labels=self.labels,
            positions=self.coord,
            atom_site_labels=self.atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
            metal_only=metal_only,
            warn_on_mismatch=False,
            detail=False,
        )

        if adjmat is None:
            if metal_only:
                self.madjmat = None
                self.madjnum = None
                return self.madjmat, self.madjnum
            else:
                self.adjmat = None
                self.adjnum = None
                return self.adjmat, self.adjnum

        adjnum = adjmat.sum(axis=1)

        if metal_only:
            self.madjmat = adjmat
            self.madjnum = adjnum
            return self.madjmat, self.madjnum
        else:
            self.adjmat = adjmat
            self.adjnum = adjnum
            return self.adjmat, self.adjnum

    def compute_adjacency_types(self, use_bond_info: bool | None = None):
        """Build adjacency if needed, then compute and store adjacency types."""
        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO
        if self.adjmat is None:
            self.build_adjacency_matrix(use_bond_info=use_bond_info, metal_only=False)
        if self.adjmat is not None:
            self.adj_types = count_bonds_by_element_pair(self.labels, self.adjmat)
        return self.adj_types

    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        """Store covalent-radius and metal-radius scaling factors."""
        self.cov_factor = cov_factor
        self.metal_factor = metal_factor

    def set_atoms(
        self,
        atomlist: list[Atom] | None = None,
        create_adjacencies: bool = False,
        atom_site_labels: list[str] | None = None,
        use_bond_info: bool | None = None,
    ):
        """Use supplied atoms or create them, link parents, and optionally set connectivity."""
        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO
        if atomlist is not None:
            self.atoms = atomlist.copy()
            for idx, at in enumerate(self.atoms):
                at.add_parent(self, index=idx)
        else:
            self.atoms = []

            df_block_metal_idxs = get_df_block_metal_idxs(self.labels)
            s_block_metal_idxs = get_alkali_alkaline_earth_metal_idxs(self.labels)

            assert self.radii is not None
            for idx, label in enumerate(self.labels):
                ismetal = (
                    elemdatabase.elementblock[label] == "d"
                    or elemdatabase.elementblock[label] == "f"
                )
                if len(get_alkali_alkaline_earth_metal_idxs([label])) > 0:
                    ismetal = True
                if len(df_block_metal_idxs) == 0 and len(s_block_metal_idxs) == 0:
                    if len(get_post_transition_metal_idxs([label])) > 0:
                        ismetal = True

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
                newatom.add_parent(self, index=idx)
                self.atoms.append(newatom)

        if atom_site_labels is not None:
            self.atom_site_labels = atom_site_labels.copy()
            for idx, at in enumerate(self.atoms):
                at.set_atom_site_label(atom_site_labels[idx])

        if create_adjacencies:
            if self.adjmat is None:
                self.build_adjacency_matrix(
                    use_bond_info=use_bond_info, metal_only=False
                )

            if self.madjmat is None:
                self.build_adjacency_matrix(
                    use_bond_info=use_bond_info, metal_only=True
                )

            if self.adjmat is not None and self.madjmat is not None:
                assert self.adjnum is not None
                assert self.madjnum is not None
                for idx, at in enumerate(self.atoms):
                    at.set_connectivity(
                        self.adjmat[idx],
                        self.madjmat[idx],
                        self.adjnum[idx],
                        self.madjnum[idx],
                    )

    def inherit_adjacency(self, parent_subtype: str):
        """Copy adjacency submatrices and parent neighbor counts using the atom map."""
        exists = self.check_parent(parent_subtype)
        if not exists:
            logger.debug(f"{parent_subtype=} does not exist")
            return None
        parent = cast("Specie", self.get_parent(parent_subtype))
        indices = self.get_parent_indices(parent_subtype)
        assert parent is not None
        assert indices is not None
        if parent.madjnum is None:
            logger.debug(f"{parent_subtype=} does not have madjnum")
            return None
        assert parent.madjmat is not None
        assert parent.adjmat is not None
        assert parent.adjnum is not None

        # Convert generic object arrays to integer connectivity arrays.
        self.madjmat = np.stack(
            extract_from_list(indices, parent.madjmat.tolist(), dimension=2), axis=0
        ).astype(int)
        self.madjnum = np.stack(
            extract_from_list(indices, parent.madjnum.tolist(), dimension=1), axis=0
        ).astype(int)
        self.adjmat = np.stack(
            extract_from_list(indices, parent.adjmat.tolist(), dimension=2), axis=0
        ).astype(int)
        self.adjnum = np.stack(
            extract_from_list(indices, parent.adjnum.tolist(), dimension=1), axis=0
        ).astype(int)

    def check_hydrogens(self):
        """Check for missing hydrogens and store carbon, donor, and water flags."""
        from cell2mol.hydrogen import detect_species_missing_hydrogens

        (
            has_missing_h,
            missing_h_in_carbon,
            missing_h_on_coordinated_donor,
            missing_h_in_water,
        ) = detect_species_missing_hydrogens(self)

        if has_missing_h:
            logger.info(
                "Missing hydrogens in specie %s | carbon=%s, coordinated_donor=%s, water=%s",
                self.formula,
                missing_h_in_carbon,
                missing_h_on_coordinated_donor,
                missing_h_in_water,
            )

        self.has_missing_H = has_missing_h
        self.missing_H_in_Carbon = missing_h_in_carbon
        self.missing_H_on_CoordDonor = missing_h_on_coordinated_donor
        self.missing_H_in_Water = missing_h_in_water

        return self.has_missing_H

    def get_protonation_states(self):
        """Enumerate protonation states for ligands and metal-free molecules; otherwise None."""
        from cell2mol.charge.protonation_states import enumerate_protonation_states

        self.protonation_states = None
        self.protonation_warning = None

        if not (self.subtype == "ligand" or self.is_metal_free_molecule):
            return self.protonation_states

        if self.subtype == "ligand":
            ligand_self = cast("Ligand", self)
            if ligand_self.is_haptic is None:
                ligand_self.update_hapticity()
            if ligand_self.denticity is None:
                ligand_self.update_denticity()
            if ligand_self.is_nitrosyl is None:
                ligand_self.detect_nitrosyl()

        self.protonation_states = enumerate_protonation_states(self)
        return self.protonation_states

    def get_plausible_charge_states(self):
        """Return cached charge candidates, or generate them for ligands and metal-free molecules."""
        from cell2mol.charge.state_generation import generate_charge_states

        if self.plausible_charge_states is not None:
            return self.plausible_charge_states

        self.plausible_charge_states = None

        if not (self.subtype == "ligand" or self.is_metal_free_molecule):
            return self.plausible_charge_states

        if self.protonation_states is None:
            self.get_protonation_states()
        logger.debug("Enumerating charge states for %s", self.formula)

        self.plausible_charge_states = generate_charge_states(self)
        return self.plausible_charge_states

    def set_charges(
        self,
        totcharge: int | None = None,
        atomic_charges: list[int] | NDArray | None = None,
        smiles: str | None = None,
        rdkit_obj: object = None,
    ) -> None:
        """Store total and atom charges, updating atom objects and supplied molecular forms."""
        if totcharge is not None:
            self.totcharge = totcharge
        elif totcharge is None and atomic_charges is not None:
            self.totcharge = int(np.sum(atomic_charges))
        elif totcharge is None and atomic_charges is None:
            self.totcharge = None
        if atomic_charges is not None:
            self.atomic_charges = atomic_charges
            if self.atoms is None:
                self.set_atoms()
            for idx, a in enumerate(self.atoms or []):
                a.set_charge(self.atomic_charges[idx])
        if smiles is not None:
            self.smiles = smiles
        if rdkit_obj is not None:
            self.rdkit_obj = rdkit_obj

    def reset_charge(self):
        """Clear assigned charges and molecular forms, including charges on atom objects."""
        self.totcharge = None
        self.atomic_charges = None
        self.smiles = None
        self.rdkit_obj = None
        self.plausible_charge_states = None

        for a in self.atoms or []:
            a.reset_charge()

    def __str__(self):
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        """Return structure, connectivity, charge, and origin details."""
        to_print = ""
        if not indirect:
            to_print += "------------- Cell2mol SPECIE Object --------------\n"
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
        if self.subtype == "molecule" and cast("Molecule", self).spin is not None:
            to_print += (
                f" Spin                         = {cast('Molecule', self).spin}\n"
            )
        if self.smiles is not None:
            to_print += f" Smiles                       = {self.smiles}\n"
        if self.origin is not None:
            to_print += f" Origin                       = {self.origin}\n"
        if not indirect:
            to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use Specie() with keyword arguments instead.")
    def from_positional(
        cls,
        labels: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
    ) -> "Specie":
        """Construct a species using the legacy positional interface."""
        return cls(
            labels=labels,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )
