from __future__ import annotations

import numpy as np
from typing import Any, TYPE_CHECKING, cast
from cell2mol.my_types import Type
from cell2mol.connectivity import build_distance_adjacency, build_cif_bond_adjacency
from cell2mol.element_utils import symbols2formula, get_radii
from cell2mol.utils.pydantic import BaseModel
from cell2mol.my_types import NDArray
from pydantic import AliasChoices, Field, model_validator
from typing_extensions import deprecated
from cell2mol.elementdata import ElementData
import logging
from cell2mol.utils import config

if TYPE_CHECKING:
    from cell2mol.classes.species import Species
    from cell2mol.classes.reference import Reference

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


_DERIVED_FIELDS = frozenset(
    {
        "n_protons_added",
        "natoms",
        "formula",
        "atnums",
        "atomic_numbers",
        "radii",
        "neighbor_counts",
        "adjnum",
    }
)


class Protonation(BaseModel):
    """Protonated coordinates, donor-site changes, and atom connectivity."""

    # Protonated structure inputs
    symbols: list[str] = Field(validation_alias=AliasChoices("symbols", "labels"))
    coord: NDArray
    cov_factor: float  # Covalent-radius scale for distance-based adjacency.

    # Added protons, with per-site lists indexed by original parent atoms.
    site_proton_counts: list[int]

    # Electrons donated from ligand lone pairs.
    ligand_donor_electrons: list[int]

    # Modes: none, deterministic (chemical rules), or combinatorial.
    mode: str | None = None

    # Special-case builder name; None for the general engine.
    origin: str | None = None

    parent: Species | None = Field(default=None)

    # Original atom labels and protonated connectivity
    atom_site_labels: list[str] | None = None
    is_valid: bool | None = Field(
        default=None, validation_alias=AliasChoices("is_valid", "status")
    )
    adjacency_matrix: NDArray | None = Field(
        default=None, validation_alias=AliasChoices("adjacency_matrix", "adjmat")
    )

    type: Type = Field(default="protonation")

    @property
    def labels(self) -> list[str]:
        """Compatibility alias for symbols."""
        return self.symbols

    @labels.setter
    def labels(self, value: list[str]) -> None:
        self.symbols = value

    @property
    def atnums(self) -> list[int]:
        """Compatibility alias for atomic_numbers."""
        return self.atomic_numbers

    @property
    def status(self) -> bool | None:
        """Compatibility alias for is_valid."""
        return self.is_valid

    @status.setter
    def status(self, value: bool | None) -> None:
        self.is_valid = value

    @property
    def n_protons_added(self) -> int:
        """Return the total number of protons added at original atom sites."""
        return int(sum(self.site_proton_counts))

    @property
    def natoms(self) -> int:
        """Return the number of atoms in the protonated structure."""
        return len(self.symbols)

    @property
    def formula(self) -> str:
        """Build the formula from the current element labels."""
        return symbols2formula(self.symbols)

    @property
    def atomic_numbers(self) -> list[int]:
        """Return atomic numbers in protonated atom order."""
        return [elemdatabase.elementnr[label] for label in self.symbols]

    @property
    def radii(self) -> NDArray:
        """Return covalent radii from the current element labels."""
        return np.asarray(get_radii(self.symbols))

    @model_validator(mode="before")
    @classmethod
    def _migrate_serialized_data(cls, data: Any) -> Any:
        """Discard legacy snapshots of derived attributes."""
        if isinstance(data, dict):
            data = data.copy()
            for old, new in {"labels": "symbols", "status": "is_valid"}.items():
                if old in data:
                    data.setdefault(new, data.pop(old))
            return {
                name: value
                for name, value in data.items()
                if name not in _DERIVED_FIELDS
            }
        return data

    @property
    def adjmat(self) -> NDArray | None:
        """Compatibility alias for adjacency_matrix."""
        return self.adjacency_matrix

    @adjmat.setter
    def adjmat(self, value: NDArray | None) -> None:
        self.adjacency_matrix = value

    @property
    def neighbor_counts(self) -> NDArray | None:
        """Return adjacency row sums for the current protonated structure."""
        if self.adjacency_matrix is None:
            return None
        return self.adjacency_matrix.sum(axis=1)

    @property
    def adjnum(self) -> NDArray | None:
        """Compatibility alias for neighbor_counts."""
        return self.neighbor_counts

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore connectivity fields from legacy pickle data."""
        for old, new in {
            "labels": "symbols",
            "status": "is_valid",
            "adjmat": "adjacency_matrix",
        }.items():
            data = state.get("__dict__", {})
            if old in data:
                data.setdefault(new, data.pop(old))
            fields = state.get("__pydantic_fields_set__", set())
            if old in fields:
                fields.discard(old)
                fields.add(new)
        for key in ("__dict__", "__pydantic_extra__", "__pydantic_fields_set__"):
            container = state.get(key)
            if container is None:
                continue
            for name in _DERIVED_FIELDS:
                if isinstance(container, dict):
                    container.pop(name, None)
                else:
                    container.discard(name)
        super().__setstate__(state)

    def model_post_init(self, __context: Any) -> None:
        """Build missing connectivity, preserving stored results and status."""
        if self.adjacency_matrix is not None:
            return

        # Serialized parent UUIDs are resolved after model construction.
        if self.parent is not None and not isinstance(self.parent, str):
            refcell = cast("Reference", self.parent.get_parent("reference"))
            bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None

            if refcell is not None:
                raw_indices = [
                    atom.get_parent_index("reference")
                    for atom in self.parent.atoms or []
                ]
                reference_indices = [idx for idx in raw_indices if idx is not None]
                refcell_labels = refcell.atom_site_labels or []
                self.atom_site_labels = [
                    refcell_labels[idx] for idx in reference_indices
                ]

            # Reuse parent connectivity when no protons were added.
            parent = self.parent
            if (
                self.n_protons_added == 0
                and parent.adjacency_matrix is not None
                and parent.adjacency_matrix.shape[0] == len(self.symbols)
            ):
                self.is_valid = True
                # Convert inherited object arrays to numeric adjacency matrices.
                self.adjacency_matrix = np.array(
                    parent.adjacency_matrix, dtype=int, copy=True
                )
                return

            use_bond_info = config.USE_BOND_INFO
            if use_bond_info:
                self.is_valid = True
                assert self.atom_site_labels is not None
                assert bond_data is not None
                self.adjacency_matrix = build_cif_bond_adjacency(
                    self.symbols,
                    self.coord,
                    atom_site_labels=self.atom_site_labels,
                    bond_data=bond_data,
                )
            else:
                self.is_valid, self.adjacency_matrix, _ = build_distance_adjacency(
                    self.symbols,
                    self.coord,
                    radii=self.radii,
                    cov_factor=self.cov_factor,
                )
            self._connect_added_protons(clear_accidental_bonds=not use_bond_info)

    def _connect_added_protons(self, *, clear_accidental_bonds: bool) -> None:
        """Connect trailing protons to donor sites in the adjacency matrix."""
        assert self.adjacency_matrix is not None
        added_index = len(self.symbols) - self.n_protons_added
        for site, count in enumerate(self.site_proton_counts):
            for _ in range(count):
                if clear_accidental_bonds:
                    self.adjacency_matrix[added_index, :] = 0
                    self.adjacency_matrix[:, added_index] = 0
                    self.adjacency_matrix[site, added_index] = 1
                    self.adjacency_matrix[added_index, site] = 1
                else:
                    # Preserve CIF connectivity and add the surrogate bond.
                    self.adjacency_matrix[site, added_index] += 1
                    self.adjacency_matrix[added_index, site] += 1
                logger.debug(
                    "Connected added proton %d to donor site %d", added_index, site
                )
                added_index += 1

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        """Return protonation inputs, status, and donor-site changes."""
        to_print = ""
        to_print += "------------- Cell2mol Protonation ----------------\n"
        to_print += f" Is Valid                        = {self.is_valid}\n"
        to_print += f" Symbols                         = {self.symbols}\n"
        if self.atom_site_labels is not None:
            to_print += f" Atom site labels                = {self.atom_site_labels}\n"
        to_print += f" Type                            = {self.type}\n"
        to_print += f" Protonation Mode                = {self.mode}\n"
        if self.origin is not None:
            to_print += f" Built by                        = {self.origin}\n"
        to_print += f" Number of added protons         = {self.n_protons_added}\n"
        to_print += f" Protons added in positions      = {self.site_proton_counts}\n"
        if sum(self.ligand_donor_electrons) > 0:
            to_print += (
                f" Ligand donor electrons          = {self.ligand_donor_electrons}\n"
            )
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use Protonation() with keyword arguments instead.")
    def from_positional(
        cls,
        labels: list[str],
        coord: NDArray,
        cov_factor: float,
        n_protons_added: int,
        site_proton_counts: list[int],
        ligand_donor_electrons: list[int],
        mode: str | None = None,
        parent: Species | None = None,
        origin: str | None = None,
    ) -> "Protonation":
        """Accept legacy arguments; added-proton totals come from site counts."""
        return cls(
            symbols=labels,
            coord=coord,
            cov_factor=cov_factor,
            site_proton_counts=site_proton_counts,
            ligand_donor_electrons=ligand_donor_electrons,
            mode=mode,
            parent=parent,
            origin=origin,
        )
