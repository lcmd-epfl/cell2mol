from __future__ import annotations

import numpy as np
from typing import Any, TYPE_CHECKING, cast
from cell2mol.my_types import Type
from cell2mol.connectivity import build_distance_adjacency, build_cif_bond_adjacency
from cell2mol.element_utils import labels2formula, get_radii
from cell2mol.utils.pydantic import BaseModel
from cell2mol.my_types import NDArray
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.elementdata import ElementData
import logging
from cell2mol.utils import config

if TYPE_CHECKING:
    from cell2mol.classes.specie import Specie
    from cell2mol.classes.reference import Reference

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


class Protonation(BaseModel):
    """Protonated coordinates, donor-site changes, and atom connectivity."""

    # Protonated structure inputs
    labels: list[str]
    coord: NDArray
    cov_factor: float  # Covalent-radius scale for distance-based adjacency.

    # Added protons, with per-site lists indexed by original parent atoms.
    n_protons_added: int

    site_proton_counts: list[int]

    # Electrons donated from ligand lone pairs.
    ligand_donor_electrons: list[int]

    # Modes: none, deterministic (chemical rules), or combinatorial.
    mode: str | None = None

    # Special-case builder name; None for the general engine.
    origin: str | None = None

    parent: Specie | None = Field(default=None)

    # Derived atom data
    natoms: int | None = None
    formula: str | None = None
    atnums: list[int] | None = None
    radii: NDArray | None = None

    # Original atom labels and protonated connectivity
    atom_site_labels: list[str] | None = None
    status: bool | None = None
    adjmat: NDArray | None = None
    adjnum: NDArray | None = None

    type: Type = Field(default="protonation")

    def model_post_init(self, __context: Any) -> None:
        """Initialize atom data and build connectivity when a parent is resolved."""
        self.natoms = len(self.labels)
        self.formula = labels2formula(self.labels)
        self.atnums = [elemdatabase.elementnr[label] for label in self.labels]
        self.radii = np.asarray(get_radii(self.labels))

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
                and parent.adjmat is not None
                and parent.adjmat.shape[0] == len(self.labels)
            ):
                self.status = True
                # Convert inherited object arrays to numeric adjacency matrices.
                self.adjmat = np.array(parent.adjmat, dtype=int, copy=True)
                self.adjnum = (
                    np.array(parent.adjnum, dtype=int, copy=True)
                    if parent.adjnum is not None
                    else self.adjmat.sum(axis=1)
                )
                return

            use_bond_info = config.USE_BOND_INFO
            if use_bond_info:
                self.status = True
                assert self.atom_site_labels is not None
                assert bond_data is not None
                self.adjmat = build_cif_bond_adjacency(
                    self.labels,
                    self.coord,
                    atom_site_labels=self.atom_site_labels,
                    bond_data=bond_data,
                )
            else:
                self.status, self.adjmat, _ = build_distance_adjacency(
                    self.labels,
                    self.coord,
                    radii=self.radii,
                    cov_factor=self.cov_factor,
                )
            self._connect_added_protons(clear_accidental_bonds=not use_bond_info)

    def _connect_added_protons(self, *, clear_accidental_bonds: bool) -> None:
        """Connect trailing protons to donor sites and update adjacency counts."""
        assert self.adjmat is not None
        added_index = len(self.labels) - self.n_protons_added
        for site, count in enumerate(self.site_proton_counts):
            for _ in range(count):
                if clear_accidental_bonds:
                    self.adjmat[added_index, :] = 0
                    self.adjmat[:, added_index] = 0
                    self.adjmat[site, added_index] = 1
                    self.adjmat[added_index, site] = 1
                else:
                    # Preserve CIF connectivity and add the surrogate bond.
                    self.adjmat[site, added_index] += 1
                    self.adjmat[added_index, site] += 1
                logger.debug(
                    "Connected added proton %d to donor site %d", added_index, site
                )
                added_index += 1
        self.adjnum = self.adjmat.sum(axis=1)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        """Return protonation inputs, status, and donor-site changes."""
        to_print = ""
        to_print += "------------- Cell2mol Protonation ----------------\n"
        to_print += f" Status                          = {self.status}\n"
        to_print += f" Labels                          = {self.labels}\n"
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
        parent: Specie | None = None,
        origin: str | None = None,
    ) -> "Protonation":
        """Construct a protonation state using the legacy positional interface."""
        return cls(
            labels=labels,
            coord=coord,
            cov_factor=cov_factor,
            n_protons_added=n_protons_added,
            site_proton_counts=site_proton_counts,
            ligand_donor_electrons=ligand_donor_electrons,
            mode=mode,
            parent=parent,
            origin=origin,
        )
