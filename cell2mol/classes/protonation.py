import numpy as np
from typing import Any
from cell2mol.my_types import Type
from cell2mol.connectivity import get_adjmatrix, get_adjmatrix_from_cif_bonds
from cell2mol.element_utils import labels2formula, get_radii
from cell2mol.utils.pydantic import BaseModel
from cell2mol.my_types import NDArray
from pydantic import Field, computed_field
from typing_extensions import Literal, deprecated
from cell2mol.elementdata import ElementData
import logging
from cell2mol.utils import config

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


class Protonation(BaseModel):
    # Required constructor parameters
    labels: list[str]
    coord: NDArray
    cov_factor: float

    # Total number of protons added
    n_protons_added: int

    # Protons added per site (len = length of parent labels)
    site_proton_counts: list[int]

    # Electrons donated from ligand lone pairs per site (len = length of parent labels)
    ligand_donor_electrons: list[int]

    # Protonation mode
    # "none" : no protonation
    # "heuristic": apply chemical rules
    # "combinatorial": explore combinations
    mode: str | Literal["none", "heuristic", "combinatorial"] | None = None

    parent: object | None = Field(default=None)

    # Computed attributes with proper defaults
    natoms: int | None = None
    formula: str | None = None
    atnums: list[int] | None = None
    radii: NDArray | None = None

    # Conditionally set attributes with None defaults (eliminates hasattr need)
    atom_site_labels_indices: list[int] | None = None
    atom_site_labels: list[str] | None = None
    status: bool | None = None
    adjmat: NDArray | None = None
    adjnum: NDArray | None = None

    # Frozen fields
    type: Type = Field(default="protonation")

    # NOTE: Removed custom model_serializer that was returning "{}" string.
    # This was breaking deserialization. Protonation now serializes normally.

    @computed_field
    @property
    def computed_natoms(self) -> int:
        return len(self.labels)

    @computed_field
    @property
    def computed_formula(self) -> str:
        return labels2formula(self.labels)

    @computed_field
    @property
    def computed_atnums(self) -> list[int]:
        return [elemdatabase.elementnr[label] for label in self.labels]

    @computed_field
    @property
    def computed_radii(self) -> list[float]:
        return get_radii(self.labels)

    def model_post_init(self, __context: Any) -> None:
        # Set computed values
        self.natoms = self.computed_natoms
        self.formula = self.computed_formula
        self.atnums = self.computed_atnums
        self.radii = self.computed_radii

        # Handle conditional attribute setting based on parent
        # Note: parent may be a string UUID during deserialization, skip in that case
        if self.parent is not None and not isinstance(self.parent, str):
            refcell = self.parent.get_parent("reference")
            bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None

            if refcell is not None:
                self.atom_site_labels_indices = [
                    atom.get_parent_index("reference") for atom in self.parent.atoms
                ]
                self.atom_site_labels = [
                    refcell.atom_site_labels[idx]
                    for idx in self.atom_site_labels_indices
                ]
            use_bond_info = config.USE_BOND_INFO
            if use_bond_info:
                self.status = True
                adjmat = get_adjmatrix_from_cif_bonds(
                    self.labels,
                    self.coord,
                    atom_site_labels=self.atom_site_labels,
                    bond_data=bond_data,
                )
                adjnum = adjmat.sum(axis=1)
                base_count = len(self.labels) - self.n_protons_added
                proton_count = 0
                for idx, n_added in enumerate(self.site_proton_counts):
                    if n_added > 0:
                        coordinating_atom_label = (
                            f"{self.labels[idx]} ({self.atom_site_labels[idx]})"
                            if self.atom_site_labels
                            else self.labels[idx]
                        )
                        logger.debug(
                            "Coordinating atom site (idx: %d) %s %s add %s protons",
                            idx,
                            coordinating_atom_label,
                            self.coord[idx],
                            n_added,
                        )
                        for _ in range(n_added):
                            added_idx = base_count + proton_count
                            proton_count += 1
                            logger.debug(
                                "     Added proton site (idx: %d) %s %s",
                                added_idx,
                                self.labels[added_idx],
                                self.coord[added_idx],
                            )
                            adjmat[idx, added_idx] += 1
                            adjmat[added_idx, idx] += 1
                            adjnum[idx] += 1
                            adjnum[added_idx] += 1

                self.adjmat = adjmat
                self.adjnum = adjnum
            else:
                # 1. Generate the initial adjacency matrix
                self.status, self.adjmat, warning = get_adjmatrix(
                    self.labels,
                    self.coord,
                    radii=self.radii,
                    cov_factor=self.cov_factor,
                    # add_atom=True,  # Not needed since we manually fix connectivity below
                )

                # 2. Synchronize adjnum with the new matrix
                self.adjnum = self.adjmat.sum(axis=1)

                # 3. Fix connectivity for unconditionally added protons
                base_count = len(self.labels) - self.n_protons_added
                proton_count = 0
                for idx, n_added in enumerate(self.site_proton_counts):
                    if n_added > 0:
                        coordinating_atom_label = (
                            f"{self.labels[idx]} ({self.atom_site_labels[idx]})"
                            if self.atom_site_labels
                            else self.labels[idx]
                        )
                        logger.debug(
                            "Coordinating atom site (idx: %d) %s %s add %s protons",
                            idx,
                            coordinating_atom_label,
                            self.coord[idx],
                            n_added,
                        )

                        for _ in range(n_added):
                            # Calculate the absolute index of the added atom in the current matrix
                            added_idx = base_count + proton_count
                            proton_count += 1
                            logger.debug(
                                "     Added proton site (idx: %d) %s %s",
                                added_idx,
                                self.labels[added_idx],
                                self.coord[added_idx],
                            )

                            # Clear all 'accidental' bonds for the added atom
                            self.adjmat[added_idx, :] = 0
                            self.adjmat[:, added_idx] = 0

                            # Enforce the single intended bond
                            self.adjmat[idx, added_idx] = 1
                            self.adjmat[added_idx, idx] = 1

                # 4. Final Recalculation: Always recalculate adjnum after manual adjmat changes
                self.adjnum = self.adjmat.sum(axis=1)

            # if warning:
            #     self.status = False

    def reorder(self, map):
        ## for protonation states with added protons, the reorder map will have fewer items. Correct it here
        mapext = np.copy(map)
        if self.n_protons_added > 0 and len(map) < len(self.labels):
            for ldx in range(0, self.n_protons_added):
                mapext = np.append(mapext, len(map) + ldx)

        assert len(mapext) == len(self.labels)
        assert len(map) == len(self.site_proton_counts)
        if len(map) > 0:
            self.labels = list(np.array(self.labels)[mapext])
            self.coord = list(np.array(self.coord)[mapext])
            self.atnums = list(np.array(self.atnums)[mapext])
            self.radii = list(np.array(self.radii)[mapext])
            # No more hasattr check needed - atom_site_labels is always defined (can be None)
            if self.atom_site_labels is not None:
                self.atom_site_labels = list(np.array(self.atom_site_labels)[map])
            self.site_proton_counts = list(np.array(self.site_proton_counts)[map])
            self.ligand_donor_electrons = list(
                np.array(self.ligand_donor_electrons)[map]
            )

            self.typ = "Reordered"
            refcell = self.parent.get_parent("reference")
            bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
            use_bond_info = config.USE_BOND_INFO
            if use_bond_info:
                self.status = True
                adjmat = get_adjmatrix_from_cif_bonds(
                    self.labels,
                    self.coord,
                    atom_site_labels=self.atom_site_labels,
                    bond_data=bond_data,
                )
                adjnum = adjmat.sum(axis=1)
                count = 0
                if len(self.site_proton_counts) > 0:
                    for idx, n_added in enumerate(self.site_proton_counts):
                        if n_added != 0:
                            count += 1
                            added_idx = len(self.site_proton_counts) - 1 + count
                            adjmat[idx, added_idx] += 1
                            adjmat[added_idx, idx] += 1
                            adjnum[idx] += 1
                            adjnum[added_idx] += 1

                self.adjmat = adjmat
                self.adjnum = adjnum
            else:
                self.status, self.adjmat, warning = get_adjmatrix(
                    self.labels,
                    self.coord,
                    radii=self.radii,
                    cov_factor=self.cov_factor,
                )
                self.adjnum = self.adjmat.sum(axis=1)
        return self

    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self):
        to_print = ""
        to_print += "------------- Cell2mol Protonation ----------------\n"
        to_print += f" Status                          = {self.status}\n"
        to_print += f" Labels                          = {self.labels}\n"
        # No more hasattr check needed - atom_site_labels is always defined (can be None)
        if self.atom_site_labels is not None:
            to_print += f" Atom site labels                = {self.atom_site_labels}\n"
        to_print += f" Type                            = {self.type}\n"
        to_print += f" Protonation Mode                = {self.mode}\n"
        to_print += f" Number of added protons         = {self.n_protons_added}\n"
        to_print += f" Protons added in positions      = {self.site_proton_counts}\n"
        if sum(self.ligand_donor_electrons) > 0:
            to_print += (
                f" Ligand donor electrons          = {self.ligand_donor_electrons}\n"
            )
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use protonation() with the keyword arguments instead.")
    def from_positional(
        cls,
        labels: list[str],
        coord: NDArray,
        cov_factor: float,
        n_protons_added: int,
        site_proton_counts: list[int],
        ligand_donor_electrons: list[int],
        mode: str = None,
        parent: object = None,
    ) -> "Protonation":
        return cls(
            labels=labels,
            coord=coord,
            cov_factor=cov_factor,
            n_protons_added=n_protons_added,
            site_proton_counts=site_proton_counts,
            ligand_donor_electrons=ligand_donor_electrons,
            mode=mode,
            parent=parent,
        )
