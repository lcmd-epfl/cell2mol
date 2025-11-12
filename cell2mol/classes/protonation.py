import json
import numpy as np
from typing import Annotated, Any
from cell2mol.my_types import Type
from cell2mol.connectivity import (
    get_radii,
    get_adjmatrix,
    get_adjmatrix_from_cif_bonds,
    labels2formula,
)
from cell2mol.utils.pydantic import BaseModel, serialize_circular_references
from cell2mol.my_types import NDArray

# Pydantic imports for the converted classes
from pydantic import Field, PlainSerializer, computed_field, model_serializer
from typing_extensions import deprecated
from cell2mol.elementdata import ElementData

elemdatabase = ElementData()


#######################################################
class Protonation(BaseModel):
    # Required constructor parameters
    labels: list[str]
    coords: list[list[float]]  # Note: renamed from 'coord' to match usage
    cov_factor: float
    added_atoms: int
    addedlist: list[int]
    block: list[int]
    metal_electrons: list[int]
    elemlist: list[str]

    # Optional constructor parameters with defaults
    tmpsmiles: str = Field(default=" ")
    o_s: int = Field(default=0)
    typ: str = Field(default="Local")
    parent: Annotated[object | None, PlainSerializer(serialize_circular_references)] = (
        Field(default=None)
    )

    # Computed attributes with proper defaults
    natoms: int | None = None
    formula: str | None = None
    atnums: list[int] | None = None
    radii: list | None = None

    # Conditionally set attributes with None defaults (eliminates hasattr need)
    atom_site_labels_indices: list[int] | None = None
    atom_site_labels: list[str] | None = None
    status: bool | None = None
    adjmat: NDArray | None = None
    adjnum: NDArray | None = None

    # Frozen fields
    version: str = Field(default="2.0", frozen=True)
    type: Type = Field(default="protonation")

    @model_serializer(
        mode="plain",
    )
    def model_dump_json(self, **kwargs) -> str:
        # print("PROTONATION.model_dump_json")
        # TODO romaingrx: add this back in the final json later
        return "{}"

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
        if self.parent is not None:
            refcell = self.parent.get_parent("reference")
            geom_bond_cif = getattr(refcell, "geom_bond_cif", None)

            if refcell is not None:
                self.atom_site_labels_indices = [
                    atom.get_parent_index("reference") for atom in self.parent.atoms
                ]
                self.atom_site_labels = [
                    refcell.atom_site_labels[idx]
                    for idx in self.atom_site_labels_indices
                ]
                # print("PROTONATION.atom_site_labels_indices", self.atom_site_labels_indices)
                # print("PROTONATION.atom_site_labels", self.atom_site_labels)

            if refcell is not None and getattr(refcell, "exist_cif_bond_moiety", False):
                self.status, adjmat, adjnum = get_adjmatrix_from_cif_bonds(
                    self.labels, self.coords, self.atom_site_labels, geom_bond_cif
                )
                print(
                    "PROTONATION.get_adjmatrix_from_cif_bonds",
                    adjmat.shape,
                    adjnum.shape,
                )
                count = 0
                if len(self.addedlist) > 0:
                    for idx, add in enumerate(self.addedlist):
                        if add != 0:
                            count += 1
                            added_idx = len(self.addedlist) - 1 + count
                            print("PROTONATION.added_idx", f"{idx=} {added_idx=}")
                            adjmat[idx, added_idx] += 1
                            adjmat[added_idx, idx] += 1
                            adjnum[idx] += 1
                            adjnum[added_idx] += 1

                self.adjmat = adjmat
                self.adjnum = adjnum
            else:
                self.status, self.adjmat, self.adjnum, warning = get_adjmatrix(
                    self.labels, self.coords, self.cov_factor, self.radii
                )
                if warning:
                    print("PROTONATION.get_adjmatrix warning:", warning)
                    self.status = False

    def reorder(self, map, debug: int = 0):
        if debug > 0:
            print("PROTONATION.REORDER. labels:", self.labels)
        if debug > 0:
            print("PROTONATION.REORDER. received map:", map)

        ## for protonation states with added atoms, the reorder map will have fewer items. Correct it here
        mapext = np.copy(map)
        if self.added_atoms > 0 and len(map) < len(self.labels):
            for ldx in range(0, self.added_atoms):
                mapext = np.append(mapext, len(map) + ldx)
            if debug > 0:
                print("PROTONATION.REORDER. extended map:", mapext)
        print("PROTONATION.REORDER. extended map:", mapext, len(mapext))
        print("PROTONATION.REORDER. map:", map, len(map))
        print("PROTONATION.REORDER. labels:", self.labels, len(self.labels))
        print("PROTONATION.REORDER. addedlist:", self.addedlist, len(self.addedlist))
        assert len(mapext) == len(self.labels)
        assert len(map) == len(self.addedlist)
        if len(map) > 0:
            self.labels = list(np.array(self.labels)[mapext])
            self.coords = list(np.array(self.coords)[mapext])
            self.atnums = list(np.array(self.atnums)[mapext])
            self.radii = list(np.array(self.radii)[mapext])
            # No more hasattr check needed - atom_site_labels is always defined (can be None)
            if self.atom_site_labels is not None:
                self.atom_site_labels = list(np.array(self.atom_site_labels)[map])
            self.addedlist = list(np.array(self.addedlist)[map])
            self.block = list(np.array(self.block)[map])
            self.metal_electrons = list(np.array(self.metal_electrons)[map])
            self.elemlist = list(np.array(self.elemlist)[map])

            self.typ = "Reordered"
            refcell = self.parent.get_parent("reference")
            geom_bond_cif = getattr(refcell, "geom_bond_cif", None)
            if refcell is not None and getattr(refcell, "exist_cif_bond_moiety", False):
                self.status, adjmat, adjnum = get_adjmatrix_from_cif_bonds(
                    self.labels, self.coords, self.atom_site_labels, geom_bond_cif
                )
                print(
                    "PROTONATION.get_adjmatrix_from_cif_bonds",
                    adjmat.shape,
                    adjnum.shape,
                )
                count = 0
                if len(self.addedlist) > 0:
                    for idx, add in enumerate(self.addedlist):
                        if add != 0:
                            count += 1
                            added_idx = len(self.addedlist) - 1 + count
                            print("PROTONATION.added_idx", f"{idx=} {added_idx=}")
                            adjmat[idx, added_idx] += 1
                            adjmat[added_idx, idx] += 1
                            adjnum[idx] += 1
                            adjnum[added_idx] += 1

                self.adjmat = adjmat
                self.adjnum = adjnum
            else:
                self.status, self.adjmat, self.adjnum, warning = get_adjmatrix(
                    self.labels, self.coords, self.cov_factor, self.radii
                )
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
        to_print += f" Type                            = {self.typ}\n"
        to_print += f" Atoms added in positions        = {self.addedlist}\n"
        to_print += f" Atoms blocked (no atoms added)  = {self.block}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use protonation() with the keyword arguments instead.")
    def from_positional(
        cls,
        labels: list[str],
        coord: list[list[float]],
        cov_factor: float,
        added_atoms: int,
        addedlist: list[int],
        block: list[int],
        metal_electrons: list[int],
        elemlist: list[str],
        tmpsmiles: str = " ",
        o_s: int = 0,
        typ: str = "Local",
        parent: object = None,
    ) -> "Protonation":
        return cls(
            labels=labels,
            coords=coord,  # Note: using coords here to match the field name
            cov_factor=cov_factor,
            added_atoms=added_atoms,
            addedlist=addedlist,
            block=block,
            metal_electrons=metal_electrons,
            elemlist=elemlist,
            tmpsmiles=tmpsmiles,
            o_s=o_s,
            typ=typ,
            parent=parent,
        )
