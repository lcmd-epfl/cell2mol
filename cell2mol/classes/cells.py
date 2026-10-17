from __future__ import annotations

from cell2mol.utils.serialization import ModelFileMixin

import logging
from typing import TYPE_CHECKING, cast

import numpy as np


from pydantic import Field
from typing_extensions import deprecated

from cell2mol.classes.cell import Cell
from cell2mol.my_types import NDArray, Type
from cell2mol.utils import BaseModel
from cell2mol.utils import config

if TYPE_CHECKING:
    from cell2mol.classes.reference import Reference
    from cell2mol.classes.unitcell import UnitCell


logger = logging.getLogger(__name__)


class Cells(ModelFileMixin, BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Required constructor parameters
    name: str
    reference: Cell
    unitcell: Cell | None = None
    cell_vector: NDArray
    cell_param: NDArray

    # Type metadata
    version: str = Field(default=config.VERSION, frozen=True)
    type: Type = Field(default="cells")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        to_print = "------------- Cell2mol CELLS Object ----------------\n"
        to_print += f" Version               = {self.version}\n"
        to_print += f" Type                  = {self.type}\n"
        to_print += f" Name (Refcode)        = {self.name}\n"
        to_print += f" Cell Parameters a:c   = {self.cell_param[0:3]}\n"
        to_print += f" Cell Parameters al:ga = {self.cell_param[3:6]}\n"
        to_print += "---------------------------------------------------\n"
        reference = cast("Reference", self.reference)
        if reference.refmoleclist is not None:
            to_print += "(Reference)                                \n"
            to_print += f" # of Ref Molecules:   = {len(reference.refmoleclist)}\n"
            to_print += " with Formula:                                  \n"
            for idx, ref in enumerate(reference.refmoleclist):
                to_print += f"    {idx}: {ref.formula} \n"
        unitcell = cast("UnitCell | None", self.unitcell)
        if unitcell is not None and unitcell.moleclist is not None:
            to_print += "(Unitcell)                                \n"
            to_print += f" # Molecules:          = {len(unitcell.moleclist)}\n"
            to_print += " with Formula:                               \n"
            for idx, mol in enumerate(unitcell.moleclist):
                to_print += f"    {idx}: {mol.formula} \n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use cells() with the keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        reference: Cell,
        unitcell: Cell,
        cell_vector: NDArray | list[list[float]],
        cell_param: NDArray | list[float],
    ) -> "Cells":
        return cls(
            name=name,
            reference=reference,
            unitcell=unitcell,
            cell_vector=np.asarray(cell_vector),
            cell_param=np.asarray(cell_param),
        )
