from __future__ import annotations
from cell2mol.classes.cell import Cell
from cell2mol.utils import BaseModel
from cell2mol.classes.molecule import Molecule
from typing_extensions import deprecated
from pydantic import Field
from cell2mol.my_types import (
    NDArray,
    Type,
    SubType,
)

import pickle

###############
#### CELLS ####
###############


class Cells(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Required constructor parameters
    name: str
    reference: Cell
    unitcell: Cell
    cell_vector: NDArray
    cell_param: NDArray

    # Frozen fields
    version: str = Field(default="2.0", frozen=True)
    type: Type = Field(default="cells")

    #######################################################
    def save(self, path):
        print(f"SAVING cell2mol CELLS object to {path}")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    #######################################################
    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self):
        to_print = "------------- Cell2mol CELLS Object ----------------\n"
        to_print += f" Version               = {self.version}\n"
        to_print += f" Type                  = {self.type}\n"
        to_print += f" Name (Refcode)        = {self.name}\n"
        to_print += f" Cell Parameters a:c   = {self.cell_param[0:3]}\n"
        to_print += f" Cell Parameters al:ga = {self.cell_param[3:6]}\n"
        # to_print += f' Cell Vector           = {self.cell_vector}\n'
        to_print += "---------------------------------------------------\n"
        if self.reference.refmoleclist is not None:
            to_print += "(Reference)                                \n"
            to_print += f" # of Ref Molecules:   = {len(self.reference.refmoleclist)}\n"
            to_print += " with Formula:                                  \n"
            for idx, ref in enumerate(self.reference.refmoleclist):
                to_print += f"    {idx}: {ref.formula} \n"
        if self.unitcell.moleclist is not None:
            to_print += "(Unitcell)                                \n"
            to_print += f" # Molecules:          = {len(self.unitcell.moleclist)}\n"
            to_print += " with Formula:                               \n"
            for idx, m in enumerate(self.unitcell.moleclist):
                to_print += f"    {idx}: {m.formula} \n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use cells() with the keyword arguments instead.")
    def from_positional(
        cls,
        name: str,
        reference: Cell,
        unitcell: Cell,
        cell_vector: object,
        cell_param: object,
    ) -> "Cells":
        return cls(
            name=name,
            reference=reference,
            unitcell=unitcell,
            cell_vector=cell_vector,
            cell_param=cell_param,
        )
