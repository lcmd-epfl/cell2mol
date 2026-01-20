from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Self

from pydantic import Field
from typing_extensions import deprecated

from cell2mol.classes.cell import Cell
from cell2mol.my_types import Format, NDArray, Type
from cell2mol.utils import BaseModel
from cell2mol.utils import config

###############
#### CELLS ####
###############

logger = logging.getLogger(__name__)


class Cells(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Required constructor parameters
    name: str
    reference: Cell
    unitcell: Cell | None = None
    cell_vector: NDArray
    cell_param: NDArray

    # Frozen fields
    version: str = Field(default=config.VERSION, frozen=True)
    type: Type = Field(default="cells")

    #######################################################
    def save(self, path: str | Path, *, format: Format = "json"):
        if format == "json":
            self._save_as_json(path)
        elif format == "pickle":
            self._save_as_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: str | Path, *, format: Format = "json") -> Self:
        if format == "json":
            return cls._load_from_json(path)
        elif format == "pickle":
            return cls._load_from_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @deprecated("Use json format instead")
    def _save_as_pickle(
        self,
        path: str | Path,
    ):
        logger.warning("Use json format instead")
        with open(path, "wb") as fil:
            pickle.dump(self, fil)

    def _save_as_json(
        self,
        path: str | Path,
    ):
        if not str(path).endswith(".json"):
            logger.warning("Use `.json` extension instead for path: %s", path)
        with open(path, "w") as fd:
            fd.write(self.to_json(indent=4))

    @classmethod
    def _load_from_json(
        cls,
        path: str | Path,
    ) -> Self:
        with open(path, "r") as fd:
            return cls.from_json(fd.read())

    @classmethod
    @deprecated("Use json format instead")
    def _load_from_pickle(
        cls,
        path: str | Path,
    ):
        with open(path, "rb") as fil:
            return pickle.load(fil)

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
