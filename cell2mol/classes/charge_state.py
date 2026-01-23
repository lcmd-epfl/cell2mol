import numpy as np
from typing import Any, Tuple
from cell2mol.classes.protonation import Protonation
from cell2mol.my_types import RDKitObject
from cell2mol.utils.pydantic import BaseModel
from pydantic import Field
from typing_extensions import deprecated
import logging

logger = logging.getLogger(__name__)


def eval_chargelist(atom_charges: list[int]) -> Tuple[int, int, bool]:
    abstotal = int(np.abs(np.sum(atom_charges)))
    abs_atlist = []
    for a in atom_charges:
        abs_atlist.append(abs(a))
    abs_atcharge = int(np.sum(abs_atlist))
    if any(b > 0 for b in atom_charges) and any(b < 0 for b in atom_charges):
        zwitt = True
    else:
        zwitt = False
    return abstotal, abs_atcharge, zwitt


class ChargeState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Required constructor parameters
    status: bool
    uncorr_total_charge: int
    uncorr_atom_charges: list[int]
    rdkit_obj: RDKitObject
    smiles: str
    charge_tried: int
    allow: bool
    protonation: Protonation = Field(...)

    # Computed attributes with proper defaults
    uncorr_abstotal: int | None = None
    uncorr_abs_atcharge: int | None = None
    uncorr_zwitt: bool | None = None
    coincide: bool | None = None

    # Copied from protonation with proper defaults
    addedlist: list[int] | None = None
    metal_electrons: list[int] | None = None
    elemlist: list[str] | None = None
    block: list[int] | None = None

    # Initialized attributes with defaults
    corr_total_charge: int = Field(default=0)
    corr_atom_charges: list[int] = Field(default_factory=list)

    # Final computed attributes with proper defaults
    corr_abstotal: int | None = None
    corr_abs_atcharge: int | None = None
    corr_zwitt: bool | None = None

    # Frozen fields
    type: str = Field(default="charge_state", frozen=True)

    def model_post_init(self, __context: Any) -> None:
        # Compute initial derived values
        self.uncorr_abstotal, self.uncorr_abs_atcharge, self.uncorr_zwitt = (
            eval_chargelist(self.uncorr_atom_charges)
        )

        # Set coincide flag
        self.coincide = self.uncorr_total_charge == self.charge_tried

        # Copy attributes from protonation
        self.addedlist = self.protonation.addedlist
        self.metal_electrons = self.protonation.metal_electrons
        self.elemlist = self.protonation.elemlist
        self.block = self.protonation.block

        # logger.debug(
        #     "Initializing ChargeState | SMILES: %s | Uncorr Charge: %d | Charge Tried: %d",
        #     self.smiles,
        #     self.uncorr_total_charge,
        #     self.charge_tried,
        # )
        # logger.debug("Added List: %s %d", self.addedlist, len(self.addedlist))
        # logger.debug("block: %s %d", self.block, len(self.block))
        # logger.debug(
        #     "Metal Electrons: %s %d", self.metal_electrons, len(self.metal_electrons)
        # )
        # logger.debug("Element List: %s %d", self.elemlist, len(self.elemlist))
        # logger.debug(
        #     "Uncorrected Atom Charges: %s %s",
        #     self.uncorr_atom_charges,
        #     len(self.uncorr_atom_charges),
        # )
        # Corrects the Charge of atoms with addedH
        count = 0
        if len(self.addedlist) > 0:
            # Iterates over the original number of ligand atoms, thus without the added H
            for idx, add in enumerate(self.addedlist):
                if add != 0:
                    count += 1
                    # logger.debug(
                    #     "Correcting atom index %d | Original Charge: %d | Added: %d | block: %d | Metal Electrons: %d",
                    #     idx,
                    #     self.uncorr_atom_charges[idx],
                    #     self.addedlist[idx],
                    #     self.block[idx],
                    #     self.metal_electrons[idx],
                    # )
                    corrected = (
                        self.uncorr_atom_charges[idx]
                        - self.addedlist[idx]
                        + self.block[idx]
                        + self.metal_electrons[idx]
                        - self.uncorr_atom_charges[len(self.addedlist) - 1 + count]
                    )
                    self.corr_atom_charges.append(corrected)
                    # last term corrects for cases in which a charge has been assigned to the added atom
                else:
                    self.corr_atom_charges.append(self.uncorr_atom_charges[idx])
            self.corr_total_charge = int(np.sum(self.corr_atom_charges))
        else:
            self.corr_total_charge = self.uncorr_total_charge
            self.corr_atom_charges = self.uncorr_atom_charges.copy()

        self.corr_abstotal, self.corr_abs_atcharge, self.corr_zwitt = eval_chargelist(
            self.corr_atom_charges
        )

    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self):
        to_print = ""
        to_print += "------------- Cell2mol Charge State ---------------\n"
        to_print += f" Status                          = {self.status}\n"
        to_print += f" Smiles                          = {self.smiles}\n"
        to_print += f" Charge Tried                    = {self.charge_tried}\n"
        to_print += f" Uncorrected Total Charge        = {self.uncorr_total_charge}\n"
        to_print += f" Corrected Total Charge          = {self.corr_total_charge}\n"
        to_print += f" Corrected Absolute Total Charge = {self.corr_abs_atcharge}\n"
        to_print += f" Corrected Is Zwitterion?        = {self.corr_zwitt}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use charge_state() with the keyword arguments instead.")
    def from_positional(
        cls,
        status: bool,
        uncorr_total_charge: int,
        uncorr_atom_charges: list[int],
        rdkit_obj: object,
        smiles: str,
        charge_tried: int,
        allow: bool,
        protonation: object,
    ) -> "ChargeState":
        return cls(
            status=status,
            uncorr_total_charge=uncorr_total_charge,
            uncorr_atom_charges=uncorr_atom_charges,
            rdkit_obj=rdkit_obj,
            smiles=smiles,
            charge_tried=charge_tried,
            allow=allow,
            protonation=protonation,
        )
