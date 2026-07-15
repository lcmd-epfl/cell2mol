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
    site_proton_counts: list[int] | None = None
    ligand_donor_electrons: list[int] | None = None
    n_protons_added: int | None = None

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
        self.site_proton_counts = self.protonation.site_proton_counts
        self.ligand_donor_electrons = self.protonation.ligand_donor_electrons
        self.n_protons_added = self.protonation.n_protons_added

        logger.debug(
            "Initializing ChargeState | Status: %s | SMILES: %s | Uncorr Charge: %d | Charge Tried: %d | N Protons Added: %d",
            self.status,
            self.smiles,
            self.uncorr_total_charge,
            self.charge_tried,
            self.n_protons_added,
        )

        # Corrects the Charge of atoms with addedH
        if len(self.site_proton_counts) > 0:
            # Iterates over the original number of ligand atoms, thus without the added H
            for idx, n_add in enumerate(self.site_proton_counts):
                if n_add > 0:
                    corrected = (
                        self.uncorr_atom_charges[idx]
                        - self.site_proton_counts[idx]
                        + self.ligand_donor_electrons[idx]
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
        to_print += f" Number of Protons Added         = {self.n_protons_added}\n"
        if self.ligand_donor_electrons and sum(self.ligand_donor_electrons) > 0:
            to_print += f" Ligand Donor Electrons          = {sum(self.ligand_donor_electrons)}\n"
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
        protonation: Protonation,
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
