import numpy as np
from typing import Any, Tuple
from cell2mol.classes.protonation import Protonation
from cell2mol.my_types import RDKitObject
from cell2mol.utils.pydantic import BaseModel
from pydantic import Field, model_validator
from typing_extensions import deprecated
import logging

logger = logging.getLogger(__name__)

# Backward-compat: the charge fields were renamed uncorr_* -> protonated_* and
# corr_* -> specie_*. This maps the legacy names to the current ones so
# previously-serialized ChargeStates (JSON or pickle) still load.
_LEGACY_FIELD_NAMES = {
    "uncorr_total_charge": "protonated_total_charge",
    "uncorr_atom_charges": "protonated_atom_charges",
    "uncorr_abstotal": "protonated_abstotal",
    "uncorr_abs_atcharge": "protonated_abs_atcharge",
    "uncorr_zwitt": "protonated_zwitt",
    "corr_total_charge": "specie_total_charge",
    "corr_atom_charges": "specie_atom_charges",
    "corr_abstotal": "specie_abstotal",
    "corr_abs_atcharge": "specie_abs_atcharge",
    "corr_zwitt": "specie_zwitt",
    "corr_rdkit_obj": "specie_rdkit_obj",
    "corr_smiles": "specie_smiles",
}


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
    protonated_total_charge: int
    protonated_atom_charges: list[int]
    rdkit_obj: RDKitObject
    smiles: str
    charge_tried: int
    allow: bool
    protonation: Protonation = Field(...)

    # Computed attributes with proper defaults
    protonated_abstotal: int | None = None
    protonated_abs_atcharge: int | None = None
    protonated_zwitt: bool | None = None
    coincide: bool | None = None

    # Copied from protonation with proper defaults
    site_proton_counts: list[int] | None = None
    ligand_donor_electrons: list[int] | None = None
    n_protons_added: int | None = None

    # Initialized attributes with defaults
    specie_total_charge: int = Field(default=0)
    specie_atom_charges: list[int] = Field(default_factory=list)

    # Final computed attributes with proper defaults
    specie_abstotal: int | None = None
    specie_abs_atcharge: int | None = None
    specie_zwitt: bool | None = None

    # The actual (deprotonated) specie SMILES / mol -- as opposed to `smiles`,
    # which is the protonated one. Computed at construction by _build_specie_mol
    # from this state's own rdkit_obj + specie_atom_charges (no parent needed).
    specie_rdkit_obj: RDKitObject | None = None
    specie_smiles: str | None = None

    # Frozen fields
    type: str = Field(default="charge_state", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_field_names(cls, data: Any) -> Any:
        # Older saved JSON cells use the pre-rename keys (uncorr_*/corr_*).
        if isinstance(data, dict):
            for old, new in _LEGACY_FIELD_NAMES.items():
                if old in data and new not in data:
                    data[new] = data.pop(old)
        return data

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Older pickled cells (.cell) restore state directly, bypassing the
        # validator above -- remap the legacy field names in the pickled
        # __dict__ / fields-set before pydantic restores them.
        for key in ("__dict__", "__pydantic_fields_set__"):
            container = state.get(key)
            if container is None:
                continue
            for old, new in _LEGACY_FIELD_NAMES.items():
                if old in container:
                    if isinstance(container, dict):
                        if new not in container:
                            container[new] = container.pop(old)
                    else:  # set (fields-set)
                        container.discard(old)
                        container.add(new)
        super().__setstate__(state)

    def model_post_init(self, __context: Any) -> None:
        # Compute initial derived values
        (
            self.protonated_abstotal,
            self.protonated_abs_atcharge,
            self.protonated_zwitt,
        ) = eval_chargelist(self.protonated_atom_charges)

        # Set coincide flag
        self.coincide = self.protonated_total_charge == self.charge_tried

        # Copy attributes from protonation
        self.site_proton_counts = self.protonation.site_proton_counts
        self.ligand_donor_electrons = self.protonation.ligand_donor_electrons
        self.n_protons_added = self.protonation.n_protons_added

        logger.debug(
            "Initializing ChargeState | Status: %s | SMILES: %s | Protonated Charge: %d | Charge Tried: %d | N Protons Added: %d",
            self.status,
            self.smiles,
            self.protonated_total_charge,
            self.charge_tried,
            self.n_protons_added,
        )

        # Specie (deprotonated) charges: build a fresh list and ASSIGN (never
        # append to the field) -- a reload re-runs this, and append would double it.
        if len(self.site_proton_counts) > 0:
            specie_charges = []
            # Iterate over the original atoms only (added H come after them).
            for idx, n_add in enumerate(self.site_proton_counts):
                if n_add > 0:
                    # last term corrects for a charge assigned to the added atom
                    specie_charges.append(
                        self.protonated_atom_charges[idx]
                        - self.site_proton_counts[idx]
                        + self.ligand_donor_electrons[idx]
                    )
                else:
                    specie_charges.append(self.protonated_atom_charges[idx])
            self.specie_atom_charges = specie_charges
            self.specie_total_charge = int(np.sum(specie_charges))
        else:
            self.specie_total_charge = self.protonated_total_charge
            self.specie_atom_charges = self.protonated_atom_charges.copy()

        self.specie_abstotal, self.specie_abs_atcharge, self.specie_zwitt = (
            eval_chargelist(self.specie_atom_charges)
        )

        # Deprotonated specie SMILES from this state's own data (rdkit_obj +
        # specie_atom_charges), no parent mutation -- available in reference mode.
        # Always finalize (even nH==0): the raw protonated SMILES isn't always the
        # canonical specie SMILES (aromaticity/charge/stereo). Skip if set (reload).
        if self.status and not self.specie_smiles:
            try:
                obj, self.specie_smiles, fixed = self._build_specie_mol()
                self.specie_rdkit_obj = obj
                if fixed:
                    # A zwitterion fix during finalize moved formal charges
                    # between atoms; resync the specie charge fields to the
                    # corrected mol.
                    self.specie_atom_charges = [
                        int(a.GetFormalCharge()) for a in obj.GetAtoms()
                    ]
                    self.specie_total_charge = int(np.sum(self.specie_atom_charges))
                    (
                        self.specie_abstotal,
                        self.specie_abs_atcharge,
                        self.specie_zwitt,
                    ) = eval_chargelist(self.specie_atom_charges)
            except Exception as e:
                logger.debug(
                    "Could not build specie mol (protonated SMILES %s): %s",
                    self.smiles,
                    e,
                )

    def _build_specie_mol(self) -> "tuple[Any, str, bool]":
        """Deprotonated specie mol/SMILES from this state's protonated rdkit_obj.

        The protonated ``rdkit_obj`` lists the specie's original atoms first and
        the added protons last, so dropping the trailing ``n_protons_added``
        atoms and stamping ``specie_atom_charges`` onto the originals recovers
        the actual specie, finalized via the shared ``finalize_specie_mol``.
        Pure -- nothing on the parent is touched.

        Returns (mol, smiles, zwitterion_fixed); when ``zwitterion_fixed`` is
        True the caller should resync the specie charge fields to ``mol``.
        """
        from rdkit import Chem
        from cell2mol.charge.smiles_handler import finalize_specie_mol

        n_orig = len(self.specie_atom_charges)
        rw = Chem.RWMol(self.rdkit_obj)
        for idx in range(rw.GetNumAtoms() - 1, n_orig - 1, -1):
            rw.RemoveAtom(idx)
        for idx, q in enumerate(self.specie_atom_charges):
            atom = rw.GetAtomWithIdx(idx)
            atom.SetFormalCharge(int(q))
            atom.SetNoImplicit(True)
        return finalize_specie_mol(rw.GetMol())

    def summary(self) -> str:
        """Compact one-liner for scanning many states (e.g. in logs).

        Shows the discriminating fields: specie charge, validity, protons
        added, zwitterion flag, and a truncated SMILES. Not used by repr/str --
        call it explicitly (e.g. `", ".join(c.summary() for c in states)`).
        """
        smi = self.specie_smiles or self.smiles or ""
        if len(smi) > 48:
            smi = smi[:45] + "..."
        tags = " zwitt" if self.specie_zwitt else ""
        nH = self.n_protons_added if self.n_protons_added is not None else "?"
        return (
            f"<ChargeState q={self.specie_total_charge:+d} "
            f"{'OK' if self.status else 'FAIL'} nH={nH}{tags} {smi!r}>"
        )

    def __repr__(self) -> str:
        """Full multi-line view -- so print(charge_state) AND print(states)
        both show complete detail. Use summary() for a compact one-liner."""
        lines = [
            "------------- Cell2mol Charge State ---------------",
            f" Status                          = {self.status}",
            f" Charge Tried                    = {self.charge_tried}",
            f" Number of Protons Added         = {self.n_protons_added}",
            f" Protonated Total Charge         = {self.protonated_total_charge}",
            f" Smiles (protonated)             = {self.smiles}",
        ]
        if self.ligand_donor_electrons and sum(self.ligand_donor_electrons) > 0:
            lines.append(
                f" Ligand Donor Electrons          = {sum(self.ligand_donor_electrons)}"
            )
        lines += [
            f" Specie Total Charge             = {self.specie_total_charge}",
            f" Specie Absolute Total Charge    = {self.specie_abs_atcharge}",
            f" Smiles (specie)                 = {self.specie_smiles}",
        ]
        if self.specie_zwitt:
            lines.append(f" Specie Is Zwitterion?           = {self.specie_zwitt}")

        lines.append("---------------------------------------------------")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    @deprecated("Use charge_state() with the keyword arguments instead.")
    def from_positional(
        cls,
        status: bool,
        protonated_total_charge: int,
        protonated_atom_charges: list[int],
        rdkit_obj: object,
        smiles: str,
        charge_tried: int,
        allow: bool,
        protonation: Protonation,
    ) -> "ChargeState":
        return cls(
            status=status,
            protonated_total_charge=protonated_total_charge,
            protonated_atom_charges=protonated_atom_charges,
            rdkit_obj=rdkit_obj,
            smiles=smiles,
            charge_tried=charge_tried,
            allow=allow,
            protonation=protonation,
        )
