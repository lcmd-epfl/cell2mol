import numpy as np
from typing import Any
from cell2mol.classes.protonation import Protonation
from cell2mol.my_types import RDKitObject
from cell2mol.utils.pydantic import BaseModel
from pydantic import Field, model_validator
from typing_extensions import deprecated
import logging

logger = logging.getLogger(__name__)

# Map legacy JSON and pickle fields to their current names.
_LEGACY_FIELD_NAMES = {
    "charge_tried": "protonated_target_charge",
    "coincide": "matches_target_charge",
    "allow": "allow_charged_fragments",
    "status": "is_valid",
    "rdkit_obj": "protonated_rdkit_obj",
    "smiles": "protonated_smiles",
    "protonated_abstotal": "protonated_abs_total_charge",
    "protonated_abs_atcharge": "protonated_sum_abs_atom_charges",
    "protonated_zwitt": "protonated_is_zwitterionic",
    "specie_abstotal": "specie_abs_total_charge",
    "specie_abs_atcharge": "specie_sum_abs_atom_charges",
    "specie_zwitt": "specie_is_zwitterionic",
    "uncorr_total_charge": "protonated_total_charge",
    "uncorr_atom_charges": "protonated_atom_charges",
    "uncorr_abstotal": "protonated_abs_total_charge",
    "uncorr_abs_atcharge": "protonated_sum_abs_atom_charges",
    "uncorr_zwitt": "protonated_is_zwitterionic",
    "corr_total_charge": "specie_total_charge",
    "corr_atom_charges": "specie_atom_charges",
    "corr_abstotal": "specie_abs_total_charge",
    "corr_abs_atcharge": "specie_sum_abs_atom_charges",
    "corr_zwitt": "specie_is_zwitterionic",
    "corr_rdkit_obj": "specie_rdkit_obj",
    "corr_smiles": "specie_smiles",
}


def eval_chargelist(atom_charges: list[int]) -> tuple[int, int, bool]:
    """Return absolute net charge, summed absolute charges, and zwitterion flag."""
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
    """Store charges and molecular forms for the original and protonated specie."""

    model_config = {"arbitrary_types_allowed": True}

    # Required inputs
    is_valid: bool
    protonated_total_charge: int
    protonated_atom_charges: list[int]
    protonated_rdkit_obj: RDKitObject
    protonated_smiles: str
    protonated_target_charge: int
    allow_charged_fragments: bool
    protonation: Protonation = Field(...)

    # Protonated charge metrics
    protonated_abs_total_charge: int | None = None
    protonated_sum_abs_atom_charges: int | None = None
    protonated_is_zwitterionic: bool | None = None
    matches_target_charge: bool | None = None

    # Protonation data
    site_proton_counts: list[int] | None = None
    ligand_donor_electrons: list[int] | None = None
    n_protons_added: int | None = None

    # Original specie charges
    specie_total_charge: int = Field(default=0)
    specie_atom_charges: list[int] = Field(default_factory=list)

    # Specie charge metrics
    specie_abs_total_charge: int | None = None
    specie_sum_abs_atom_charges: int | None = None
    specie_is_zwitterionic: bool | None = None

    # Original specie molecular forms restored from the protonated state.
    specie_rdkit_obj: RDKitObject | None = None
    specie_smiles: str | None = None

    type: str = Field(default="charge_state", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_field_names(cls, data: Any) -> Any:
        """Rename legacy fields before model validation."""
        if isinstance(data, dict):
            for old, new in _LEGACY_FIELD_NAMES.items():
                if old in data and new not in data:
                    data[new] = data.pop(old)
        return data

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rename legacy pickle fields, which bypass model validation."""
        for key in ("__dict__", "__pydantic_fields_set__"):
            container = state.get(key)
            if container is None:
                continue
            for old, new in _LEGACY_FIELD_NAMES.items():
                if old in container:
                    if isinstance(container, dict):
                        if new not in container:
                            container[new] = container.pop(old)
                    else:  # Pydantic fields set
                        container.discard(old)
                        container.add(new)
        super().__setstate__(state)

    def model_post_init(self, __context: Any) -> None:
        """Compute charge metrics and restore the original specie."""
        (
            self.protonated_abs_total_charge,
            self.protonated_sum_abs_atom_charges,
            self.protonated_is_zwitterionic,
        ) = eval_chargelist(self.protonated_atom_charges)

        self.matches_target_charge = (
            self.protonated_total_charge == self.protonated_target_charge
        )

        self.site_proton_counts = self.protonation.site_proton_counts
        self.ligand_donor_electrons = self.protonation.ligand_donor_electrons
        self.n_protons_added = self.protonation.n_protons_added

        self._log_initialization()

        # Replace the charge list to avoid duplicates on reload.
        if len(self.site_proton_counts) > 0:
            specie_charges = []
            # Original atoms precede added hydrogens.
            for idx, n_add in enumerate(self.site_proton_counts):
                if n_add > 0:
                    # Undo added protons and account for donated electrons.
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

        (
            self.specie_abs_total_charge,
            self.specie_sum_abs_atom_charges,
            self.specie_is_zwitterionic,
        ) = eval_chargelist(self.specie_atom_charges)

        # Finalize even without added H; reuse existing SMILES on reload.
        if self.is_valid and not self.specie_smiles:
            try:
                obj, self.specie_smiles, fixed = self._build_specie_mol()
                self.specie_rdkit_obj = obj
                if fixed:
                    # Sync charge metrics if finalization reports a repair.
                    self.specie_atom_charges = [
                        int(a.GetFormalCharge()) for a in obj.GetAtoms()
                    ]
                    self.specie_total_charge = int(np.sum(self.specie_atom_charges))
                    (
                        self.specie_abs_total_charge,
                        self.specie_sum_abs_atom_charges,
                        self.specie_is_zwitterionic,
                    ) = eval_chargelist(self.specie_atom_charges)
            except Exception as e:
                logger.debug(
                    "Could not build specie mol (protonated SMILES %s): %s",
                    self.protonated_smiles,
                    e,
                )

    def _log_initialization(self) -> None:
        """Log protonated sites and charges at debug level."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        parent = getattr(self.protonation, "parent", None)
        parent_atoms = getattr(parent, "atoms", None) or []
        donor_electrons = self.ligand_donor_electrons or []
        protonated_sites = []
        for idx, n_added in enumerate(self.site_proton_counts or []):
            if not n_added:
                continue
            # Include donated electrons when nonzero.
            donated = donor_electrons[idx] if idx < len(donor_electrons) else 0
            donated_note = f"+{donated}e-" if donated else ""
            if idx < len(parent_atoms):
                atom = parent_atoms[idx]
                label = getattr(atom, "label", "?")
                site_label = getattr(atom, "atom_site_label", None) or label
                protonated_sites.append(
                    f"{label}({site_label})+{n_added}H{donated_note}"
                )
            else:
                protonated_sites.append(f"?+{n_added}H{donated_note}")

        logger.debug(
            "Initializing ChargeState | Valid: %s | SMILES: %s | Protonated Charge: %d | Protonated Target Charge: %d | N Protons Added: %d | Protonated sites: %s",
            self.is_valid,
            self.protonated_smiles,
            self.protonated_total_charge,
            self.protonated_target_charge,
            self.n_protons_added,
            ", ".join(protonated_sites) if protonated_sites else "none",
        )

    def _build_specie_mol(self) -> tuple[RDKitObject, str, bool]:
        """Restore the original specie by removing added H and applying its charges.

        Return (mol, SMILES, False); the flag is retained for compatibility.
        The parent and protonated molecule remain unchanged.
        """
        from rdkit import Chem
        from cell2mol.charge.molecule_builder import finalize_specie_mol

        n_orig = len(self.specie_atom_charges)
        rw = Chem.RWMol(self.protonated_rdkit_obj)
        for idx in range(rw.GetNumAtoms() - 1, n_orig - 1, -1):
            rw.RemoveAtom(idx)
        for idx, q in enumerate(self.specie_atom_charges):
            atom = rw.GetAtomWithIdx(idx)
            atom.SetFormalCharge(int(q))
            atom.SetNoImplicit(True)
        return finalize_specie_mol(rw.GetMol())

    def summary(self) -> str:
        """Return charge, status, added H, zwitterion flag, and shortened SMILES."""
        smi = self.specie_smiles or self.protonated_smiles or ""
        if len(smi) > 48:
            smi = smi[:45] + "..."
        tags = " zwitt" if self.specie_is_zwitterionic else ""
        nH = self.n_protons_added if self.n_protons_added is not None else "?"
        return (
            f"<ChargeState q={self.specie_total_charge:+d} "
            f"{'OK' if self.is_valid else 'FAIL'} nH={nH}{tags} {smi!r}>"
        )

    def __repr__(self) -> str:
        """Return full charge-state details; use summary() for a compact view."""
        lines = [
            "------------- Cell2mol Charge State ---------------",
            f" Valid                          = {self.is_valid}",
            f" Protonated Target Charge                    = {self.protonated_target_charge}",
            f" Number of Protons Added         = {self.n_protons_added}",
            f" Protonated Total Charge         = {self.protonated_total_charge}",
            f" Smiles (protonated)             = {self.protonated_smiles}",
        ]
        if self.ligand_donor_electrons and sum(self.ligand_donor_electrons) > 0:
            lines.append(
                f" Ligand Donor Electrons          = {sum(self.ligand_donor_electrons)}"
            )
        lines += [
            f" Specie Total Charge             = {self.specie_total_charge}",
            f" Specie Sum Abs Atom Charges    = {self.specie_sum_abs_atom_charges}",
            f" Smiles (specie)                 = {self.specie_smiles}",
        ]
        if self.specie_is_zwitterionic:
            lines.append(
                f" Specie Is Zwitterion?           = {self.specie_is_zwitterionic}"
            )

        lines.append("---------------------------------------------------")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    @deprecated("Use ChargeState() with keyword arguments instead.")
    def from_positional(
        cls,
        is_valid: bool,
        protonated_total_charge: int,
        protonated_atom_charges: list[int],
        protonated_rdkit_obj: RDKitObject,
        protonated_smiles: str,
        protonated_target_charge: int,
        allow_charged_fragments: bool,
        protonation: Protonation,
    ) -> "ChargeState":
        return cls(
            is_valid=is_valid,
            protonated_total_charge=protonated_total_charge,
            protonated_atom_charges=protonated_atom_charges,
            protonated_rdkit_obj=protonated_rdkit_obj,
            protonated_smiles=protonated_smiles,
            protonated_target_charge=protonated_target_charge,
            allow_charged_fragments=allow_charged_fragments,
            protonation=protonation,
        )
