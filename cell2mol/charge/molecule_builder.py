from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem
from cell2mol.elementdata import ElementData
from cell2mol.my_types import RDKitObject
import logging

if TYPE_CHECKING:
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.specie import Specie

logger = logging.getLogger(__name__)
elemdatabase = ElementData()

# Exclude carbanions: sulfonium ylides are genuine charge-separated species.
_YLIDE_ACCEPTORS = frozenset(("O", "S", "Se", "N"))


############################################################
# Charge repair and obligate charge separation
############################################################


def collapse_hypervalent_ylides(mol: RDKitObject):
    """Convert X(+)-Y(-) to X=Y when both atoms can support the neutral valences.

    Preserve obligate pairs such as N-oxides, nitro groups and phosphorus ylides.
    """
    rw_mol = Chem.RWMol(mol)
    fixed = False

    for bond in list(rw_mol.GetBonds()):
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue

        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        for center, partner in ((begin, end), (end, begin)):
            # Only ±1 pairs can become neutral after one bond promotion.
            if center.GetFormalCharge() != 1 or partner.GetFormalCharge() != -1:
                continue

            center_label, partner_label = center.GetSymbol(), partner.GetSymbol()
            if not _can_expand_octet(center_label):
                logger.debug(
                    "\tKeeping %s(+)-%s(-): %s cannot expand its octet",
                    center_label,
                    partner_label,
                    center_label,
                )
                continue
            if partner_label not in _YLIDE_ACCEPTORS:
                continue

            center_valence = _total_bond_order(center) + 1
            partner_valence = _total_bond_order(partner) + 1
            if center_valence not in _neutral_valences(center_label):
                continue
            if partner_valence not in _neutral_valences(partner_label):
                continue

            bond.SetBondType(Chem.BondType.DOUBLE)
            center.SetFormalCharge(0)
            partner.SetFormalCharge(0)
            fixed = True

            logger.debug(
                "\tCollapsed hypervalent ylide %s(+)-%s(-) to %s(=%s) "
                "(valences %d / %d)",
                center_label,
                partner_label,
                center_label,
                partner_label,
                center_valence,
                partner_valence,
            )
            break

    out = rw_mol.GetMol()
    if fixed:
        # See fix_zwitterions: promoting a bond invalidates the cached valences.
        out.UpdatePropertyCache(strict=False)
    return out, fixed


def shift_13_ylides(mol: RDKitObject):
    """Shift X(+)=A-Y(-) to X-A=Y when both ends can be neutralized.

    Require a neutral middle atom and ordinary bonds to preserve obligate pairs.
    """
    rw_mol = Chem.RWMol(mol)
    fixed = False

    for middle in rw_mol.GetAtoms():
        if middle.GetFormalCharge() != 0 or middle.GetIsAromatic():
            continue

        to_cation = to_anion = None
        for bond in middle.GetBonds():
            if bond.GetIsAromatic():
                continue
            other = bond.GetOtherAtom(middle)
            if (
                bond.GetBondType() == Chem.BondType.DOUBLE
                and other.GetFormalCharge() == 1
            ):
                to_cation = (bond, other)
            elif (
                bond.GetBondType() == Chem.BondType.SINGLE
                and other.GetFormalCharge() == -1
            ):
                to_anion = (bond, other)

        if to_cation is None or to_anion is None:
            continue

        double_bond, cation = to_cation
        single_bond, anion = to_anion
        if anion.GetSymbol() not in _YLIDE_ACCEPTORS:
            continue

        # Both endpoints must support their resulting neutral valences.
        cation_valence = _total_bond_order(cation) - 1
        anion_valence = _total_bond_order(anion) + 1
        if cation_valence not in _neutral_valences(cation.GetSymbol()):
            continue
        if anion_valence not in _neutral_valences(anion.GetSymbol()):
            continue

        double_bond.SetBondType(Chem.BondType.SINGLE)
        single_bond.SetBondType(Chem.BondType.DOUBLE)
        cation.SetFormalCharge(0)
        anion.SetFormalCharge(0)
        fixed = True

        logger.debug(
            "\tShifted 1,3 ylide %s(+)=%s-%s(-) to %s-%s=%s (valences %d / %d)",
            cation.GetSymbol(),
            middle.GetSymbol(),
            anion.GetSymbol(),
            cation.GetSymbol(),
            middle.GetSymbol(),
            anion.GetSymbol(),
            cation_valence,
            anion_valence,
        )

    out = rw_mol.GetMol()
    if fixed:
        # See fix_zwitterions: moving a bond invalidates the cached valences.
        out.UpdatePropertyCache(strict=False)
    return out, fixed


def obligate_charge_separation_atoms(mol: RDKitObject) -> set[int]:
    """Find adjacent opposite charges whose cation cannot expand its octet,
    such as nitro, N-oxide, diazo and azide pairs.
    """
    obligate: set[int] = set()

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        for center, partner in ((begin, end), (end, begin)):
            if center.GetFormalCharge() <= 0 or partner.GetFormalCharge() >= 0:
                continue
            if _can_expand_octet(center.GetSymbol()):
                # Octet-expandable cations have a neutral alternative, so separation is optional.
                continue
            obligate.add(center.GetIdx())
            obligate.add(partner.GetIdx())

    return obligate


def fix_zwitterions(mol: RDKitObject):
    """Neutralize adjacent opposite charges by reducing a double bond to single.

    Return (molecule, changed).
    """
    rw_mol = Chem.RWMol(mol)
    fixed = False

    for atom in rw_mol.GetAtoms():
        fcharge = atom.GetFormalCharge()

        if fcharge > 0:
            atom_idx = atom.GetIdx()
            atom_label = atom.GetSymbol()
            neighbors = atom.GetNeighbors()

            logger.debug(
                "A positive atom: %s (idx=%d, charge=%d)",
                atom_label,
                atom_idx,
                fcharge,
            )

            # Skip common stable zwitterionic groups (Nitro and Nitrate)
            neighbor_labels = [n.GetSymbol() for n in neighbors]
            is_nitro_nitrate = (
                atom_label == "N"
                and len(neighbors) == 3
                and neighbor_labels.count("O") in [2, 3]
            )
            if is_nitro_nitrate:
                logger.debug("\tSkipping nitro or nitrate group.")
                continue

            # Skip Carbonyl oxygens
            if (
                atom_label == "O"
                and len(neighbors) == 1
                and neighbor_labels.count("C") == 1
            ):
                logger.debug("\tSkipping carbonyl group.")
                continue

            for neighbor in neighbors:
                n_fcharge = neighbor.GetFormalCharge()
                n_label = neighbor.GetSymbol()
                neighbor_idx = neighbor.GetIdx()

                # Skip if labels are identical (e.g., O+-O- peroxide artifacts)
                if n_label == atom_label:
                    logger.debug("\tSkipping neighbor with same label: %s", n_label)
                    continue

                if n_fcharge < 0:
                    # Skip specific stable pairs like N-oxide (N+-O-)
                    if atom_label == "N" and n_label == "O":
                        logger.debug("\tSkipping N+ with O- neighbor.")
                        continue

                    # Skip Boron with full coordination (B- with 4 neighbors)
                    if n_label == "B" and len(neighbor.GetNeighbors()) == 4:
                        logger.debug("\tSkipping fully coordinated Boron (B-).")
                        continue

                    bond = rw_mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)

                    if bond and bond.GetBondTypeAsDouble() == 2.0:
                        fixed = True
                        logger.debug(
                            "\tFixing adjacent charges between %s and %s",
                            atom_label,
                            n_label,
                        )

                        diff = min(fcharge, abs(n_fcharge))
                        atom.SetFormalCharge(fcharge - diff)
                        neighbor.SetFormalCharge(n_fcharge + diff)

                        logger.debug(
                            "\tAdjusted charges: atom %d (%d -> %d), neighbor %d (%d -> %d)",
                            atom_idx,
                            fcharge,
                            atom.GetFormalCharge(),
                            neighbor_idx,
                            n_fcharge,
                            neighbor.GetFormalCharge(),
                        )

                        valence_electrons = elemdatabase.valenceelectrons[n_label]
                        num_radicals = (
                            valence_electrons
                            - neighbor.GetFormalCharge()
                            - neighbor.GetDegree()
                        )
                        neighbor.SetNumRadicalElectrons(max(0, num_radicals))

                        logger.debug(
                            "\tSet radical electrons for neighbor %d to %d",
                            neighbor_idx,
                            num_radicals,
                        )

                        logger.debug(
                            "\tChanging bond between %d and %d from DOUBLE to SINGLE",
                            atom_idx,
                            neighbor_idx,
                        )
                        bond.SetBondType(Chem.BondType.SINGLE)
                    else:
                        if bond:
                            logger.debug(
                                "\tSkipping fix: bond between %d and %d is %s (not DOUBLE)",
                                atom_idx,
                                neighbor_idx,
                                bond.GetBondType(),
                            )
                        continue

    out = rw_mol.GetMol()
    if fixed:
        # Refresh cached valences after changing bonds and charges.
        out.UpdatePropertyCache(strict=False)
    return out, fixed


############################################################
# Valence helpers for charge repair
############################################################


def _can_expand_octet(symbol: str) -> bool:
    """Return whether a p-block element from period 3 onward can expand its octet."""
    return (
        elemdatabase.elementblock[symbol] == "p"
        and elemdatabase.elementperiod[symbol] >= 3
    )


def _neutral_valences(symbol: str) -> tuple[int, ...]:
    """Return neutral bond-order sums from valence electrons and octet-expansion rules."""
    base = 8 - elemdatabase.valenceelectrons[symbol]
    if base < 1:
        return ()
    if _can_expand_octet(symbol):
        return tuple(base + 2 * k for k in range(3))
    return (base,)


def _total_bond_order(atom) -> int:
    return int(sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()))


############################################################
# Molecule finalization and SMILES generation
############################################################


def finalize_specie_mol(mol: RDKitObject):
    """Sanitize the specie molecule and generate conformer-independent SMILES.

    Return (molecule, SMILES, False); the final field is retained for compatibility.
    """
    obj = Chem.Mol(mol)
    fixed = False
    obj.RemoveAllConformers()
    # Remove 3D-derived chirality to keep SMILES conformer-independent.
    Chem.RemoveStereochemistry(obj)
    Chem.SanitizeMol(
        obj,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        catchErrors=True,
    )
    Chem.DetectBondStereochemistry(obj, -1)
    Chem.AssignStereochemistry(obj, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(obj, -1)
    return obj, Chem.MolToSmiles(obj), fixed


def generate_tmc_rdkit_obj_smiles(mol: "Molecule"):
    metals = mol.metals or []
    all_metals_indices = [met.get_parent_index("molecule") for met in metals]
    logger.debug(
        "Found metals %s with indices %s",
        [met.atom_site_label for met in metals]
        if getattr(metals[0], "atom_site_label", None) is not None
        else [met.label for met in metals],
        all_metals_indices,
    )
    temp_mol = Chem.RWMol()

    for met in metals:
        a = Chem.Atom(met.label)
        a.SetFormalCharge(int(met.charge or 0))  # Assign the metal oxidation state
        met_mol_idx = met.get_parent_index("molecule")
        assert met_mol_idx is not None
        a.SetIntProp("__mol_idx", met_mol_idx)
        if met.atom_site_label is not None:
            a.SetProp("__atom_site_label", met.atom_site_label)

        idx = temp_mol.AddAtom(a)
        logger.debug(
            "Add metal atom %s to rdkit molecule object", Chem.MolToSmiles(temp_mol)
        )

    for lig in mol.ligands or []:
        assert lig.rdkit_obj is not None
        for lig_atom, a in zip(lig.atoms or [], lig.rdkit_obj.GetAtoms()):
            a.SetFormalCharge(int(lig_atom.charge or 0))
            lig_atom_mol_idx = lig_atom.get_parent_index("molecule")
            assert lig_atom_mol_idx is not None
            a.SetIntProp("__mol_idx", lig_atom_mol_idx)

            if lig_atom.atom_site_label is not None:
                a.SetProp("__atom_site_label", lig_atom.atom_site_label)

        logger.debug(
            "Add ligand with %s (Q=%s) to rdkit molecule object",
            lig.formula,
            lig.totcharge,
        )

        temp_mol = Chem.CombineMols(temp_mol, lig.rdkit_obj)

    new_mol = Chem.RWMol(temp_mol)

    if mol.natoms != new_mol.GetNumAtoms():
        raise ValueError(
            "Number of atoms in cell2mol and rkdit molecule object disagrees"
        )

    new_order = []
    for idx in range(mol.natoms):
        mol_idx = [
            a.GetIdx() for a in new_mol.GetAtoms() if a.GetIntProp("__mol_idx") == idx
        ][0]
        new_order.append(mol_idx)

    new_mol = Chem.RenumberAtoms(new_mol, new_order)
    new_mol = Chem.RWMol(new_mol)

    for met in metals:
        met_idx = met.get_parent_index("molecule")
        assert met_idx is not None

        coord_sphere_atoms = met.coord_sphere_atoms or []
        coordinating_atoms_labels = [atom.label for atom in coord_sphere_atoms]

        coordinating_atoms_indices = [
            atom.get_parent_index("molecule") for atom in coord_sphere_atoms
        ]
        logger.debug(
            "%s%s coordinates to %s %s",
            met.label,
            f" ({met.atom_site_label})" if met.atom_site_label else "",
            coordinating_atoms_labels,
            coordinating_atoms_indices,
        )

        for idx in coordinating_atoms_indices:
            assert idx is not None
            if new_mol.GetBondBetweenAtoms(idx, met_idx):
                logger.debug(
                    "Already added %s %s",
                    idx,
                    new_mol.GetBondBetweenAtoms(idx, met_idx).GetBondType(),
                )
            elif idx in all_metals_indices:
                new_mol.AddBond(idx, met_idx, Chem.BondType.UNSPECIFIED)
            else:
                new_mol.AddBond(idx, met_idx, Chem.BondType.DATIVE)

    smiles = Chem.MolToSmiles(new_mol.GetMol())
    logger.info("TMC_SMILES: %s", smiles)

    tmc_rdkit_obj = Chem.MolFromSmiles(smiles)  # Hydrogens are removed

    try:
        Chem.SanitizeMol(tmc_rdkit_obj)
        return new_mol.GetMol(), Chem.MolToSmiles(tmc_rdkit_obj)
    except Exception:
        try:
            logger.info("TMC_SMILES: SanitizeMol with extra keywords")
            Chem.SanitizeMol(
                tmc_rdkit_obj,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                catchErrors=True,
            )
            return new_mol.GetMol(), Chem.MolToSmiles(tmc_rdkit_obj)
        except Exception:
            return new_mol.GetMol(), smiles

    if mol.is_haptic:
        logger.info("TMC_SMILES: %s %s", mol.is_haptic, mol.haptic_type)
        tmc_rdkit_obj = Chem.rdmolops.DativeBondsToHaptic(tmc_rdkit_obj)
        logger.info("TMC_SMILES: %s", Chem.MolToSmiles(tmc_rdkit_obj))

    tmc_smiles = Chem.MolToSmiles(tmc_rdkit_obj)
    return tmc_rdkit_obj, tmc_smiles
    return new_mol, tmc_rdkit_obj


############################################################
# Specie and metal bond assembly
############################################################


def create_bonds_specie(specie: "Specie", rdkit_obj: RDKitObject | None = None):
    """Populate specie.atoms with one Bond per bond in the deprotonated RDKit molecule."""
    from cell2mol.classes import Bond

    if rdkit_obj is None:
        rdkit_obj = specie.rdkit_obj
    assert rdkit_obj is not None

    n_atoms = specie.natoms
    n_atoms_rdkit = rdkit_obj.GetNumAtoms()
    if n_atoms != n_atoms_rdkit:
        logger.error(
            "create_bonds_specie: %s atom-count mismatch (specie %d vs rdkit %d); "
            "expected the deprotonated specie rdkit_obj",
            specie.subtype,
            n_atoms,
            n_atoms_rdkit,
        )
        return False

    for idx, rdkit_atom in enumerate(rdkit_obj.GetAtoms()):
        for b in rdkit_atom.GetBonds():
            a1 = b.GetBeginAtomIdx()
            a2 = b.GetEndAtomIdx()
            end_label = specie.atoms[a2].label
            end_symbol = rdkit_obj.GetAtomWithIdx(a2).GetSymbol()
            # Skip if the end-atom element disagrees (allow D<->H for deuterium).
            if end_label != end_symbol and not (end_label == "D" and end_symbol == "H"):
                logger.debug("Bond end-atom mismatch %s vs %s", end_label, end_symbol)
                continue
            start, end = (a2, a1) if a2 == idx else (a1, a2)
            specie.atoms[idx].add_bond(
                Bond.from_positional(
                    specie.atoms[start], specie.atoms[end], b.GetBondTypeAsDouble()
                )
            )

        if not specie.atoms[idx].bonds and specie.natoms != 1:
            logger.error(
                "NO BONDS for %s with %s RDKit object index %d.",
                specie.atoms[idx].label,
                specie.subtype,
                idx,
            )
            return False

    return True


def create_metal_ligand_bonds(mol: "Molecule"):
    # Adds Metal-Ligand Bonds, with a zero order:
    from cell2mol.classes import Bond

    assert mol.madjmat is not None
    if (
        mol.is_metal_complex
        or mol.has_s_block_metal
        or mol.is_post_transition_metal_complex
    ):
        for lig in mol.ligands or []:
            for at in lig.atoms or []:
                count = 0
                index_1 = at.get_parent_index("molecule")
                assert index_1 is not None
                for met in mol.metals or []:
                    index_2 = met.get_parent_index("molecule")
                    assert index_2 is not None
                    isconnected = mol.madjmat[index_1, index_2] == 1
                    if isconnected:
                        if index_1 < index_2:
                            bond_startatom = at
                            bond_endatom = met
                        else:
                            bond_startatom = met
                            bond_endatom = at
                        newbond = Bond.from_positional(bond_startatom, bond_endatom, 1)
                        at.add_bond(newbond)
                        met.add_bond(newbond)
                        count += 1
                if count != at.mconnec:
                    logger.error(
                        "Error creating bonds for atom: \n%s\n of ligand: \n%s\n",
                        at,
                        lig,
                    )


def create_metal_metal_bonds(mol: "Molecule"):
    from cell2mol.classes import Bond

    # Adds Metal-Metal Bonds, with a zero order:
    assert mol.madjmat is not None
    if (
        mol.is_metal_complex
        or mol.has_s_block_metal
        or mol.is_post_transition_metal_complex
    ):
        metals = mol.metals or []
        if len(metals) > 1:
            logger.debug("Creating Metal-Metal Bonds for molecule %s", mol.formula)
            for idx, met1 in enumerate(metals):
                index_1 = met1.get_parent_index("molecule")
                assert index_1 is not None
                for jdx, met2 in enumerate(metals):
                    if idx <= jdx:
                        continue
                    index_2 = met2.get_parent_index("molecule")
                    assert index_2 is not None
                    isconnected = mol.madjmat[index_1, index_2] == 1
                    if isconnected:
                        if index_1 < index_2:
                            bond_startatom = met1
                            bond_endatom = met2
                        else:
                            bond_startatom = met2
                            bond_endatom = met1
                        newbond = Bond.from_positional(bond_startatom, bond_endatom, 0)
                        met1.add_bond(newbond)
                        met2.add_bond(newbond)
