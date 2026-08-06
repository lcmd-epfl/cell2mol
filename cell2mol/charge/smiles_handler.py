from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem
from cell2mol.elementdata import ElementData
from cell2mol.my_types import RDKitObject
import logging

if TYPE_CHECKING:
    from cell2mol.classes.molecule import Molecule

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def fix_zwitterions(mol):
    """
    Fixes zwitterionic artifacts by adjusting formal charges between adjacent atoms
    with opposite charges in an RDKit molecule object.

    Args:
        mol: Input RDKit molecule.

    Returns:
        tuple: (Corrected Chem.Mol, bool indicating if changes were made).
    """
    rw_mol = Chem.RWMol(mol)
    fixed = False

    for atom in rw_mol.GetAtoms():
        fcharge = atom.GetFormalCharge()

        # Target positively charged atoms to find adjacent negative partners
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

            # Skip Carbonyl oxygens (usually shouldn't have + charge, but safety check)
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

                    # Minimize zwitterion only if a bond exists and is currently a DOUBLE bond
                    # (This logic implies converting a charged double bond to a neutral single bond)
                    if bond and bond.GetBondTypeAsDouble() == 2.0:
                        fixed = True
                        logger.debug(
                            "\tFixing adjacent charges between %s and %s",
                            atom_label,
                            n_label,
                        )

                        # Determine the charge magnitude to neutralize
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

                        # Recalculate and estimate radical electrons for the neighbor
                        # Formula: Valence - FormalCharge - Total Bonds
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

                        # Downgrade bond from DOUBLE to SINGLE
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

    return rw_mol.GetMol(), fixed


def finalize_specie_mol(mol):
    """Zwitterion-fix, sanitize and canonicalize a specie RDKit mol.

    Used by `ChargeState._build_specie_mol` to finalize the deprotonated specie
    mol. Any 3D conformer is dropped so the structure-based stereo perception
    below is a no-op and the SMILES is conformer-independent.

    Returns:
        tuple: (Chem.Mol, canonical SMILES, bool whether a zwitterion was fixed).
    """
    obj, fixed = fix_zwitterions(mol)
    obj.RemoveAllConformers()
    # Strip any chiral tags inherited from the 3D-perceived rdkit_obj so the
    # SMILES is stereo-free and conformer-independent.
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
        # lig_atom : atom object from cell2mol
        # a : atom object from rdkit object
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
            # print(idx, new_mol.GetBondBetweenAtoms(idx, met_idx))
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
            # print(idx, new_mol.GetBondBetweenAtoms(idx, met_idx).GetBondType())

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


def create_bonds_specie(specie, rdkit_obj: RDKitObject | None = None):
    """Build cell2mol Bond objects on ``specie.atoms`` from an RDKit mol.

    The mol is the specie's own (deprotonated) ``rdkit_obj`` -- SMILES and
    zwitterion corrections are already baked into it upstream (ChargeState) --
    so it has exactly ``specie.natoms`` atoms in the same order. One cell2mol
    Bond is created per RDKit bond; nothing else is computed here.
    """
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
    if mol.iscomplex or mol.has_ia_iia or mol.has_post_transition_metal:
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
                        # Chem.BondType.DATIVE
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
    if mol.iscomplex or mol.has_ia_iia or mol.has_post_transition_metal:
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
