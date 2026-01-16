from rdkit import Chem
from typing import Tuple
from cell2mol.element_utils import (
    get_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
)
from cell2mol.elementdata import ElementData
import logging

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def correct_smiles_ligand(ligand: object) -> Tuple[bool, bool]:
    """
    Constructs an RDKit molecule from cell2mol ligand object.
    This function synchronizes the cell2mol ligand object with RDKit,
    handling bond orders, hybridization, and zwitterionic corrections.
    """

    rwlig = Chem.RWMol()

    # --- atoms ---
    for atom in ligand.atoms:
        rd_atom = Chem.Atom(atom.atnum)
        rd_atom.SetFormalCharge(int(atom.charge))
        rd_atom.SetNoImplicit(True)
        rwlig.AddAtom(rd_atom)

    # --- metal context ---
    labels = ligand.get_parent("molecule").labels
    metal_idxs = get_metal_idxs(labels)
    alkali_idxs = get_alkali_alkaline_earth_metal_idxs(labels)

    def skip_bond(bond) -> bool:
        blabels = [bond.atom1.label, bond.atom2.label]

        if any(elemdatabase.elementblock[l] in {"d", "f"} for l in blabels):
            return True
        if get_alkali_alkaline_earth_metal_idxs(blabels):
            return True
        if (
            not metal_idxs
            and not alkali_idxs
            and get_post_transition_metal_idxs(blabels)
        ):
            return True
        return False

    btype_map = {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC,
    }

    hyb_map = {
        1: Chem.HybridizationType.S,
        2: Chem.HybridizationType.SP,
        3: Chem.HybridizationType.SP2,
        4: Chem.HybridizationType.SP3,
    }

    # --- bonds & hybridization ---
    for jdx, atom in enumerate(ligand.atoms):
        if atom.bonds is None:
            logger.error("Ligand atom %s has no bond information.", atom.label)
            raise ValueError("Ligand atom bonds are not set")

        nbonds = 0

        for b in atom.bonds:
            if skip_bond(b):
                continue

            begin_idx = b.atom1.get_parent_index("ligand")
            end_idx = b.atom2.get_parent_index("ligand")
            nbonds += 1

            btype = btype_map.get(b.order, Chem.BondType.SINGLE)

            if b.order == 1.5:
                rwlig.GetAtomWithIdx(begin_idx).SetIsAromatic(True)
                rwlig.GetAtomWithIdx(end_idx).SetIsAromatic(True)

            if begin_idx == jdx and end_idx > jdx:
                rwlig.AddBond(begin_idx, end_idx, btype)

        rwlig.GetAtomWithIdx(jdx).SetHybridization(
            hyb_map.get(nbonds, Chem.HybridizationType.UNSPECIFIED)
        )

    # --- zwitterion correction ---
    temp_obj = rwlig.GetMol()
    logger.debug(
        "Ligand structure before zwitterion fix: %s", Chem.MolToSmiles(temp_obj)
    )

    obj, fix_zwitterions = fix_zwitterions_in_adjacent_atoms(temp_obj)

    if fix_zwitterions:
        logger.debug("Zwitterions fixed. Updated SMILES: %s", Chem.MolToSmiles(obj))

    try:
        Chem.SanitizeMol(
            obj,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            catchErrors=True,
        )

        Chem.DetectBondStereochemistry(obj, -1)
        Chem.AssignStereochemistry(obj, flagPossibleStereoCenters=True, force=True)
        Chem.AssignAtomChiralTagsFromStructure(obj, -1)

        final_smiles = Chem.MolToSmiles(obj)
        logger.debug(
            "Ligand %s: Original [%s] -> Corrected [%s]",
            ligand.formula,
            ligand.smiles,
            final_smiles,
        )

        ligand.smiles = final_smiles
        ligand.rdkit_obj = obj

        if fix_zwitterions:
            ligand.set_charges(
                atomic_charges=[a.GetFormalCharge() for a in obj.GetAtoms()]
            )

        return True, fix_zwitterions

    except Exception as e:
        logger.error("RDKit processing failed for ligand %s: %s", ligand.formula, e)
        return False, fix_zwitterions


def fix_zwitterions_in_adjacent_atoms(mol):
    """
    Fixes zwitterionic artifacts by adjusting formal charges between adjacent atoms
    with opposite charges in an RDKit molecule object.

    Args:
        mol: Input RDKit molecule.

    Returns:
        tuple: (Corrected Chem.Mol, bool indicating if changes were made).
    """
    rw_mol = Chem.RWMol(mol)
    fix_zwitterions = False

    for atom in rw_mol.GetAtoms():
        fcharge = atom.GetFormalCharge()

        # Target positively charged atoms to find adjacent negative partners
        if fcharge > 0:
            atom_idx = atom.GetIdx()
            atom_label = atom.GetSymbol()
            neighbors = atom.GetNeighbors()

            logger.debug(
                "Checking positive atom: %s (idx=%d, charge=%d)",
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
                        fix_zwitterions = True
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

    return rw_mol.GetMol(), fix_zwitterions


def generate_tmc_rdkit_obj_smiles(mol: object):
    all_metals_indices = [met.get_parent_index("molecule") for met in mol.metals]
    logger.debug(
        "Found metals %s %s",
        [met.atom_site_label for met in mol.metals],
        all_metals_indices,
    )
    temp_mol = Chem.RWMol()

    for met in mol.metals:
        a = Chem.Atom(met.label)
        a.SetFormalCharge(int(met.charge))  # Assign the metal oxidation state
        a.SetIntProp("__mol_idx", met.get_parent_index("molecule"))
        if getattr(met, "atom_site_label", None) is not None:
            a.SetProp("__atom_site_label", met.atom_site_label)

        idx = temp_mol.AddAtom(a)
        logger.debug("Add metal atom %s", Chem.MolToSmiles(temp_mol))

    for lig in mol.ligands:
        # lig_atom : atom object from cell2mol
        # a : atom object from rdkit object
        for lig_atom, a in zip(lig.atoms, lig.rdkit_obj.GetAtoms()):
            a.SetFormalCharge(int(lig_atom.charge))
            a.SetIntProp("__mol_idx", lig_atom.get_parent_index("molecule"))

            if getattr(lig_atom, "atom_site_label", None) is not None:
                a.SetProp("__atom_site_label", lig_atom.atom_site_label)

        logger.debug("Add ligand with %s %s", lig.formula, lig.totcharge)

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

    for met in mol.metals:
        met_idx = met.get_parent_index("molecule")

        coordinating_atoms_labels = [atom.label for atom in met.coord_sphere]

        coordinating_atoms_indices = [
            atom.get_parent_index("molecule") for atom in met.coord_sphere
        ]
        logger.debug(
            "%s (%s) coordinates to %s %s",
            met.label,
            met.atom_site_label,
            coordinating_atoms_labels,
            coordinating_atoms_indices,
        )

        for idx in coordinating_atoms_indices:
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


def create_bonds_specie(specie, rdkit_obj: object = None):
    from cell2mol.classes import Bond

    logger.debug(
        "CREATE_bonds_specie: %s %s %s", specie.formula, specie.subtype, specie.smiles
    )
    n_atoms = specie.natoms  # e.g. 9
    if rdkit_obj is not None:
        rdkit_obj = rdkit_obj
    else:
        rdkit_obj = specie.rdkit_obj

    n_atoms_rdkit = rdkit_obj.GetNumAtoms()  # e.g.10

    if n_atoms == n_atoms_rdkit:
        logger.debug(
            "Number of atoms in %s object and RDKit object are equal: %d %d",
            specie.subtype,
            n_atoms,
            n_atoms_rdkit,
        )

        # e.g. idx 0, 1, 2, 3, 4, 5, 6, 7, 8
        for idx, rdkit_atom in enumerate(rdkit_obj.GetAtoms()):
            # logger.debug(
            #     "%d %s Number of bonds : %d",
            #     idx,
            #     rdkit_atom.GetSymbol(),
            #     len(rdkit_atom.GetBonds()),
            # )
            if len(rdkit_atom.GetBonds()) == 0:
                logger.debug(
                    "NO BONDS CREATED for %s due to no bonds in %s RDKit object",
                    specie.atoms[idx].label,
                    specie.subtype,
                )
            else:
                for b in rdkit_atom.GetBonds():
                    bond_startatom = b.GetBeginAtomIdx()
                    bond_endatom = b.GetEndAtomIdx()
                    bond_order = b.GetBondTypeAsDouble()
                    if (
                        specie.atoms[bond_endatom].label == "D"
                        and rdkit_obj.GetAtomWithIdx(bond_endatom).GetSymbol() == "H"
                    ):
                        if bond_endatom == idx:
                            start = bond_endatom
                            end = bond_startatom
                        elif bond_startatom == idx:
                            start = bond_startatom
                            end = bond_endatom
                        # create new bond object
                        # logger.debug(
                        #     "BOND CREATED %d %d %d %f %s %s",
                        #     idx,
                        #     start,
                        #     end,
                        #     bond_order,
                        #     specie.atoms[start].label,
                        #     specie.atoms[end].label,
                        # )
                        new_bond = Bond.from_positional(
                            specie.atoms[start], specie.atoms[end], bond_order
                        )
                        specie.atoms[idx].add_bond(new_bond)

                    elif (
                        specie.atoms[bond_endatom].label
                        != rdkit_obj.GetAtomWithIdx(bond_endatom).GetSymbol()
                    ):
                        logger.debug(
                            "Error with Bond EndAtom %s %s",
                            specie.atoms[bond_endatom].label,
                            rdkit_obj.GetAtomWithIdx(bond_endatom).GetSymbol(),
                        )
                    else:
                        if bond_endatom == idx:
                            start = bond_endatom
                            end = bond_startatom
                        elif bond_startatom == idx:
                            start = bond_startatom
                            end = bond_endatom

                        # create new bond object
                        # logger.debug(
                        #     "BOND CREATED %d %d %d %f %s %s",
                        #     idx,
                        #     start,
                        #     end,
                        #     bond_order,
                        #     specie.atoms[start].label,
                        #     specie.atoms[end].label,
                        # )
                        new_bond = Bond.from_positional(
                            specie.atoms[start], specie.atoms[end], bond_order
                        )
                        specie.atoms[idx].add_bond(new_bond)

                if specie.atoms[idx].bonds is not None:
                    pass
                else:
                    if specie.natoms == 1:
                        pass
                    else:
                        logger.error(
                            "NO BONDS for %s with %s RDKit object index %d. Please check the RDKit object.",
                            specie.atoms[idx].label,
                            specie.subtype,
                            idx,
                        )
                        return False  # return False if no bonds are created
    else:
        logger.debug(
            "Number of atoms in %s object and RDKit object are different: %d %d",
            specie.subtype,
            n_atoms,
            n_atoms_rdkit,
        )
        non_bonded_atoms = list(range(0, n_atoms_rdkit))[n_atoms:]
        logger.debug("NON_BONDED_ATOMS %s", non_bonded_atoms)

        # e.g. idx 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        for idx, rdkit_atom in enumerate(rdkit_obj.GetAtoms()):
            # logger.debug(
            #     "\t%d %s Number of bonds : %d",
            #     idx,
            #     rdkit_atom.GetSymbol(),
            #     len(rdkit_atom.GetBonds()),
            # )
            if len(rdkit_atom.GetBonds()) == 0:
                logger.debug(
                    "NO BONDS CREATED for %s due to no bonds in %s RDKit object",
                    rdkit_atom.GetSymbol(),
                    specie.subtype,
                )
            else:
                for b in rdkit_atom.GetBonds():
                    bond_startatom = b.GetBeginAtomIdx()
                    bond_endatom = b.GetEndAtomIdx()
                    bond_order = b.GetBondTypeAsDouble()

                    if (
                        bond_startatom in non_bonded_atoms
                        or bond_endatom in non_bonded_atoms
                    ):
                        logger.debug(
                            "NO BOND CREATED %d or %d is not in the specie.atoms. It belongs to %s.",
                            bond_startatom,
                            bond_endatom,
                            non_bonded_atoms,
                        )
                    else:
                        if bond_endatom == idx:
                            start = bond_endatom
                            end = bond_startatom
                        elif bond_startatom == idx:
                            start = bond_startatom
                            end = bond_endatom

                        # create new bond object
                        # logger.debug(
                        #     "BOND CREATED %d %d %d %f %s %s",
                        #     idx,
                        #     start,
                        #     end,
                        #     bond_order,
                        #     specie.atoms[start].label,
                        #     specie.atoms[end].label,
                        # )
                        new_bond = Bond.from_positional(
                            specie.atoms[start], specie.atoms[end], bond_order
                        )
                        specie.atoms[idx].add_bond(new_bond)

                if idx not in non_bonded_atoms:
                    if specie.atoms[idx].bonds is not None:
                        pass
                    else:
                        if specie.natoms == 1:
                            pass
                        else:
                            logger.error(
                                "NO BONDS for %s with %s RDKit object index %d. Please check the RDKit object.",
                                specie.atoms[idx].label,
                                specie.subtype,
                                idx,
                            )
                            return False  # return False if no bonds are created
                else:
                    logger.debug(
                        "NO BONDS for %s with %s RDKit object index %d because it is an added atom",
                        rdkit_atom.GetSymbol(),
                        specie.subtype,
                        idx,
                    )

    return True


def create_metal_ligand_bonds(mol: object):
    # Adds Metal-Ligand Bonds, with a zero order:
    from cell2mol.classes import Bond

    if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
        for lig in mol.ligands:
            for at in lig.atoms:
                count = 0
                index_1 = at.get_parent_index("molecule")
                for met in mol.metals:
                    index_2 = met.get_parent_index("molecule")
                    isconnected = mol.madjmat[index_1, index_2] == 1
                    if isconnected:
                        if index_1 < index_2:
                            bond_startatom = at
                            bond_endatom = met
                        else:
                            bond_startatom = met
                            bond_endatom = at
                        newbond = Bond.from_positional(bond_startatom, bond_endatom, 0)
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


def create_metal_metal_bonds(mol: object):
    from cell2mol.classes import Bond

    # Adds Metal-Metal Bonds, with a zero order:
    if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
        if len(mol.metals) > 1:
            logger.debug("Creating Metal-Metal Bonds for molecule %s", mol.formula)
            for idx, met1 in enumerate(mol.metals):
                index_1 = met1.get_parent_index("molecule")
                for jdx, met2 in enumerate(mol.metals):
                    if idx <= jdx:
                        continue
                    index_2 = met2.get_parent_index("molecule")
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
