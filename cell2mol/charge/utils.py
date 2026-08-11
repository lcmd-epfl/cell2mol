from __future__ import annotations

import numpy as np
import logging
from cell2mol.my_types import RDKitObject
from rdkit import Chem
from cell2mol.charge.xyz2mol import get_proto_mol, AC2mol, chiral_stereo_check
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDetermineBonds

logger = logging.getLogger(__name__)

# Backtracking steps rdDetermineBonds may take before giving up.
rddeterminebonds_max_iterations = 1000

# Monatomic noble gases
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}

# Ligand or molecules requiring manual charge assignment
MANUAL_CHARGE_ASSIGN_SPECIES = {
    "O4-Cl",
    "N3",
    "N2",
    "N-O",
    "I3",
    "I4",
    "I5",
    "I6",
    "H",
    "H2",
    "N-O3",
    "Te2",
    "N",
    "C",
    "H8-B3",
    "F6-Si",
    "O2",
    "Br3",
    "C-N-S",
}

# Species relying on 3c-2e bridge bonds, which no plain SMILES can express. Built
# from the specie's own adjacency instead, which also keeps them independent of
# per-CIF atom ordering.
BRIDGED_CLUSTER_CHARGES = {
    # arachno-[B3H8]-, octahydrotriborate: 6 terminal H + 2 bridging H over a
    # B3 framework. Not deltahedral, so is_borane_cage does not claim it.
    "H8-B3": -1,
}

HALOGENS = {"F", "Cl", "Br", "I"}

# Plausible oxidation states by atomic symbol
# Source: Venkataraman et al., J. Chem. Educ. 1997, 74, 915.
METAL_OXIDATION_STATES = {
    # Alkali metals
    "Li": [1],
    "Na": [1],
    "K": [1],
    "Rb": [1],
    "Cs": [1],
    "Fr": [1],
    # Alkaline earth metals
    "Be": [2],
    "Mg": [2],
    "Ca": [2],
    "Sr": [2],
    "Ba": [2],
    "Ra": [2],
    # 1st-row transition metals
    "Sc": [3],
    "Ti": [2, 3, 4],
    "V": [1, 2, 3, 4, 5],
    "Cr": [0, 2, 3],
    "Mn": [1, 2, 3],
    "Fe": [2, 3],
    "Co": [1, 2, 3],
    "Ni": [2, 3],
    "Cu": [1, 2],
    "Zn": [2],
    # 2nd-row transition metals
    "Y": [3],
    "Zr": [2, 3, 4],
    "Nb": [1, 3, 4, 5],
    "Mo": [0, 2, 4, 5, 6],
    "Tc": [1, 2, 3, 4, 5],
    "Ru": [2, 3],
    "Rh": [1, 2, 3],
    "Pd": [0, 2],
    "Ag": [1],
    "Cd": [2],
    # 3rd-row transition metals
    "Hf": [4],
    "Ta": [2, 3, 4, 5],
    "W": [0, 2, 4, 5, 6],
    "Re": [1, 2, 3, 4, 5, 7],
    "Os": [2, 3, 4, 5, 6],
    "Ir": [1, 3],
    "Pt": [0, 2, 4],
    "Au": [1, 3],
    "Hg": [2],
    # Post-transition metals
    "Al": [1, 3],
    "Ga": [3],
    "Ge": [2, 4],
    "In": [3],
    "Sn": [2, 4],
    "Tl": [1, 3],
    "Pb": [2, 4],
    "Bi": [3],
    # Lanthanides
    "La": [3],
    "Ce": [3, 4],
    "Pr": [3],
    "Nd": [2, 3],
    "Pm": [3],
    "Sm": [2, 3],
    "Eu": [3],
    "Gd": [3],
    "Tb": [3],
    "Dy": [3],
    "Ho": [3],
    "Er": [3],
    "Tm": [3],
    "Yb": [2, 3],
    "Lu": [3],
    # Actinides
    "Ac": [3],
    "Th": [3, 4],
    "Pa": [3, 4, 5],
    "U": [3, 4, 5, 6],
    "Np": [3, 4, 5, 6],
    "Pu": [3, 4, 5, 6],
    "Am": [3, 4, 5, 6],
    "Cm": [3, 4],
    "Bk": [3],
    "Cf": [3],
    "Es": [3],
    "Fm": [2, 3],
    "Md": [2, 3],
    "No": [2, 3],
    "Lr": [3],
}


# Oxidation states seen in mononuclear complexes in the CSD, as
# {element: {oxidation state: number of structures}}. METAL_OXIDATION_STATES above
# is the curated common subset that drives the primary charge search; this is the
# full attested range, and charge_balancer uses it to bound and to rank the
# fallbacks that run when that search finds no answer or several. Transition
# metals only -- elements absent here fall back to the curated table.
METAL_OS_OBSERVED: dict[str, dict[int, int]] = {
    # 1st-row transition metals
    "Sc": {2: 2, 3: 185},
    "Ti": {0: 4, 2: 59, 3: 180, 4: 1612},
    "V": {0: 11, 1: 32, 2: 95, 3: 296, 4: 726, 5: 949},
    "Cr": {0: 435, 1: 19, 2: 225, 3: 687, 4: 31, 5: 39, 6: 107},
    "Mn": {0: 6, 1: 1057, 2: 2522, 3: 752, 4: 117, 5: 22, 7: 3},
    "Fe": {0: 187, 1: 127, 2: 4734, 3: 2813, 4: 61, 5: 2},
    "Co": {0: 12, 1: 497, 2: 6023, 3: 3115, 4: 8, 5: 3},
    "Ni": {0: 267, 1: 247, 2: 11236, 3: 570, 4: 26, 6: 1},
    "Cu": {1: 3052, 2: 12128, 3: 95},
    "Zn": {1: 3, 2: 8115},
    # 2nd-row transition metals
    "Y": {2: 1, 3: 409},
    "Zr": {2: 31, 3: 4, 4: 705},
    "Nb": {0: 1, 1: 12, 2: 3, 3: 19, 4: 33, 5: 194},
    "Mo": {0: 205, 1: 10, 2: 161, 3: 63, 4: 291, 5: 137, 6: 1123},
    "Tc": {1: 58, 2: 10, 3: 42, 4: 15, 5: 114, 6: 5, 7: 8},
    "Ru": {0: 119, 1: 22, 2: 5556, 3: 440, 4: 222, 5: 2, 6: 16},
    "Rh": {0: 2, 1: 2232, 2: 37, 3: 1599, 4: 3, 5: 3},
    "Pd": {0: 177, 1: 6, 2: 10342, 3: 19, 4: 91},
    "Ag": {1: 1987, 2: 24, 3: 26},
    "Cd": {1: 1, 2: 2280},
    # 3rd-row transition metals
    "Hf": {2: 2, 3: 1, 4: 203},
    "Ta": {0: 1, 1: 3, 2: 4, 3: 14, 4: 22, 5: 290},
    "W": {0: 197, 1: 2, 2: 209, 3: 8, 4: 159, 5: 50, 6: 363},
    "Re": {0: 1, 1: 751, 2: 46, 3: 202, 4: 124, 5: 868, 6: 31, 7: 153},
    "Os": {0: 3, 1: 1, 2: 280, 3: 86, 4: 177, 5: 12, 6: 95, 8: 2},
    "Ir": {1: 1092, 2: 11, 3: 2985, 4: 25, 5: 12},
    "Pt": {0: 137, 1: 26, 2: 6939, 3: 47, 4: 890, 6: 1},
    "Au": {0: 2, 1: 2865, 2: 9, 3: 1183},
    "Hg": {2: 1084},
}


# The search can converge on a valence-consistent but absurd structure -- a ring
# drawn with alternating +1/-1 charges and no double bonds, to represent a small
# net charge. Anything far past this is an artifact, not real chemistry.
MAX_EXCESS_CHARGE_SEPARATION = 4

rdkit_atomic_valence: dict[int, list[int]] = {
    1: [1],  # H
    5: [3, 4],  # B
    6: [4],  # C
    7: [3, 4],  # N
    8: [2, 1, 3],  # O
    9: [1],  # F
    14: [4],  # Si
    15: [5, 3],  # P
    16: [6, 3, 2, 1],  # S
    17: [1],  # Cl
    32: [4],  # Ge
    35: [1],  # Br
    53: [1],  # I
}


def aromatic_info(mol: RDKitObject, added_indices=None):
    if added_indices is None:
        added_indices = []

    # Count aromatic atoms
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())

    # Count aromatic rings
    ring_info = mol.GetRingInfo()
    bond_rings = ring_info.BondRings()
    aromatic_ring_count = 0
    for bond_indices in bond_rings:
        if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_indices):
            aromatic_ring_count += 1

    # Check if hydrogen atoms are added in to aromatic atoms
    check = any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in added_indices)

    return {
        "Aromatic atoms": aromatic_atoms,
        "Number of aromatic rings": aromatic_ring_count,
        "Added to aromatic atoms": check,
    }


def _nitro_charge_atom_indices(mol: Chem.Mol, natoms: int) -> set[int]:
    """Atoms whose formal charge belongs to a nitro group -- the N(+1) and its
    single-bonded terminal O(-1). Detected locally, no sanitisation needed. Used to
    exempt this obligate separation from the excess-charge check.
    """
    nitro_atoms: set[int] = set()
    for i in range(natoms):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1:
            continue
        o_minus = None
        has_o_double = False
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(i, nb.GetIdx())
            if (
                nb.GetFormalCharge() == -1
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                o_minus = nb.GetIdx()
            elif (
                nb.GetFormalCharge() == 0 and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                has_o_double = True
        if o_minus is not None and has_o_double:
            nitro_atoms.add(i)
            nitro_atoms.add(o_minus)
    return nitro_atoms


def _malformed_nitro_atom(mol: Chem.Mol, natoms: int) -> int | None:
    """Index of a nitrogen drawn N([O-])[O-] rather than [N+](=O)[O-]. N cannot expand
    its octet, so one terminal O is always doubly bonded; the all-single form
    passes every valence check but is two charge units too negative. Rejected, not
    repaired -- promoting the bond would change the requested total charge.
    """
    for i in range(natoms):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 7:
            continue

        terminal_o_minus = 0
        has_double_bonded_o = False
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(i, neighbor.GetIdx())
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                has_double_bonded_o = True
            elif (
                bond.GetBondType() == Chem.BondType.SINGLE
                and neighbor.GetDegree() == 1
                and neighbor.GetFormalCharge() == -1
            ):
                terminal_o_minus += 1

        if terminal_o_minus >= 2 and not has_double_bonded_o:
            return i

    return None


def check_rdkit_obj_connectivity(mol: Chem.Mol, natoms: int, charge: int) -> bool:
    """
    Validates the chemical sanity of an RDKit molecule object by checking
    valences, lone pairs, bond connectivity, and that the total formal
    charge actually matches the requested target charge.
    """
    pt = Chem.GetPeriodicTable()
    is_correct = True

    # 0. Total Formal Charge Check
    total_formal_charge = sum(
        mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(natoms)
    )
    if total_formal_charge != charge:
        logger.debug(
            "   Total formal charge mismatch: got %d, expected %d",
            total_formal_charge,
            charge,
        )
        is_correct = False

    # 0b/0c. A structure needing far more formal charge than the net demands is a
    # search artifact, not a resonance form. Nitro is obligate -- a polynitro
    # compound legitimately carries many -- so its charges are excluded before
    # measuring, and it must be drawn [N+](=O)[O-], never N([O-])[O-].
    bad_nitro = _malformed_nitro_atom(mol, natoms)
    if bad_nitro is not None:
        logger.debug(
            "   Malformed nitro on atom %d: two terminal O(-1) and no N=O; "
            "the correct [N+](=O)[O-] carries a different total charge",
            bad_nitro,
        )
        is_correct = False

    nitro_charge_atoms = _nitro_charge_atom_indices(mol, natoms)
    total_abs_atom_charge = sum(
        abs(mol.GetAtomWithIdx(i).GetFormalCharge())
        for i in range(natoms)
        if i not in nitro_charge_atoms
    )
    if total_abs_atom_charge > abs(charge) + MAX_EXCESS_CHARGE_SEPARATION:
        logger.debug(
            "   Excessive charge separation: total |formal charge| on atoms = %d, "
            "target net charge = %d (max allowed excess = %d)",
            total_abs_atom_charge,
            charge,
            MAX_EXCESS_CHARGE_SEPARATION,
        )
        is_correct = False

    for i in range(natoms):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        # Old : valence = atom.GetTotalValence()
        try:
            # New RDKit API (2024.03+)
            valence = atom.GetValence(Chem.ValenceType.TOTAL)
        except AttributeError:
            # Fallback for older RDKit versions if ValenceType doesn't exist
            valence = atom.GetTotalValence()

        # Calculate lone pairs: (Valence Electrons - Formal Charge - Shared Electrons) / 2
        num_valence_electrons = pt.GetNOuterElecs(atom.GetAtomicNum())
        lone_pairs = (num_valence_electrons - formal_charge - valence) / 2

        # 1. Lone Pair Sanity Check
        if lone_pairs not in [0, 1, 2, 3, 4]:
            logger.debug(
                "   Lone pair error at atom %d (%s): %f", i, symbol, lone_pairs
            )
            is_correct = False

        # 2. Aromaticity & Bond Consistency Check
        # We check total shared electrons (valence) against the RDKit valence model
        is_aromatic = atom.GetIsAromatic()

        if not is_aromatic:
            try:
                # New RDKit API
                explicit = atom.GetValence(Chem.ValenceType.EXPLICIT)
                implicit = atom.GetValence(Chem.ValenceType.IMPLICIT)
            except AttributeError:
                # Old RDKit API
                explicit = atom.GetExplicitValence()
                implicit = atom.GetImplicitValence()

            # Check if calculated shared electrons match expected valence
            if valence != explicit + implicit:
                logger.debug("   Valence mismatch at atom %d (%s)", i, symbol)
                is_correct = False

            # 3. Total Electron Count Check
            # Shared electrons + electrons in lone pairs + charge should equal outer shell count
            calc_total_elecs = valence + (int(lone_pairs) * 2) + formal_charge
            if calc_total_elecs != num_valence_electrons:
                logger.debug(
                    "   Total electron count mismatch at atom %d (%s)", i, symbol
                )
                is_correct = False

    # 4. A C=C the search failed to place comes back split as [C+]-[C-] or
    # [C-]-[C-], whose lone pairs would pair into the missing bond -- which is
    # what makes a toluene ring look like a valid -2/-4 state. Only C/C pairs are
    # tested, and aromatic or already-multiply-bonded ones are left intact.
    for bond in mol.GetBonds():
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        if begin_atom.GetAtomicNum() != 6 or end_atom.GetAtomicNum() != 6:
            continue
        q_begin, q_end = begin_atom.GetFormalCharge(), end_atom.GetFormalCharge()
        if q_begin == 0 or q_end == 0:
            continue

        opposite_sign = (q_begin > 0) != (q_end > 0)
        both_carbanion = q_begin < 0 and q_end < 0
        if both_carbanion:
            # Only a localised (non-aromatic) single/double bond has a lone
            # pair pair-up available; an aromatic ring anion or an existing
            # triple bond does not.
            if begin_atom.GetIsAromatic() or end_atom.GetIsAromatic():
                continue
            if bond.GetBondTypeAsDouble() >= 3:
                continue
        elif not opposite_sign:
            continue  # two carbocations cannot pair into a double bond

        logger.debug(
            "   Reducible C%+d/C%+d pair at atoms %d-%d (missing C=C double bond)",
            q_begin,
            q_end,
            begin_atom.GetIdx(),
            end_atom.GetIdx(),
        )
        is_correct = False
        break

    return is_correct


def generate_rdkit_mol_from_rdDetermineBonds(
    atoms,
    coords,
    AC,
    charge=0,
    sanitize=True,
    allow_charged_fragments=True,
    embed_chiral=True,
):
    """Build an RDKit Mol from atomic numbers, coordinates and an adjacency matrix,
    assigning bond orders with rdDetermineBonds.DetermineBondOrders.
    """

    atoms = list(map(int, atoms))
    coords = np.asarray(coords, dtype=float)
    ac = np.asarray(AC)
    n_atoms = len(atoms)

    if coords.shape != (n_atoms, 3):
        raise ValueError(f"coords must have shape ({n_atoms}, 3), got {coords.shape}")

    if ac.shape != (n_atoms, n_atoms):
        raise ValueError(f"AC must have shape ({n_atoms}, {n_atoms}), got {ac.shape}")

    rwMol = Chem.RWMol()

    # Add atoms
    for atomic_num in atoms:
        rwMol.AddAtom(Chem.Atom(atomic_num))

    # Add connectivity as single bonds first
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if ac[i, j] != 0:
                rwMol.AddBond(i, j, Chem.BondType.SINGLE)

    mol = rwMol.GetMol()

    # Add coordinates
    conf = Chem.Conformer(n_atoms)
    conf.Set3D(True)

    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    mol.AddConformer(conf, assignId=True)

    # Assign bond orders using existing connectivity. Capped: without a limit
    # this hangs outright on some species, and the caller cannot tell a slow
    # search from a stuck one. Raises RuntimeError when the cap is reached.
    rdDetermineBonds.DetermineBondOrders(
        mol,
        charge=charge,
        allowChargedFragments=allow_charged_fragments,
        embedChiral=embed_chiral,
        maxIterations=rddeterminebonds_max_iterations,
    )

    if sanitize:
        Chem.SanitizeMol(mol)

    return mol


def generate_rdkit_mol_from_AC2mol(
    atoms,
    AC,
    charge=0,
    allow_charged_fragments=True,
    embed_chiral=True,
    diagnostics=None,
):
    """Build an RDKit Mol from atomic numbers, coordinates and an adjacency matrix,
    assigning bond orders with the modified AC2mol from xyz2mol. ``diagnostics``,
    if given, is filled in with why bond assignment gave up when the reason is
    worth telling the user about.
    """
    new_mols, BO = AC2mol(
        mol=get_proto_mol(atoms),
        AC=AC,
        atoms=atoms,
        charge=charge,
        allow_charged_fragments=allow_charged_fragments,
        diagnostics=diagnostics,
    )

    # Early Exit if no candidates found
    if not new_mols:
        logger.warning(f"No mol found for charge {charge}")
        return None

    # Stereo and Chirality Validation
    if embed_chiral:
        if not all(chiral_stereo_check(mol) for mol in new_mols):
            logger.error("Chirality check failed for one or more candidates")
            return None

    return new_mols[0]  # use the first candidate as default
