from __future__ import annotations

import numpy as np
import logging
from cell2mol.my_types import RDKitObject
from rdkit import Chem
from cell2mol.charge.xyz2mol import get_proto_mol, AC2mol, chiral_stereo_check
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDetermineBonds

logger = logging.getLogger(__name__)

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
}

# Manual species whose structure cannot be written as a plain SMILES because it
# relies on 3-centre-2-electron bridge bonds. Their mol is built from the
# specie's own adjacency instead of a registry SMILES (see
# generate_manual_charge_state), which also keeps them independent of the
# per-CIF atom ordering a fixed SMILES string could not track.
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
    "Cr": [0, 2, 3],  # +5 intentionally excluded
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


# AC2mol's combinatorial bond-order search can converge on a technically
# valence-consistent but chemically absurd resonance structure: e.g. an
# entire ring system drawn with alternating +1/-1 formal charges and no
# double bonds at all, just to represent a small net charge; anything
# far beyond that is a search artifact, not real chemistry.
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
    """Indices of atoms whose formal charge belongs to a nitro group,
    ``[N+](=O)[O-]`` -- the N(+1) and the single-bonded terminal O(-1).

    Detected locally (no sanitisation/aromaticity needed): an N with formal
    charge +1 bearing both a single-bonded O(-1) and a double-bonded O(0).
    Used to exempt this obligate charge separation from the excess-charge check.
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

    # 0b. Charge Separation Sanity Check
    # A valid Lewis structure can concentrate the net charge on a small
    # number of atoms (a real zwitterion/ylide), but a structure requiring
    # far more formal charge than the net charge demands is a bond-order
    # search artifact (see MAX_EXCESS_CHARGE_SEPARATION above), not a
    # legitimate resonance form.
    #
    # Exception: a nitro group [N+](=O)[O-] is an *obligate* charge-separated
    # group -- there is no neutral Lewis structure for it -- so its +1/-1 pair
    # is real chemistry, not a search artifact. A polynitro compound may
    # legitimately carry many (e.g. an octanitro-porphyrin: 8 * (|+1|+|-1|) =
    # 16). Exclude nitro N+/O- charges before measuring separation; a genuine
    # artifact (alternating +/- around a ring/chain) is untouched.
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

        # logger.debug(
        #     "Charge: %d | Atom: %2d %2s | Q: %2d | V: %2d | LP: %2d | Correct: %s",
        #     charge,
        #     i,
        #     symbol,
        #     formal_charge,
        #     valence,
        #     lone_pairs,
        #     is_correct,
        # )

    # 4. Reducible charged-carbon pair check
    # rdDetermineBonds sometimes "pays" for a C=C double bond it fails to
    # place by splitting it into an adjacent charged-carbon pair, so a bond
    # that should be C=C comes back as either:
    #   - [C+]-[C-] (opposite charges): the carbanion lone pair would fill
    #     the carbocation's empty orbital to give a neutral C=C -- an
    #     electron-conserving stand-in for the missing double bond; or
    #   - [C-]-[C-] (both carbanions): the two lone pairs would pair into the
    #     missing double bond, neutralising both.
    # Either way a lower-|charge| structure exists at a nearby charge, so the
    # separated form is a bond-order-search artifact, not a ground-state
    # Lewis form -- it is what leaves a toluene ring looking like a valid
    # -2/-4 state or a cyclopentadiene ring like a valid -4 tetra-anion, even
    # though every per-atom valence check above passes.
    #
    # Only C/C pairs are tested, so genuine heteroatom ylides (P+=C-, diazo
    # C-=N+) are untouched. Opposite-sign pairs are always rejected
    # (including the aromatic [c+]-[c-] form). Same-sign carbanion pairs are
    # rejected only when non-aromatic and not already multiply bonded, so a
    # delocalised aromatic poly-anion (Cp-, cyclooctatetraene dianion, ...)
    # and an acetylide/carbide triple bond ([C-]#[C-], no room for a further
    # bond) are both left intact. Two carbocations are never flagged -- with
    # no lone pair on either, they cannot pair into a double bond.
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
    """
    Build an RDKit Mol object from atomic numbers, coordinates, and
    adjacency matrix from a protonation state, and
    then assign bond orders using rdDetermineBonds.DetermineBondOrders.
    Parameters
    ----------
    atoms : list[int]
        Atomic numbers, e.g. [6, 1, 1, 1, 1]
    coords : array-like, shape (n_atoms, 3)
        Cartesian coordinates.
    AC : array-like, shape (n_atoms, n_atoms)
        Connectivity matrix. Nonzero means bonded.
    charge : int
        Total molecular charge.
    sanitize : bool
        Whether to sanitize after bond-order assignment.
    allow_charged_fragments : bool
        Whether to allow charged fragments.
    embed_chiral : bool
        Whether to embed chiral information.

    Returns
    -------
    mol : rdkit.Chem.Mol
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

    # Assign bond orders using existing connectivity
    rdDetermineBonds.DetermineBondOrders(
        mol,
        charge=charge,
        allowChargedFragments=allow_charged_fragments,
        embedChiral=embed_chiral,
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
):
    """
    Build an RDKit Mol object from atomic numbers, coordinates, and
    adjacency matrix from a protonation state, and
    then assign bond orders
    - using rdDetermineBonds.DetermineBondOrders.
    - modified AC2mol from xyz2mol
    Parameters
    ----------
    atoms : list[int]
        Atomic numbers, e.g. [6, 1, 1, 1, 1]
    coords : array-like, shape (n_atoms, 3)
        Cartesian coordinates.
    AC : array-like, shape (n_atoms, n_atoms)
        Connectivity matrix. Nonzero means bonded.
    charge : int
        Total molecular charge.
    sanitize : bool
        Whether to sanitize after bond-order assignment.
    allow_charged_fragments : bool
        Whether to allow charged fragments.
    embed_chiral : bool
        Whether to embed chiral information.

    Returns
    -------
    mol : rdkit.Chem.Mol
    """
    new_mols, BO = AC2mol(
        mol=get_proto_mol(atoms),
        AC=AC,
        atoms=atoms,
        charge=charge,
        allow_charged_fragments=allow_charged_fragments,
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
