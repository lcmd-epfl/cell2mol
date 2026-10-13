#!/usr/bin/env python
# This file includes code from xyz2mol.py
# Original work:
#   Jensen Group (2018)
#   Licensed under the MIT License
#   https://github.com/jensengroup/xyz2mol
#
# Modifications:
#   - Adapted for use in cell2mol

import copy
import itertools
import logging
import math

try:
    from rdkit.Chem import rdEHTTools  # requires RDKit 2019.9.1 or later
except ImportError:
    rdEHTTools = None

from collections import defaultdict

import numpy as np
import networkx as nx

from rdkit import Chem
from cell2mol.elementdata import ElementData
from cell2mol.element_utils import labels2formula

logger = logging.getLogger(__name__)

elemdatabase = ElementData()

valence_electrons = []
for i in elemdatabase.elementsym:
    try:
        valence_electrons.append(
            elemdatabase.valenceelectrons[elemdatabase.elementsym[i]]
        )
    except KeyError:
        continue
global atomic_valence
global atomic_valence_electrons

atomic_valence_electrons = dict(zip(elemdatabase.elementsym, valence_electrons))

valence_combinations_limit = 1_000_000_000
num_try_limit = 100


def get_atomic_valences(k):
    symb = elemdatabase.elementsym[k]
    block = elemdatabase.elementblock[symb]
    group = elemdatabase.elementgroup[symb]
    period = elemdatabase.elementperiod[symb]
    ave = elemdatabase.valenceelectrons[symb]
    if k == 5:  # B
        return [3, 4]
    if k == 6:  # C
        return [4, 2]
    if k == 7:  # N
        return [3, 4]
    if k == 8:  # O
        return [2, 1, 3]
    if k == 13:  # Al
        return [3, 4, 5]
    if k == 14:  # Si
        return [4]
    if k == 15:  # P
        return [3, 5]  # [5,4,3]
    if k == 16:  # S
        return [2, 4, 6]
    if k == 32:  # Ge
        return [4, 6]
    if k == 33:  # As
        return [5, 3]  # [5,4,3]
    if k == 34:  # Se
        return [2, 4, 6]
    if k == 51:  # Sb
        return [6, 5, 4, 3]  # [5,4,3]
    if k == 52:  # Te
        return [2, 4, 6]
    if k == 53:  # I
        return [1, 3, 5]
    if k == 82:  # Pb
        return [2, 4, 6]
    if k == 83:  # Bi
        return [3, 5, 6, 7, 8]
    if group == 17:  # F, Cl, Br, At
        return [1]
    if block == "s" and period == 1:
        av = 2 - ave
    elif group == 1 and period != 1:
        av = 20
    elif group == 2:
        av = 20
    elif group == 18:
        av = 0
    elif block == "p" and group != 18:
        av = 8 - ave
    elif block in ["d", "f"]:
        av = 20
    else:
        av = 1
    return [av]


atomic_valence = defaultdict(list)
for k in elemdatabase.elementsym:
    try:
        av = get_atomic_valences(k)
        atomic_valence[k].extend(av)
    except KeyError:
        continue
backup = copy.deepcopy(atomic_valence)


def str_atom(atom):
    """
    convert integer atom to string atom
    """
    atom = elemdatabase.elementsym[atom]
    return atom


def int_atom(atom):
    """
    convert str atom to integer atom
    """
    atom = elemdatabase.elementnr[atom]
    return atom


def get_UA(maxValence_list, valence_list):
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    BO = AC.copy()
    DU_save = []

    ua, du = UA, DU
    while DU_save != du:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(du)
        ua, du = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs_new(ua, AC, du, use_graph=use_graph)[0]

    return BO


def valences_not_too_large(BO, valences):
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def charge_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
    allow_carbenes=True,
):
    total_q = 0

    q_list = []

    if allow_charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(
                atom,
                atomic_valence_electrons[atom],
                BO_valences[i],
                neighbor_atoms=_neighbor_atomic_numbers(AC, i, atoms),
            )
            total_q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if (
                    not allow_carbenes
                    and number_of_single_bonds_to_C == 2
                    and BO_valences[i] == 2
                ):
                    total_q += 1
                    q = 2
                    logger.info("Carbenes are not allowed in this molecule")
                if number_of_single_bonds_to_C == 3 and total_q + 1 < charge:
                    total_q += 2
                    q = 1

            if q != 0:
                q_list.append(q)
    return charge == total_q


def BO_is_OK(
    BO,
    AC,
    charge,
    DU,
    atomic_valence_electrons,
    atoms,
    valences,
    allow_charged_fragments=True,
    allow_carbenes=True,
):
    """Check bond-order, unsaturation and charge consistency."""

    if not valences_not_too_large(BO, valences):
        return False

    check_sum = (BO - AC).sum() == sum(DU)
    check_charge = charge_is_OK(
        BO,
        AC,
        charge,
        DU,
        atomic_valence_electrons,
        atoms,
        valences,
        allow_charged_fragments,
        allow_carbenes=True,
    )

    if check_charge and check_sum:
        return True

    return False


# Electronegative substituents preserve the lone pair in four-coordinate Sb anions.
_SB_III_SUBSTITUENTS = frozenset({7, 8, 9, 16, 17, 34, 35, 53})  # N O F S Cl Se Br I


def _neighbor_atomic_numbers(matrix, index, atoms) -> list[int]:
    """Atomic numbers bonded to ``index``, read off a BO or adjacency matrix."""
    if matrix is None:
        return []
    return [j_atom for j, j_atom in enumerate(atoms) if j != index and matrix[index][j]]


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence, neighbor_atoms=None):
    """Calculate formal charge from bond-order sum and, when needed, neighbour elements."""
    label = elemdatabase.elementsym[atom]
    group = elemdatabase.elementgroup[label]

    found = False
    # Alkali except H
    if group == 1 and label != "H" and BO_valence == 0 and not found:
        charge = 1
        found = True
    # Hydrogen
    elif atom == 1 and not found:
        charge = 1 - BO_valence
        found = True
    # Alkaline earth metals
    elif group == 2 and BO_valence == 0 and not found:
        charge = 2
        found = True
    # Boron
    elif atom == 5 and not found:
        charge = 3 - BO_valence
        found = True
    # Ionic bonding requires element-specific charge rules.
    elif atom == 15 and BO_valence == 5 and not found:  # PX5
        charge = 0
        found = True
    elif atom == 15 and BO_valence == 6 and not found:  # PX6
        charge = -1
        found = True
    elif atom == 16 and BO_valence == 6 and not found:  # SX6
        charge = 0
        found = True
    elif atom == 16 and BO_valence == 4 and not found:
        charge = 0
        found = True
    elif atom == 16 and BO_valence == 5 and not found:
        charge = 1
        found = True
    elif atom == 32 and BO_valence == 6 and not found:
        # GeX6 has charge 4-6=-2.
        charge = -2
        found = True
    elif atom == 33 and BO_valence == 6 and not found:  # AsX6
        charge = -1
        found = True
    elif atom == 50 and BO_valence == 4 and not found:  # SnX4
        charge = 0
        found = True
    elif atom == 51 and BO_valence in (3, 5) and not found:  # SbR3, SbX5
        charge = 0
        found = True
    elif atom == 51 and BO_valence == 6 and not found:  # SbX6-: 5 - 6 = -1
        charge = -1
        found = True
    elif atom == 51 and BO_valence == 4 and not found:
        # [SbR4]+ (Sb(V), no lone pair) vs [R2SbX2]- (Sb(III), lone pair kept).
        # Only the substituents tell them apart; default to the anion.
        neighbors = neighbor_atoms or []
        sb_iii = any(z in _SB_III_SUBSTITUENTS for z in neighbors)
        charge = -1 if (sb_iii or not neighbors) else 1
        found = True
    elif atom == 52 and BO_valence in (2, 4, 6) and not found:  # TeX2, TeX4, TeX6
        charge = 0
        found = True
    elif atom == 53 and BO_valence in (3, 5) and not found:  # I(III)/I(V), hypervalent
        charge = 0
        found = True
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def BO2mol(
    mol,
    BO_matrix,
    atoms,
    atomic_valence_electrons,
    mol_charge,
    allow_charged_fragments=True,
):
    """Build an RDKit molecule from bond orders and atomic charges.

    Based on code by Paolo Toscani.
    """

    n_bo = len(BO_matrix)
    n_atoms = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))
    if n_bo != n_atoms:
        raise RuntimeError(
            "sizes of adjMat ({0:d}) and Atoms {1:d} differ".format(n_bo, n_atoms)
        )

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    for i in range(n_bo):
        for j in range(i + 1, n_bo):
            bo = int(round(BO_matrix[i, j]))
            if bo == 0:
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(
            mol, atoms, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge
        )
    else:
        mol = set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences)

    return mol


def set_atomic_charges(
    mol, atoms, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge
):
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(
            atom,
            atomic_valence_electrons[atom],
            BO_valences[i],
            neighbor_atoms=_neighbor_atomic_numbers(BO_matrix, i, atoms),
        )
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                q += 1
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if abs(charge) > 0:
            a.SetFormalCharge(int(charge))

    return mol


def set_atomic_radicals(
    mol, atoms, atomic_valence_electrons, BO_valences, use_atom_maps=False
):
    """The number of radical electrons = absolute atomic charge."""

    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        if abs(charge) > 0:
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1 :]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs_new(UA, AC, DU, use_graph=True):
    n_ua = 10000
    matching_ids = dict()
    matching_ids2 = dict()
    for i, du in zip(UA, DU):
        if du > 1:
            matching_ids[i] = n_ua
            matching_ids2[n_ua] = i
            n_ua += 1

    bonds = get_bonds(UA, AC)
    for i, j in bonds:
        if i in matching_ids:
            bonds.append(tuple(sorted([matching_ids[i], j])))

        elif j in matching_ids:
            bonds.append(tuple(sorted([i, matching_ids[j]])))

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        UA_pair = UA_pairs[0]

        remove_pairs = []
        add_pairs = []
        for i, j in UA_pair:
            if i in matching_ids2 and j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], matching_ids2[j]]))
            elif i in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], j]))
            elif j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([i, matching_ids2[j]]))

        for p1, p2 in zip(remove_pairs, add_pairs):
            UA_pair.remove(p1)
            UA_pair.append(p2)
        return [UA_pair]

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def get_UA_pairs(UA, AC, use_graph=True):
    bonds = get_bonds(UA, AC)

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def get_sorted_valences_list(valences_list_of_lists, atoms):
    """Generate valence combinations lazily in element order O, N, C, P, S, then others."""
    priority_order = [8, 7, 6, 15, 16]
    groups = {num: [] for num in priority_order}
    others = []

    for i, atomicNum in enumerate(atoms):
        if atomicNum in groups:
            groups[atomicNum].append(i)
        else:
            others.append(i)

    # A flat product keeps valence combinations lazy; nested products expand groups eagerly.
    reordered_indices = []
    for num in priority_order:
        reordered_indices.extend(groups[num])

    reordered_indices.extend(others)

    ordered_valences = [valences_list_of_lists[i] for i in reordered_indices]

    total_expected_combinations = math.prod(len(v) for v in ordered_valences)

    if total_expected_combinations > valence_combinations_limit:
        logger.error(
            f"Search space too large ({total_expected_combinations:,}). "
            "Terminating valence generation to prevent hang."
        )
        return None  # Explicitly return None instead of the generator

    restore_map = [0] * len(atoms)
    for sorted_pos, original_pos in enumerate(reordered_indices):
        restore_map[original_pos] = sorted_pos

    def valence_generator():
        # Lazy -- callers take only the first num_try_limit combinations.
        for combined in itertools.product(*ordered_valences):
            yield tuple(combined[restore_map[i]] for i in range(len(atoms)))

    return valence_generator()


def score_BO(BO, atoms, atomic_valence_electrons, allow_charged_fragments=True):
    """Rank valid bond orders by (total absolute charge, number of charged atoms); lower is better."""
    if not allow_charged_fragments:
        return (0, 0)

    BO_valences = list(BO.sum(axis=1))
    total_abs_charge = 0
    num_charged_atoms = 0
    for i, (atom, BO_valence) in enumerate(zip(atoms, BO_valences)):
        q = get_atomic_charge(
            atom,
            atomic_valence_electrons[atom],
            BO_valence,
            neighbor_atoms=_neighbor_atomic_numbers(BO, i, atoms),
        )
        if q != 0:
            total_abs_charge += abs(q)
            num_charged_atoms += 1

    return (total_abs_charge, num_charged_atoms)


def _nitro_forced_valences(AC, atoms) -> dict[int, list[int]]:
    """Return forced atom valences for nitro/nitrate groups with exactly one N=O bond."""
    forced = {}
    for i, atomicNum in enumerate(atoms):
        if atomicNum != 7:
            continue
        terminal_oxygens = [
            j
            for j, bonded in enumerate(AC[i])
            if bonded and j != i and atoms[j] == 8 and int(sum(AC[j])) == 1
        ]
        if len(terminal_oxygens) < 2:
            continue
        forced[i] = [4]
        forced[terminal_oxygens[0]] = [2]
        for j in terminal_oxygens[1:]:
            forced[j] = [1]
    return forced


def _narrowed_oxygen_valences(valences_list_of_lists, atoms):
    """Drop valence 3 from every oxygen that has another option."""
    narrowed = []
    for atomic_num, options in zip(atoms, valences_list_of_lists):
        if atomic_num == 8 and 3 in options and len(options) > 1:
            options = [valence for valence in options if valence != 3]
        narrowed.append(options)
    return narrowed


def AC2BO(
    AC,
    atoms,
    charge,
    allow_charged_fragments=True,
    use_graph=True,
    allow_carbenes=True,
    diagnostics=None,
):
    """Infer bond orders from adjacency using unsaturated atoms and their valence deficits."""

    valences_list_of_lists = []
    AC_valence = [int(x) for x in AC.sum(axis=1)]

    formula = labels2formula([elemdatabase.elementsym[atom] for atom in atoms])
    wrong = 0

    nitro_forced = _nitro_forced_valences(AC, atoms)

    for i, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        # valence can't be smaller than number of neighbours
        if len(atomic_valence[atomicNum]) == 0:
            logger.warning(
                "AC2BO: atom index %d (atomic number %d) has no possible valences "
                "defined in the database.",
                i,
                atomicNum,
            )
        possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]

        if atomicNum == 6 and valence == 1:
            if 2 in possible_valence:
                possible_valence.remove(2)
        if atomicNum == 6 and not allow_carbenes and valence == 2:
            if 2 in possible_valence:
                possible_valence.remove(2)
        if atomicNum == 6 and valence == 2:
            possible_valence.append(3)
        if atomicNum == 7:
            if valence not in possible_valence:
                possible_valence.append(valence)
        if atomicNum != 6 and allow_charged_fragments:
            # Hardcoded valences can omit charged states such as I(2) or P(4).
            if valence not in possible_valence:
                possible_valence.append(valence)
        if i in nitro_forced:
            possible_valence = list(nitro_forced[i])
        if atomicNum == 16 and valence == 1 and formula == "C-S":
            possible_valence = [3]
        if atomicNum == 34 and valence == 1 and formula == "C-Se":
            possible_valence = [3]
        if atomicNum == 52 and valence == 1 and formula == "C-Te":
            possible_valence = [3]
        if len(possible_valence) == 0:
            element = elemdatabase.elementsym[atomicNum]
            max_valence = max(atomic_valence[atomicNum])

            if elemdatabase.elementgroup[element] in (1, 2):
                # Alkali and alkaline earth metals
                logger.warning(
                    "  Atom %s (index %d) has valence %d, which exceeds the allowed maximum (%d) "
                    "for group %d elements.",
                    element,
                    i,
                    valence,
                    max_valence,
                    elemdatabase.elementgroup[element],
                )
                possible_valence.append(valence)

            elif elemdatabase.elementperiod[element] < 3:
                logger.warning(
                    "  Atom %s (index %d) has valence %d, which exceeds the allowed maximum (%d) "
                    "for period %d elements.",
                    element,
                    i,
                    valence,
                    max_valence,
                    elemdatabase.elementperiod[element],
                )
                possible_valence.append(max_valence)
            else:
                possible_valence.append(valence)
        valences_list_of_lists.append(possible_valence)
    if wrong > 0:
        return None, atomic_valence_electrons

    # Prune oxygen valences before limiting the search.
    projected = math.prod(len(options) for options in valences_list_of_lists)
    if projected > valence_combinations_limit and 8 in atoms:
        narrowed = _narrowed_oxygen_valences(valences_list_of_lists, atoms)
        narrowed_total = math.prod(len(options) for options in narrowed)
        if narrowed_total < projected:
            logger.warning(
                "Valence search space for %s is %d, over the %d limit; "
                "restricting oxygen to [2, 1] brings it to %d.",
                formula,
                projected,
                valence_combinations_limit,
                narrowed_total,
            )
            valences_list_of_lists = narrowed

    best_BO = AC.copy()
    best_status_BO = None
    best_status_score = None  # holds (total_abs_charge, num_charged_atoms) tuples

    sorted_gen = get_sorted_valences_list(valences_list_of_lists, atoms)

    if sorted_gen is None:
        logger.warning("AC2BO terminating: Valence search space exceeded limit.")
        # Record charge-independent search limits so callers can report skipped species.
        if diagnostics is not None:
            diagnostics["valence_search_too_large"] = True
        return None, None

    # Limit enumeration without materializing all combinations.
    count = 0
    max_count = num_try_limit
    top_valences = list(itertools.islice(sorted_gen, max_count))

    for count, valences in enumerate(top_valences, 1):
        UA, DU_from_AC = get_UA(valences, AC_valence)

        check_len = len(UA) == 0

        if check_len:
            check_bo = BO_is_OK(
                AC,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs_new(UA, AC, DU_from_AC, use_graph=use_graph)

        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )
            charge_OK = charge_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
                allow_carbenes=allow_carbenes,
            )

            if status:
                score = score_BO(
                    BO,
                    atoms,
                    atomic_valence_electrons,
                    allow_charged_fragments=allow_charged_fragments,
                )
                if best_status_score is None or score < best_status_score:
                    best_status_BO = BO.copy()
                    best_status_score = score
                    logger.debug(
                        "  formula=%s status=%s charge=%s count=%s score=%s (new best)",
                        formula,
                        status,
                        charge,
                        count,
                        score,
                    )
                # Stop when all necessary charge sits on one atom.
                if best_status_score == (abs(charge), 1) or best_status_score == (0, 0):
                    logger.debug(
                        "  formula=%s charge=%s count=%s reached minimal charge "
                        "spread %s, stopping early",
                        formula,
                        charge,
                        count,
                        best_status_score,
                    )
                    return best_status_BO, atomic_valence_electrons
            elif (
                BO.sum() >= best_BO.sum()
                and valences_not_too_large(BO, valences)
                and charge_OK
            ):
                best_BO = BO.copy()

            count += 1
            if count > max_count:
                if best_status_BO is not None:
                    logger.debug(
                        "  reached max count %s (charge=%d) count: %d, "
                        "returning best-scored status BO (score=%s)",
                        formula,
                        charge,
                        count,
                        best_status_score,
                    )
                    return best_status_BO, atomic_valence_electrons
                logger.debug(
                    "  reached max count %s (charge=%d) count: %d",
                    formula,
                    charge,
                    count,
                )
                return best_BO, atomic_valence_electrons

    if best_status_BO is not None:
        return best_status_BO, atomic_valence_electrons

    return best_BO, atomic_valence_electrons


def AC2mol(
    mol,
    AC,
    atoms,
    charge,
    allow_charged_fragments=True,
    use_graph=True,
    allow_carbenes=True,
    diagnostics=None,
):
    """Build mol from AC; temporarily override atomic_valence without persisting changes."""
    overrides = None
    if not allow_charged_fragments:
        overrides = {8: [2, 1], 7: [3, 2], 6: [4, 2]}

    backup = copy.deepcopy(atomic_valence)

    try:
        if overrides is not None:
            for k, v in overrides.items():
                atomic_valence[k] = list(v)  # copy to avoid sharing caller’s list

        BO, atomic_valence_electrons = AC2BO(
            AC,
            atoms,
            charge,
            allow_charged_fragments=allow_charged_fragments,
            use_graph=use_graph,
            allow_carbenes=allow_carbenes,
            diagnostics=diagnostics,
        )
        if BO is None:
            return [], None

        mol = BO2mol(
            mol,
            BO,
            atoms,
            atomic_valence_electrons,
            charge,
            allow_charged_fragments=allow_charged_fragments,
        )
        return [mol], BO

    finally:
        atomic_valence.clear()
        atomic_valence.update(backup)


def get_proto_mol(atoms):
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(int(atoms[i]))
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def read_xyz_file(filename, look_for_charge=True):
    atomic_symbols = []
    xyz_coordinates = []
    charge = 0

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                continue  # atom count, recovered from the coordinate lines
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates


def chiral_stereo_check(mol):
    """
    Find and embed chiral information into the model based on the coordinates

    Args:
        mol - rdkit molecule, with embeded conformer

    """
    try:
        Chem.SanitizeMol(mol)
        Chem.DetectBondStereochemistry(mol, -1)
        Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
        Chem.AssignAtomChiralTagsFromStructure(mol, -1)
        return True
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                catchErrors=True,
            )
            Chem.DetectBondStereochemistry(mol, -1)
            Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
            Chem.AssignAtomChiralTagsFromStructure(mol, -1)
            return True

        except Chem.rdchem.AtomValenceException as e:
            logger.warning("Failed to process molecule: %s", e)
            return False


def xyz2mol(
    atoms,
    coordinates,
    AC,
    covalent_factor,
    charge=0,
    allow_charged_fragments=True,
    use_graph=True,
    use_huckel=False,
    embed_chiral=True,
    exportBO=False,
    allow_carbenes=False,
):
    """Return RDKit molecules from atomic numbers, coordinates, adjacency and charge.

    Optionally export bond orders and allow charged fragments or carbenes.
    """
    ac = np.array(AC)
    mol = get_proto_mol(atoms)
    new_mols, BO = AC2mol(
        mol,
        ac,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
        allow_carbenes=False,
    )

    if exportBO:
        return new_mols, BO
    else:
        return new_mols
