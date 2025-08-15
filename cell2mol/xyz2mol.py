#!/usr/bin/env python

#################
#### XYZ2MOL ####
#################

import copy
import itertools
import rdkit
from rdkit.Chem import rdmolops
from rdkit.Chem import rdchem

try:
    from rdkit.Chem import rdEHTTools  # requires RDKit 2019.9.1 or later
except ImportError:
    rdEHTTools = None

from collections import defaultdict

import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from cell2mol.elementdata import ElementData
from cell2mol.connectivity import labels2formula
elemdatabase = ElementData()

###############################
#### RUBEN Changes for V14 ####
###############################
valence_electrons = []
for i in elemdatabase.elementsym:
    try:
        valence_electrons.append(
            elemdatabase.valenceelectrons[elemdatabase.elementsym[i]]
        )
    except KeyError:
        continue

atomic_valence_electrons = dict(zip(elemdatabase.elementsym, valence_electrons))


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
    if k == 16 : # S
        return [2, 4, 6]
    if k ==32 :  # Ge
        return [4, 6]
    if k == 33:  # As
        return [5, 3]  # [5,4,3]
    if k == 51:  # Sb
        return [6, 5, 3]  # [5,4,3]
    if k == 52: # Te
        return [2, 4, 6]
    if group == 17: # F, Cl, Br, I
        return [1]
        # return [1, 2]   # Cl [1, 7]
    if block == "s" and period == 1:
        av = 2 - ave
    elif group == 1 and period != 1:
        av = 0
    elif group == 2:
        av = 0
    elif group == 18:
        av = 0
    elif block == "p" and group != 18:
        av = 8 - ave
    elif block in "d":
        av = 20 
    else:
        av = 1
    return [av]


atomic_valence = defaultdict(list)
for k in elemdatabase.elementsym:
    try:
        # print(f"Getting atomic valence for element number {k}={elemdatabase.elementsym[k]}")
        av = get_atomic_valences(k)
        atomic_valence[k].extend(av)
        # print(f"Got atomic valence {atomic_valence[k]} as a result.")
    except KeyError:
        continue
# atomic_valence = defaultdict(list)

# atomic_valence[1] = [1]
# atomic_valence[5] = [3, 4]
# atomic_valence[6] = [4, 2]
# atomic_valence[7] = [3, 4]
# atomic_valence[8] = [2, 1, 3]  # [2,1,3]
# atomic_valence[9] = [1, 7]
# atomic_valence[13] = [3, 4] 
# atomic_valence[14] = [4]
# atomic_valence[15] = [3, 5]  # [5,4,3]
# atomic_valence[16] = [2, 4, 6]  # [6,3,2]
# atomic_valence[17] = [1, 7]
# atomic_valence[18] = [0]
# atomic_valence[32] = [4]
# atomic_valence[33] = [5, 3]
# atomic_valence[35] = [1, 7]
# atomic_valence[34] = [2]
# atomic_valence[52] = [2]
# atomic_valence[53] = [1, 7]



# print(f"XYZ2MOL: {atomic_valence=}")
# print(f"XYZ2MOL: {atomic_valence_electrons=}")
###############################


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
    """ """
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """ """
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        #UA_pairs = get_UA_pairs(UA, AC, use_graph=use_graph)[0]
        UA_pairs = get_UA_pairs_new(UA, AC, DU, use_graph=use_graph)[0]

    return BO


def valences_not_too_large(BO, valences):
    """ """
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
    # total charge
    Q = 0

    # charge fragment list
    q_list = []

    if allow_charged_fragments:

        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if not allow_carbenes and number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                    print("\t\tCarbenes are not allowed in this molecule")
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)
    #print("charge_is_OK: Q", Q, "charge", charge)
    return charge == Q


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
    """
    Sanity of bond-orders

    args:
        BO -
        AC -
        charge -
        DU -


    optional
        allow_charges_fragments -


    returns:
        boolean - true of molecule is OK, false if not
    """

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
        allow_carbenes=True
    )

    if check_charge and check_sum:
        return True

    return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    """ """
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
    # elif atom == 6 and BO_valence == 2 and not found:
    #     charge = 0
    #     found = True
    # elif atom == 13 and not found and not found:
    #     charge = 3 - BO_valence
    #     found = True 
    # Ionic Bonds are not correctly captured, exceptions are needed for atoms with tendency to form them
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
    elif atom == 33 and BO_valence == 6 and not found:  # AsX6
        charge = -1
        found = True
    elif atom == 51 and BO_valence == 6 and not found:  # SbX6
        charge = -1
        found = True
    elif atom == 50 and BO_valence == 4 and not found:  # SnX4
        charge = 0 
        found = True

    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def clean_charges(mol):
    """
    This hack should not be needed anymore, but is kept just in case

    """

    Chem.SanitizeMol(mol)
    # rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
    #              '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
    #              '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
    #              '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
    #              '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
    #              '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    rxn_smarts = [
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][CX3-,NX3-:5][#6,#7:6]1=[#6,#7:7]>>"
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][-0,-0:5]=[#6,#7:6]1[#6-,#7-:7]",
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3](=[#6,#7:4])[#6,#7:5]=[#6,#7:6][CX3-,NX3-:7]1>>"
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]([#6-,#7-:4])=[#6,#7:5][#6,#7:6]=[-0,-0:7]1",
    ]

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(
                smarts.split(">>")[0]
            )  # Construct a molecule from a SMARTS string.
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
                Chem.SanitizeMol(fragment)
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol, fragment)

    return mol


def BO2mol(
    mol,
    BO_matrix,
    atoms,
    atomic_valence_electrons,
    mol_charge,
    allow_charged_fragments=True,
):
    """
    based on code written by Paolo Toscani

    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.

    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule

    optional:
        allow_charged_fragments - bool - allow charged fragments

    returns
        mol - updated rdkit molecule with bond connectivity

    """

    l = len(BO_matrix)
    l2 = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))
    # print("BO_valences", BO_valences)  #Sergi

    if l != l2:
        raise RuntimeError(
            "sizes of adjMat ({0:d}) and Atoms {1:d} differ".format(l, l2)
        )

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    for i in range(l):
        for j in range(i + 1, l):
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
    """ """
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                q += 1
                charge = 0
            # if BO_valences[i] == 2:
            #     print("set_atomic_charges", "carbon atom SetNumRadicalElectrons 2")
            #     a.SetNumRadicalElectrons(2)
            #     charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if abs(charge) > 0:
            a.SetFormalCharge(int(charge))

    # mol = clean_charges(mol)

    return mol


# def set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences):
#     """

#     The number of radical electrons = absolute atomic charge

#     """
#     for i, atom in enumerate(atoms):
#         a = mol.GetAtomWithIdx(i)
#         charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

#         if abs(charge) > 0:
#             a.SetNumRadicalElectrons(abs(int(charge)))

#     return mol

def set_atomic_radicals(
    mol, atoms, atomic_valence_electrons, BO_valences, use_atom_maps=False
):
    """The number of radical electrons = absolute atomic charge."""
    atomic_valence[8] = [2, 1]
    atomic_valence[7] = [3, 2]
    atomic_valence[6] = [4, 2]

    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        if use_atom_maps:
            a.SetAtomMapNum(i + 1)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        if abs(charge) > 0:
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol

def get_bonds(UA, AC):
    """ """
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1 :]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds

def get_UA_pairs_new(UA, AC, DU, use_graph=True):
    """"""
    N_UA = 10000
    matching_ids = dict()
    matching_ids2 = dict()
    for i, du in zip(UA, DU):
        if du > 1:
            matching_ids[i] = N_UA
            matching_ids2[N_UA] = i
            N_UA += 1

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
                # UA_pair.remove(tuple([i,j]))
                # UA_pair.append(tuple([matching_ids2[i], matching_ids2[j]]))
            elif i in matching_ids2:
                # UA_pair.remove(tuple([i,j]))
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([matching_ids2[i], j]))
                # UA_pair.append(tuple([matching_ids2[i],j]))
            elif j in matching_ids2:
                remove_pairs.append(tuple([i, j]))
                add_pairs.append(tuple([i, matching_ids2[j]]))

                # UA_pair.remove(tuple([i,j]))
                # UA_pair.append(tuple([i,matching_ids2[j]]))
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
    """ """

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
    
    valences_list = itertools.product(*valences_list_of_lists)
    O_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 8
    ]
    N_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 7
    ]
    C_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 6
    ]
    P_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 15
    ]
    S_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 16
    ]

    O_sums = []
    for v_list in itertools.product(*O_valences):
        O_sums.append(v_list)
        # if sum(v_list) not in O_sums:
        #    O_sums.append(v_list))

    N_sums = []
    for v_list in itertools.product(*N_valences):
        N_sums.append(v_list)
        # if sum(v_list) not in N_sums:
        #    N_sums.append(sum(v_list))

    C_sums = []
    for v_list in itertools.product(*C_valences):
        C_sums.append(v_list)
        # if sum(v_list) not in C_sums:
        #    C_sums.append(sum(v_list))

    P_sums = []
    for v_list in itertools.product(*P_valences):
        P_sums.append(v_list)

    S_sums = []
    for v_list in itertools.product(*S_valences):
        S_sums.append(v_list)

    order_dict = dict()
    for i, v_list in enumerate(
        itertools.product(*[O_sums, N_sums, C_sums, P_sums, S_sums])
    ):
        order_dict[v_list] = i

    valence_order_list = []
    for valence_list in valences_list:
        C_sum = []
        N_sum = []
        O_sum = []
        P_sum = []
        S_sum = []
        for v, atomicNum in zip(valence_list, atoms):
            if atomicNum == 6:
                C_sum.append(v)
            if atomicNum == 7:
                N_sum.append(v)
            if atomicNum == 8:
                O_sum.append(v)
            if atomicNum == 15:
                P_sum.append(v)
            if atomicNum == 16:
                S_sum.append(v)

        order_idx = order_dict[
            (tuple(O_sum), tuple(N_sum), tuple(C_sum), tuple(P_sum), tuple(S_sum))
        ]
        valence_order_list.append(order_idx)

    sorted_valences_list = [
        y
        for x, y in sorted(
            zip(valence_order_list, list(itertools.product(*valences_list_of_lists)))
        )
    ]

    return sorted_valences_list

def AC2BO_new(
    AC, atoms, charge, allow_charged_fragments=True, use_graph=True, allow_carbenes=True
):
    """Implemenation of algorithm shown in Figure 2.

    UA: unsaturated atoms

    DU: degree of unsaturation (u matrix in Figure)

    best_BO: Bcurr in Figure
    """

    global atomic_valence
    global atomic_valence_electrons

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    # AC_valence = list(AC.sum(axis=1))
    AC_valence = [int(x) for x in AC.sum(axis=1)]
    #print("Atom labels:", [elemdatabase.elementsym[atom] for atom in atoms])
    #print(f"{AC_valence=}")

    for i, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        # valence can't be smaller than number of neighbourgs
        possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
        if atomicNum == 6 and valence == 1:
            possible_valence.remove(2)
        if atomicNum == 6 and not allow_carbenes and valence == 2:
            possible_valence.remove(2)
        if atomicNum == 6 and valence == 2:
            possible_valence.append(3)
        if atomicNum == 16 and valence == 1:
            possible_valence = [1, 2]

        if not possible_valence:
            print(
                "Valence of atom",
                i,
                "is",
                valence,
                "which bigger than allowed max",
                max(atomic_valence[atomicNum]),
                ". Stopping",
            )
            #possible_valence.append(valence) # this is the added line
        valences_list_of_lists.append(possible_valence)

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = itertools.product(*valences_list_of_lists)

    best_BO = AC.copy()

    O_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 8
    ]
    N_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 7
    ]
    C_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 6
    ]
    P_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 15
    ]
    S_valences = [
        v_list
        for v_list, atomicNum in zip(valences_list_of_lists, atoms)
        if atomicNum == 16
    ]

    O_sums = []
    for v_list in itertools.product(*O_valences):
        O_sums.append(v_list)
        # if sum(v_list) not in O_sums:
        #    O_sums.append(v_list))

    N_sums = []
    for v_list in itertools.product(*N_valences):
        N_sums.append(v_list)
        # if sum(v_list) not in N_sums:
        #    N_sums.append(sum(v_list))

    C_sums = []
    for v_list in itertools.product(*C_valences):
        C_sums.append(v_list)
        # if sum(v_list) not in C_sums:
        #    C_sums.append(sum(v_list))

    P_sums = []
    for v_list in itertools.product(*P_valences):
        P_sums.append(v_list)

    S_sums = []
    for v_list in itertools.product(*S_valences):
        S_sums.append(v_list)

    order_dict = dict()
    for i, v_list in enumerate(
        itertools.product(*[O_sums, N_sums, C_sums, P_sums, S_sums])
    ):
        order_dict[v_list] = i

    valence_order_list = []
    for valence_list in valences_list:
        C_sum = []
        N_sum = []
        O_sum = []
        P_sum = []
        S_sum = []
        for v, atomicNum in zip(valence_list, atoms):
            if atomicNum == 6:
                C_sum.append(v)
            if atomicNum == 7:
                N_sum.append(v)
            if atomicNum == 8:
                O_sum.append(v)
            if atomicNum == 15:
                P_sum.append(v)
            if atomicNum == 16:
                S_sum.append(v)
        # print("AC2BO_new: O_sum", O_sum, "N_sum", N_sum, "C_sum", C_sum, "P_sum", P_sum, "S_sum", S_sum)
        order_idx = order_dict[
            (tuple(O_sum), tuple(N_sum), tuple(C_sum), tuple(P_sum), tuple(S_sum))
        ]
        valence_order_list.append(order_idx)

    sorted_valences_list = [
        y
        for x, y in sorted(
            zip(valence_order_list, list(itertools.product(*valences_list_of_lists)))
        )
    ]
    print("\tAC2BO_new: sorted_valences_list", len(sorted_valences_list))
    max_count = min(200, int(len(sorted_valences_list)*0.3))
    #print(f"AC2BO_new: {len(sorted_valences_list)=} {max_count=}")

    for valences in sorted_valences_list[:max_count]:  # valences_list:
        # print(f"\tSending", valences, AC_valence, "to get_UA")
        UA, DU_from_AC = get_UA(valences, AC_valence)
        #print(f"\tAC2BO_new: {UA=}, {DU_from_AC=}")
        # check_len = len(UA) == 0
        # if check_len:
        #     check_bo = BO_is_OK(
        #         AC,
        #         AC,
        #         charge,
        #         DU_from_AC,
        #         atomic_valence_electrons,
        #         atoms,
        #         valences,
        #         allow_charged_fragments=allow_charged_fragments,
        #         allow_carbenes=allow_carbenes,
        #     )
        # else:
        #     check_bo = None

        # if check_len and check_bo:
        #     return AC, atomic_valence_electrons

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
                return BO, atomic_valence_electrons
            elif (
                BO.sum() >= best_BO.sum()
                and valences_not_too_large(BO, valences)
                and charge_OK
            ):
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons


def AC2BO (AC, atoms, charge, allow_charged_fragments=True, use_graph=True, allow_carbenes=True):
    """
    implemenation of algorithm shown in Figure 2
    UA: unsaturated atoms
    DU: degree of unsaturation (u matrix in Figure)
    best_BO: Bcurr in Figure
    """

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    # AC_valence = list(AC.sum(axis=1))
    AC_valence = [int(x) for x in AC.sum(axis=1)]
    #print("Atom labels:", [elemdatabase.elementsym[atom] for atom in atoms])
    #print(f"{AC_valence=}")
    formula = labels2formula([elemdatabase.elementsym[atom] for atom in atoms])    
    wrong = 0

    for i, (atomicNum, valence) in enumerate(zip(atoms, AC_valence)):
        # valence can't be smaller than number of neighbours
        if len(atomic_valence[atomicNum]) == 0:
            print(
                "In AC2BO",
                i,
                "Atomic number",
                atomicNum,
                "Has no possible valences assigned in database",
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
            #print("Possible valences for:", atomicNum,"are",possible_valence, valence)
            if valence not in possible_valence:
                possible_valence.append(valence)
        # if atomicNum == 15:
        #    print("Possible valences for:", atomicNum,"are",possible_valence, valence)
        if len(possible_valence) == 0:
            element = elemdatabase.elementsym[atomicNum]
            if elemdatabase.elementgroup[element] == 1 or elemdatabase.elementgroup[element] == 2 : # Alkali and Alkaline earth metals
                print('WARNING!! Valence of atom', element, i,\
                    'is', valence,'which is bigger than allowed max',max(atomic_valence[atomicNum]),'. Stopping')
                possible_valence.append(valence)
            elif elemdatabase.elementperiod[element] < 3 : # e.g. HOLMOK
                print('WARNING!! Valence of atom', element, i,\
                    'is', valence,'which bigger than allowed max',max(atomic_valence[atomicNum]),'. Stopping')
                possible_valence.append(valence)
                # wrong += 1
            else:
                possible_valence.append(valence)
            # sys.exit()
        valences_list_of_lists.append(possible_valence)
    #print(f"{wrong=}")
    #print(f"\tAC2BO: {formula=} {len(valences_list_of_lists)=} {[vl for vl in valences_list_of_lists]=}")
    if wrong > 0:
        # print(f"AC2BO: {wrong=}")
        return None, atomic_valence_electrons
    
    #print(f"\tAC2BO: {valences_list_of_lists=}")
    
    # convert [[4],[2,1]] to [[4,2],[4,1]]
    # valences_list = []
    # for i in itertools.product(*valences_list_of_lists):
    #     tmp = []
    #     for j in i:
    #         tmp.append(j)
    #     valences_list.append(tmp)

    best_BO = AC.copy()
    BO_is_OK_list = []
    sorted_valences_list = get_sorted_valences_list(valences_list_of_lists, atoms)

    # max_count = max(1000, int(len(valences_list)*0.3))
    # #print(f"AC2BO: {formula=} {len(valences_list)=} {max_count=}")

    # count = 0
    # for valences in valences_list:

    count = 0
    max_count = min(len(sorted_valences_list), 50)
    #print(f"\tAC2BO: {sorted_valences_list=}")
    print(f"\tAC2BO: {formula=} {len(sorted_valences_list)=} {max_count=}")
    # if len(sorted_valences_list) > 1000:
    #     return None, atomic_valence_electrons
    for valences in sorted_valences_list:  # valences_list:
        UA, DU_from_AC = get_UA(valences, AC_valence)

        check_len = len(UA) == 0
        #print (f"\tAC2BO: check_len", check_len)
        #print(f"\tUA", UA)
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
            #print(f"\tAC2BO: {formula=} return AC", check_len, check_bo, f"{charge=} {count=}")
            return AC, atomic_valence_electrons
        
        UA_pairs_list = get_UA_pairs(UA, AC, use_graph=use_graph)
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
                print(f"\tAC2BO: {formula=} status", status, f"{charge=} {count=}")
                return BO, atomic_valence_electrons
            elif (
                BO.sum() >= best_BO.sum()
                and valences_not_too_large(BO, valences)
                and charge_OK
            ):
                #print(f"\tAC2BO: status", status, "BO.sum()", BO.sum(), "best_BO.sum()", best_BO.sum())
                best_BO = BO.copy()
            
            count += 1
            if count > max_count :
                print(f"\tOver maximum counts AC2BO: {formula=} {charge=} {count=}")
                return best_BO, atomic_valence_electrons
            
    return best_BO, atomic_valence_electrons


def AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_graph=True, allow_carbenes=True):
    """ """

    # convert AC matrix to bond order (BO) matrix
    # BO, atomic_valence_electrons = AC2BO_new(
    BO, atomic_valence_electrons = AC2BO(
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
        allow_carbenes=allow_carbenes,
    )
    if BO is None:
        return [], None
    
    # add BO connectivity and charge info to mol object
    mol = BO2mol(
        mol,
        BO,
        atoms,
        atomic_valence_electrons,
        charge,
        allow_charged_fragments=allow_charged_fragments,
    )

    # If charge is not correct don't return mol
    #     if Chem.GetFormalCharge(mol) != charge:    ## SERGI MOD
    #         return []                              ## SERGI MOD

    # BO2mol returns an arbitrary resonance form. Let's make the rest
    # mols = rdchem.ResonanceMolSupplier(mol, Chem.UNCONSTRAINED_CATIONS, Chem.UNCONSTRAINED_ANIONS)
    # mols = [mol for mol in mols]

    return [mol], BO


def get_proto_mol(atoms):
    """ """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        #print(f"XYZ2MOL.PROTO_MOL: doing {atoms[i]=}")
        a = Chem.Atom(int(atoms[i]))
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def read_xyz_file(filename, look_for_charge=True):
    """ """

    atomic_symbols = []
    xyz_coordinates = []
    charge = 0
    title = ""

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                title = line
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates


#def xyz2AC(atoms, xyz, charge, covalent_factor, use_huckel=False):
#    """
#
#    atoms and coordinates to atom connectivity (AC)
#
#    args:
#        atoms - int atom types
#        xyz - coordinates
#        charge - molecule charge
#
#    optional:
#        use_huckel - Use Huckel method for atom connecitivty
#
#    returns
#        ac - atom connectivity matrix
#        mol - rdkit molecule
#
#    """
#
#    if use_huckel:
#        return xyz2AC_huckel(atoms, xyz, charge)
#    else:
#        return xyz2AC_vdW(atoms, xyz, covalent_factor)
#
#
#def xyz2AC_vdW(atoms, xyz, covalent_factor):
#
#    # Get mol template
#    mol = get_proto_mol(atoms)
#
#    # Set coordinates
#    conf = Chem.Conformer(mol.GetNumAtoms())
#    for i in range(mol.GetNumAtoms()):
#        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
#    mol.AddConformer(conf)
#
#    AC = get_AC(mol, covalent_factor)
#
#    return AC, mol
#
#
#def get_AC(mol, covalent_factor=1.3):
#    """
#
#    Generate adjacent matrix from atoms and coordinates.
#
#    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0 is not
#
#
#    covalent_factor - 1.3 is an arbitrary factor
#
#    args:
#        mol - rdkit molobj with 3D conformer
#
#    optional
#        covalent_factor - increase covalent bond length threshold with facto
#
#    returns:
#        AC - adjacent matrix
#
#    """
#
#    # Calculate distance matrix
#    dMat = Chem.Get3DDistanceMatrix(mol)
#
#    pt = Chem.GetPeriodicTable()
#    num_atoms = mol.GetNumAtoms()
#    AC = np.zeros((num_atoms, num_atoms), dtype=int)
#
#    for i in range(num_atoms):
#        a_i = mol.GetAtomWithIdx(i)
#        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * covalent_factor
#        for j in range(i + 1, num_atoms):
#            a_j = mol.GetAtomWithIdx(j)
#            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * covalent_factor
#            if dMat[i, j] <= Rcov_i + Rcov_j:
#                AC[i, j] = 1
#                AC[j, i] = 1
#
#    return AC
#
#
## http://rdkit.blogspot.com/2019/06/doing-extended-hueckel-calculations.html
#
#
#def xyz2AC_huckel(atomicNumList, xyz, charge):
#    """
#
#    args
#        atomicNumList - atom type list
#        xyz - coordinates
#        charge - molecule charge
#
#    returns
#        ac - atom connectivity
#        mol - rdkit molecule
#
#    """
#    mol = get_proto_mol(atomicNumList)
#
#    conf = Chem.Conformer(mol.GetNumAtoms())
#    for i in range(mol.GetNumAtoms()):
#        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
#    mol.AddConformer(conf)
#
#    num_atoms = len(atomicNumList)
#    AC = np.zeros((num_atoms, num_atoms)).astype(int)
#
#    mol_huckel = Chem.Mol(mol)
#    mol_huckel.GetAtomWithIdx(0).SetFormalCharge(
#        charge
#    )  # mol charge arbitrarily added to 1st atom
#
#    passed, result = rdEHTTools.RunMol(mol_huckel)
#    opop = (
#        result.GetReducedOverlapPopulationMatrix()
#    )  # The reduced overlap population matrix provides the Mulliken overlap population between atoms in the molecule. It's returned as a vector representing a symmetric matrix
#    tri = np.zeros((num_atoms, num_atoms))
#    tri[
#        np.tril(np.ones((num_atoms, num_atoms), dtype=bool))
#    ] = opop  # lower triangular to square matrix
#    for i in range(num_atoms):
#        for j in range(i + 1, num_atoms):
#            pair_pop = abs(tri[j, i])
#            if pair_pop >= 0.15:  # arbitry cutoff for bond. May need adjustment
#                AC[i, j] = 1
#                AC[j, i] = 1
#
#    return AC, mol


def chiral_stereo_check(mol):
    """
    Find and embed chiral information into the model based on the coordinates

    args:
        mol - rdkit molecule, with embeded conformer

    """
    try:
        #Chem.SanitizeMol(mol)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, 
                         catchErrors=True)
        Chem.DetectBondStereochemistry(mol, -1)
        Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
        Chem.AssignAtomChiralTagsFromStructure(mol, -1)
        return True
    
    except rdkit.Chem.rdchem.AtomValenceException as e:
        print(f"Failed to process molecule: {e}")
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
    allow_carbenes=False
):
    """
    Generate a rdkit molobj from atoms, coordinates and a total_charge.

    args:
        atoms - list of atom types (int)
        coordinates - 3xN Cartesian coordinates
        charge - total charge of the system (default: 0)

    optional:
        allow_charged_fragments - alternatively radicals are made
        use_graph - use graph (networkx)
        use_huckel - Use Huckel method for atom connectivity prediction
        embed_chiral - embed chiral information to the molecule

    returns:
        mols - list of rdkit molobjects

    """
    AC = np.array(AC)
    mol = get_proto_mol(atoms)
    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    #AC, mol = xyz2AC(atoms, coordinates, charge, covalent_factor, use_huckel=use_huckel)
    # Convert AC to bond order matrix and add connectivity and charge info to
    # mol object
    new_mols, BO = AC2mol(
        mol,
        AC,
        atoms,
        charge,
        allow_charged_fragments=allow_charged_fragments,
        use_graph=use_graph,
        allow_carbenes=False
    )

    # Check for stereocenters and chiral centers -> Move to get_charge function
    # if embed_chiral:
    #     chiral_stereo_check(new_mol))

    if exportBO:
        return new_mols, BO
    else:
        return new_mols

########################
#### END OF XYZ2MOL ####
########################
