# Fullerene species that require special handling
FULLERENES = {"C60", "C72", "C80"}
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
}


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


def aromatic_info(mol: object, added_indices=None):
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
