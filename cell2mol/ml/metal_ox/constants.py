"""Vocabularies and label mapping shared by every variant (cell2mol-free)."""

ATOM_TYPES = [
    "O", "N", "C", "S", "I", "Cl", "P", "Br", "H",
    "As", "Sb", "F", "Se", "Te", "Si", "B", "D",
]

METAL_TYPES = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
]

OX_STATES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
OX_TO_CLASS = {ox: i for i, ox in enumerate(OX_STATES)}
CLASS_TO_OX = {i: ox for ox, i in OX_TO_CLASS.items()}
