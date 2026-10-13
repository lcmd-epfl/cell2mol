"""Metal oxidation state tables and candidate rules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from cell2mol.classes.metal import Metal
    from cell2mol.classes.molecule import Molecule
logger = logging.getLogger(__name__)

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


def get_plausible_metal_os(metal: Metal) -> list[int]:
    """Common oxidation states for a metal atom.

    Source: Venkataraman et al., J. Chem. Educ. 1997, 74, 915.
    """

    mol = cast("Molecule", metal.get_parent("molecule"))
    if mol.is_haptic is None:
        mol.update_hapticity()

    metal_os = METAL_OXIDATION_STATES[metal.label]

    # Allow 0 oxidation state for selected metals under specific conditions
    zero_os_metals = {"Fe", "Ni", "Ru"}
    if metal.label in zero_os_metals:
        has_CO = any(lig.formula == "C-O" for lig in mol.ligands or [])
        if (has_CO or mol.is_haptic) and 0 not in metal_os:
            metal_os.append(0)

    return metal_os
