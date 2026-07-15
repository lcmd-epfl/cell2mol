import numpy as np
from cell2mol.elementdata import ElementData
from cell2mol.my_types import NDArray 
elemdatabase = ElementData()

TRANSITION_METALS = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}

ALKALI_METALS = {"Li", "Na", "K", "Rb", "Cs", "Fr"}

ALKALINE_EARTH_METALS = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}

ALKALI_AND_ALKALINE_EARTH_METALS = ALKALI_METALS | ALKALINE_EARTH_METALS

LANTHANIDES = {
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
}

ACTINIDES = {
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
}

POST_TRANSITION_METALS = {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po", "At"}

METALLOIDS = {"B", "Si", "Ge", "As", "Sb", "Te"}

HAPTIC_PRETTY = {
    # eta2
    "eta2(C,C)": "η²-C,C",
    # eta3
    "eta3(C,C,C)": "η³-C,C,C",
    # eta4
    "eta4(C,C,C,C)": "η⁴-C₄",
    "eta4(C,C,C,O)": "η⁴-C₃O",
    # eta5
    "eta5(Cp)": "η⁵-Cp",
    "eta5(C5)": "η⁵-C₅",
    "eta5(As5)": "η⁵-As₅",
    "eta5(P5)": "η⁵-P₅",
    # eta6+
    "eta6(C6)": "η⁶-C₆",
    "eta6(benzene)": "η⁶-benzene",
    "eta7(C7)": "η⁷-C₇",
    "eta7(CHT)": "η⁷-cycloheptatrienyl",
    "eta8(C8)": "η⁸-C₈",
    "eta8(COT)": "η⁸-cyclooctatetraenyl",
    "eta5,5(Cp,Cp)": "η⁵,η⁵-Cp,Cp",
}


def labels2formula(labels: list[str]):
    elems = elemdatabase.elementnr.keys()
    formula = []
    for z in elems:
        nz = list(labels).count(z)
        if nz > 1:
            formula.append(f"{z}{nz}-")
        if nz == 1:
            formula.append(f"{z}-")
    formula = "".join(formula)[:-1]
    return formula


def labels2ratio(labels: list[str]):
    elems = elemdatabase.elementnr.keys()
    ratio = []
    for z in elems:
        nz = list(labels).count(z)
        if nz > 0:
            ratio.append(nz)
    return ratio


def labels2electrons(labels: list[str]):
    if isinstance(labels, list):
        eleccount = 0
        for label in labels:
            eleccount += elemdatabase.elementnr[label]
    elif isinstance(labels, str):
        eleccount = elemdatabase.elementnr[labels]
    return eleccount


def get_metal_idxs(labels: list[str]) -> list[int]:
    """Transition metals, lanthanides, and actinides."""
    d_f_metals = TRANSITION_METALS | LANTHANIDES | ACTINIDES
    return [i for i, label in enumerate(labels) if label in d_f_metals]


def get_transition_metal_idxs(labels: list[str]) -> list[int]:
    """Transition metals."""
    return [i for i, label in enumerate(labels) if label in TRANSITION_METALS]


def get_lanthanide_actinide_idxs(labels: list[str]) -> list[int]:
    """Lanthanides and actinides."""
    lanthanide_actinide_elements = LANTHANIDES | ACTINIDES
    return [
        i for i, label in enumerate(labels) if label in lanthanide_actinide_elements
    ]


def get_alkali_alkaline_earth_metal_idxs(labels: list[str]) -> list[int]:
    """Alkali metals (Group 1) and alkaline earth metals (Group 2)."""
    return [
        i for i, label in enumerate(labels) if label in ALKALI_AND_ALKALINE_EARTH_METALS
    ]


def get_post_transition_metal_idxs(labels: list[str]) -> list[int]:
    """Post-transition metals."""
    return [i for i, label in enumerate(labels) if label in POST_TRANSITION_METALS]


def get_metalloid_idxs(labels: list[str]) -> list[int]:
    """Metalloids."""
    return [i for i, label in enumerate(labels) if label in METALLOIDS]


def get_radii(labels: list[str]):
    radii = []
    for lab in labels:
        if lab[-1].isdigit():
            label = lab[:-1]
        else:
            label = lab
        radii.append(elemdatabase.CovalentRadius3[label])
    return radii


def get_element_count(labels: list[str], heavy_only: bool = False) -> NDArray:
    elems: list[str] = list(elemdatabase.elementnr.keys())
    elem_to_idx: dict[str, int] = {elem: idx for idx, elem in enumerate(elems)}

    count = np.zeros(len(elems), dtype=int)

    for label in labels:
        if heavy_only and label in {"H", "D"}:
            continue

        idx = elem_to_idx.get(label)
        if idx is not None:
            count[idx] += 1

    return count
