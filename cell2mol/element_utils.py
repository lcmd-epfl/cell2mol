import numpy as np
from cell2mol.elementdata import ElementData

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

POST_TRANSITION_METALS = {"Al", "Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi"}

METALLOIDS = {"B", "Si", "Ge", "As", "Sb", "Te"}


def labels2formula(labels: list):
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


def labels2ratio(labels):
    elems = elemdatabase.elementnr.keys()
    ratio = []
    for z in elems:
        nz = list(labels).count(z)
        if nz > 0:
            ratio.append(nz)
    return ratio


def labels2electrons(labels):
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


def get_alkali_alkaline_earth_metal_idxs(labels: list[str]) -> list[int]:
    """Alkali metals (Group 1) and alkaline earth metals (Group 2)."""
    return [
        i for i, label in enumerate(labels) if label in ALKALI_AND_ALKALINE_EARTH_METALS
    ]


def get_non_transition_metal_idxs(labels: list[str]) -> list[int]:
    """Post-transition metals and metalloids."""
    non_transition_metals = POST_TRANSITION_METALS | METALLOIDS

    return [i for i, label in enumerate(labels) if label in non_transition_metals]


def get_post_transition_metal_idxs(labels: list[str]) -> list[int]:
    """Post-transition metals."""
    return [i for i, label in enumerate(labels) if label in POST_TRANSITION_METALS]


def get_radii(labels: list):
    radii = []
    for lab in labels:
        if lab[-1].isdigit():
            label = lab[:-1]
        else:
            label = lab
        radii.append(elemdatabase.CovalentRadius3[label])
    return radii


def get_element_count(labels: list, heavy_only: bool = False) -> np.ndarray:
    elems = list(elemdatabase.elementnr.keys())
    count = np.zeros((len(elems)), dtype=int)
    for label in labels:
        for jdx, elem in enumerate(elems):
            if label == elem:
                count[jdx] += 1
            if (label == "H" or label == "D") and heavy_only:
                count = 0
    return count
