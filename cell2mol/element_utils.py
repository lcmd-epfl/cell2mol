import numpy as np
from cell2mol.elementdata import ElementData
from cell2mol.my_types import NDArray

elemdatabase = ElementData()

NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}

HALOGENS = {"F", "Cl", "Br", "I"}

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


def symbols2formula(symbols: list[str]):
    elems = elemdatabase.elementnr.keys()
    formula = []
    for z in elems:
        nz = list(symbols).count(z)
        if nz > 1:
            formula.append(f"{z}{nz}-")
        if nz == 1:
            formula.append(f"{z}-")
    formula = "".join(formula)[:-1]
    return formula


def symbols2ratio(symbols: list[str]):
    elems = elemdatabase.elementnr.keys()
    ratio = []
    for z in elems:
        nz = list(symbols).count(z)
        if nz > 0:
            ratio.append(nz)
    return ratio


def symbols2electrons(symbols: list[str]):
    if isinstance(symbols, list):
        eleccount = 0
        for symbol in symbols:
            eleccount += elemdatabase.elementnr[symbol]
    return eleccount


def get_df_block_metal_idxs(symbols: list[str]) -> list[int]:
    """Transition metals, lanthanides, and actinides."""
    d_f_metals = TRANSITION_METALS | LANTHANIDES | ACTINIDES
    return [i for i, symbol in enumerate(symbols) if symbol in d_f_metals]


def get_transition_metal_idxs(symbols: list[str]) -> list[int]:
    """Transition metals."""
    return [i for i, symbol in enumerate(symbols) if symbol in TRANSITION_METALS]


def get_lanthanide_actinide_idxs(symbols: list[str]) -> list[int]:
    """Lanthanides and actinides."""
    lanthanide_actinide_elements = LANTHANIDES | ACTINIDES
    return [
        i for i, symbol in enumerate(symbols) if symbol in lanthanide_actinide_elements
    ]


def get_alkali_alkaline_earth_metal_idxs(symbols: list[str]) -> list[int]:
    """Alkali metals (Group 1) and alkaline earth metals (Group 2)."""
    return [
        i
        for i, symbol in enumerate(symbols)
        if symbol in ALKALI_AND_ALKALINE_EARTH_METALS
    ]


def get_post_transition_metal_idxs(symbols: list[str]) -> list[int]:
    """Post-transition metals."""
    return [i for i, symbol in enumerate(symbols) if symbol in POST_TRANSITION_METALS]


def get_metalloid_idxs(symbols: list[str]) -> list[int]:
    """Metalloids."""
    return [i for i, symbol in enumerate(symbols) if symbol in METALLOIDS]


def get_radii(symbols: list[str]):
    radii = []
    for symbol in symbols:
        if symbol[-1].isdigit():
            symbol = symbol[:-1]
        else:
            symbol = symbol
        radii.append(elemdatabase.CovalentRadius3[symbol])
    return radii


def get_element_count(symbols: list[str], heavy_only: bool = False) -> NDArray:
    elems: list[str] = list(elemdatabase.elementnr.keys())
    elem_to_idx: dict[str, int] = {elem: idx for idx, elem in enumerate(elems)}

    count = np.zeros(len(elems), dtype=int)

    for symbol in symbols:
        if heavy_only and symbol in {"H", "D"}:
            continue

        idx = elem_to_idx.get(symbol)
        if idx is not None:
            count[idx] += 1

    return count
