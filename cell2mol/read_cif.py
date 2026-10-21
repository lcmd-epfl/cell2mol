"""Extract CIF structure and metadata, and compare reported formula compositions."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
import re
import networkx as nx
import pandas as pd
from typing import TYPE_CHECKING, TypedDict
from ase import Atoms
from collections import Counter
from cell2mol.element_utils import (
    symbols2formula,
    get_alkali_alkaline_earth_metal_idxs,
    TRANSITION_METALS,
)
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cell2mol.classes.molecule import Molecule


class MoietyRecord(TypedDict):
    """Formula, multiplicity, reported charge, and compound category."""

    formula: str
    ratio: float | str
    charge: int
    type: str


class FormulaMatch(TypedDict):
    """Matched formulas and their absolute element-count differences."""

    ref: str
    match: str
    diff_dict: dict[str, int]


def detect_cif_issues(cif_file_path):
    """Check radical text, unspecified atom labels, missing coordinates, and polymeric formulas."""
    radical = False
    disorder = False
    notfound_atom = False

    with open(cif_file_path, "r") as ciffile:
        file_content = ciffile.read()
        if "radical" in file_content:
            radical = True
        elif "_atom_site_fract_x" not in file_content:
            notfound_atom = True

    UNSPECIFIED_MARKERS = ("?", "*")
    atom_site_labels, _, _ = extract_atom_sites(cif_file_path)
    if any(
        marker in (label or "")
        for label in atom_site_labels
        for marker in UNSPECIFIED_MARKERS
    ):
        logger.warning("Disorder or Unspecified atoms found in Wyckoff positions.")
        disorder = True

    moiety_dicts = extract_formula_moieties(cif_file_path)
    if not moiety_dicts:
        polymeric = False
    else:
        polymeric = any("n(" in moiety["formula"] for moiety in moiety_dicts)

    return radical, disorder, notfound_atom, polymeric


def prefilter_cif(cif_file_path):
    """Return (acceptable, message), rejecting detected CIF issues before processing."""
    radical, disorder, notfound_atom, polymeric = detect_cif_issues(cif_file_path)

    messages = []

    if any([radical, disorder, notfound_atom, polymeric]):
        if radical:
            messages.append("Radical structures are currently not supported.")
        if disorder:
            messages.append("Disordered structures are currently not supported.")
        if notfound_atom:
            messages.append(
                "Input cannot be processed: fractional coordinates are missing."
            )
        if polymeric:
            messages.append("Polymeric structures are currently not supported.")
        return False, "\n".join(messages)
    else:
        return True, ""


def extract_bonds_and_moieties(cif_file_path):
    """Extract bond distances and connected moiety labels; return (None, None) if absent."""
    full_text = Path(cif_file_path).read_text()
    loop_sections = re.split(r"\nloop_\n", full_text)

    geom_bond_data = []

    for section in loop_sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        # Parse CIF bond-loop rows.
        if lines[0].startswith("_geom_bond"):
            headers = []
            start_idx = -1

            for i, line in enumerate(lines):
                if line.strip().startswith("_geom_bond"):
                    headers.append(line.strip())
                else:
                    start_idx = i
                    break

            required = {
                "_geom_bond_atom_site_label_1",
                "_geom_bond_atom_site_label_2",
                "_geom_bond_distance",
            }
            if not required.issubset(headers):
                continue

            header_map = {h: idx for idx, h in enumerate(headers)}
            for line in lines[start_idx:]:
                if line.strip().startswith("_") or line.strip() == "loop_":
                    break
                parts = line.split()
                if len(parts) >= len(headers):
                    try:
                        atom_1 = parts[header_map["_geom_bond_atom_site_label_1"]]
                        atom_2 = parts[header_map["_geom_bond_atom_site_label_2"]]
                        dist = float(
                            parts[header_map["_geom_bond_distance"]].split("(")[0]
                        )
                        geom_bond_data.append((atom_1, atom_2, dist))
                    except (KeyError, ValueError, IndexError):
                        continue

    # Bond-connected atom labels define the moieties.
    G = nx.Graph()
    for atom1, atom2, _ in geom_bond_data:
        G.add_edge(atom1, atom2)

    moieties = list(nx.connected_components(G))
    moiety_list = [sorted(list(group)) for group in moieties]

    if geom_bond_data == []:
        geom_bond_data = None
    if moiety_list == []:
        moiety_list = None
    if geom_bond_data is None and moiety_list is None:
        logger.info("No _geom_bond information found in CIF file.")
    else:
        logger.info(
            "_geom_bond information found in CIF: %d bonds, %d moieties",
            len(geom_bond_data) if geom_bond_data else 0,
            len(moiety_list) if moiety_list else 0,
        )
    return geom_bond_data, moiety_list


def extract_atom_sites(cif_file_path):
    """Extract atom-site labels, element symbols, and fractional coordinates from CIF loops."""
    full_text = Path(cif_file_path).read_text()
    loop_sections = re.split(r"\nloop_\n", full_text)

    atom_site_data = []

    for section in loop_sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        # Parse fractional atom-site coordinates.
        if lines[0].startswith("_atom_site"):
            headers = []
            start_idx = -1

            for i, line in enumerate(lines):
                if line.strip().startswith("_atom_site"):
                    headers.append(line.strip())
                else:
                    start_idx = i
                    break

            required = {
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            }
            if not required.issubset(headers):
                continue

            header_map = {h: idx for idx, h in enumerate(headers)}

            for line in lines[start_idx:]:
                if line.strip().startswith("_") or line.strip() == "loop_":
                    break
                parts = line.split()
                if len(parts) >= len(headers):
                    try:
                        label = parts[header_map["_atom_site_label"]]
                        symbol = parts[header_map["_atom_site_type_symbol"]]
                        x = float(parts[header_map["_atom_site_fract_x"]].split("(")[0])
                        y = float(parts[header_map["_atom_site_fract_y"]].split("(")[0])
                        z = float(parts[header_map["_atom_site_fract_z"]].split("(")[0])
                        atom_site_data.append((label, symbol, x, y, z))
                    except (KeyError, ValueError, IndexError):
                        continue

    atom_site_labels = [entry[0] for entry in atom_site_data]
    ref_labels = [entry[1] for entry in atom_site_data]
    ref_fracs = [[entry[2], entry[3], entry[4]] for entry in atom_site_data]

    return atom_site_labels, ref_labels, ref_fracs


def extract_formula_moieties(file_path: str | Path) -> list[MoietyRecord] | None:
    """Parse formula moieties from semicolon-delimited CIF text; return None if the tag is absent."""
    uploaded_file_path = Path(file_path)
    with uploaded_file_path.open("r", encoding="utf-8") as file:
        cif_data_uploaded = file.read()
    if "_chemical_formula_moiety" in cif_data_uploaded:
        moiety_match_uploaded = re.search(
            r"_chemical_formula_moiety\s*;\s*(.*?)\s*;", cif_data_uploaded, re.DOTALL
        )
        moiety_string_uploaded = (
            moiety_match_uploaded.group(1) if moiety_match_uploaded else ""
        )

        moieties_uploaded = moiety_string_uploaded.split(",")
        moieties_uploaded = [moiety.replace("\n", "") for moiety in moieties_uploaded]
        moiety_tuples = [parse_moiety(m.strip()) for m in moieties_uploaded]
        moiety_dicts: list[MoietyRecord] = [
            {"formula": f, "ratio": s, "charge": c, "type": t}
            for f, s, c, t in moiety_tuples
        ]
    else:
        return None
    return moiety_dicts


def sum_formula_counts(
    formulas: Sequence[str], ratios: Sequence[float | str] | None = None
) -> dict[str, float] | None:
    """Sum formula element counts with optional ratios; return None for invalid ratios."""
    if ratios is None:
        ratios = [1.0] * len(formulas)
    if len(ratios) != len(formulas):
        return None
    if any(not isinstance(r, (int, float)) for r in ratios):
        return None
    total: dict[str, float] = {}
    for f, r in zip(formulas, ratios):
        c = parse_formula_string(f)
        for el, n in c.items():
            total[el] = total.get(el, 0.0) + n * float(r)
    return {el: float(n) for el, n in total.items()}


def compare_element_totals(
    cif_totals: dict[str, float], ref_totals: dict[str, float], atol: float = 1e-8
) -> tuple[pd.DataFrame, bool]:
    """Return an element-count comparison table and whether all differences are within atol."""
    elements = sorted(set(cif_totals) | set(ref_totals))
    rows = []
    all_ok = True
    for el in elements:
        cif_val = cif_totals.get(el, 0.0)
        ref_val = ref_totals.get(el, 0.0)
        delta = cif_val - ref_val
        ok = abs(delta) <= atol
        all_ok = all_ok and ok
        rows.append(
            {
                "Element": el,
                "CIF_total": cif_val,
                "Refcell_total": ref_val,
                "Delta (CIF-Ref)": delta,
                "OK": ok,
            }
        )
    df = pd.DataFrame(rows)
    return df, all_ok


def parse_formula_string(formula_str: str) -> Counter[str]:
    """Parse element symbols and integer counts from a flat formula string."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula_str)
    return Counter({el: int(cnt) if cnt else 1 for el, cnt in tokens})


def compute_formula_differences(f1: str, f2: str) -> dict[str, int]:
    """Return absolute element-count differences, omitting unchanged elements."""
    c1 = parse_formula_string(f1)
    c2 = parse_formula_string(f2)
    all_elements = set(c1) | set(c2)
    return {el: abs(c1[el] - c2[el]) for el in all_elements if c1[el] != c2[el]}


def _non_hydrogen_signature(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    """Return an order-independent composition signature excluding hydrogen."""
    return tuple(sorted((el, cnt) for el, cnt in counter.items() if el != "H"))


def find_closest_formula_matches(
    reference: list[str], target: list[str]
) -> dict[int, FormulaMatch]:
    """Match each reference formula to a target by composition.

    Prefer exact matches, then equal non-H composition with the closest H count.
    Otherwise penalize non-H differences more strongly than H differences."""
    parsed_targets = [(tgt, parse_formula_string(tgt)) for tgt in target]
    matches: dict[int, FormulaMatch] = {}

    for i, ref in enumerate(reference):
        c_ref = parse_formula_string(ref)

        # Prefer exact composition matches.
        for tgt, c_tgt in parsed_targets:
            if c_tgt == c_ref:
                matches[i] = {"ref": ref, "match": tgt, "diff_dict": {}}
                break
        else:
            # Then match the non-hydrogen composition.
            sig_ref = _non_hydrogen_signature(c_ref)
            hinsensitive = [
                (tgt, c_tgt)
                for tgt, c_tgt in parsed_targets
                if _non_hydrogen_signature(c_tgt) == sig_ref
            ]

            if hinsensitive:
                # Minimize the hydrogen-count difference.
                best_tgt, best_ct = min(
                    hinsensitive,
                    key=lambda x: abs(x[1].get("H", 0) - c_ref.get("H", 0)),
                )
                matches[i] = {
                    "ref": ref,
                    "match": best_tgt,
                    "diff_dict": compute_formula_differences(ref, best_tgt),
                }
            else:
                # Otherwise weight non-H differences more strongly.
                def weighted_score(c_tgt: Counter[str]) -> float:
                    """Weight non-H count differences by 10 and H differences by 0.1."""
                    elems = set(c_ref) | set(c_tgt)
                    score = 0.0
                    for el in elems:
                        diff = abs(c_ref.get(el, 0) - c_tgt.get(el, 0))
                        if el == "H":
                            score += 0.1 * diff
                        else:
                            score += 10.0 * diff
                    return score

                best_tgt, best_ct = min(
                    parsed_targets, key=lambda x: weighted_score(x[1])
                )
                matches[i] = {
                    "ref": ref,
                    "match": best_tgt,
                    "diff_dict": compute_formula_differences(ref, best_tgt),
                }

    return matches


def extract_chemical_name(
    cif_file_path: str | Path, tag: str = "_chemical_name_systematic"
) -> str | None:
    """Extract and normalize a chemical-name tag from inline or multiline CIF text."""
    try:
        with open(cif_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error("File not found: %s", cif_file_path)
        return None

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.lstrip().startswith(tag):
            # Accept an inline, optionally quoted name.
            after = line.split(tag, 1)[1].strip()
            if after:
                if (after[0] in "'\"" and after[-1:] == after[0]) and len(after) >= 2:
                    val = after[1:-1]
                else:
                    val = after
                return " ".join(val.split())

            i += 1
            # Multiline CIF text starts and ends with a column-one semicolon.
            if i < n and lines[i].startswith(";"):
                i += 1
                buf = []
                while i < n:
                    if lines[i].startswith(";"):
                        break
                    buf.append(lines[i].rstrip("\n"))
                    i += 1
                text = "".join(buf)
                text = " ".join(text.split())
                return text or None
            else:
                # Otherwise use the next nonempty line.
                while i < n and not lines[i].strip():
                    i += 1
                if i < n:
                    val = lines[i].strip()
                    return " ".join(val.split()) if val else None
                return None
        i += 1
    return None


# Match ferrocenium before its shorter ferrocene fragment.
METAL_OS_SPECIAL_CASES = [
    ("ferrocenium", ("iron", 3)),  # ferrocenium cation -> Fe(III)
    ("ferrocen", ("iron", 2)),  # ferrocene / ferrocenyl -> Fe(II)
]


def extract_metal_oxidation_states(
    chemical_name: str,
) -> list[tuple[str, int | None]] | None:
    """Extract named oxidation states from Roman numerals and name-encoded special cases."""
    oxidation_states: list[tuple[str, int | None]] = []

    from cell2mol.standardize_metal_os import elemdatabase, PREFIX_MAP

    prefixes = "|".join(sorted(PREFIX_MAP, key=len, reverse=True))

    # Check name-encoded states before explicit Roman numerals.
    fragments = "|".join(fragment for fragment, _ in METAL_OS_SPECIAL_CASES)
    special_pattern = rf"\b(?:(?:{prefixes})-?)?(?P<fragment>{fragments})"
    special_states = dict(METAL_OS_SPECIAL_CASES)
    for match in re.finditer(special_pattern, chemical_name, re.IGNORECASE):
        fragment = match.group("fragment").lower()
        oxidation_states.append((match.group(0).lower(), special_states[fragment][1]))

    # Accept single states and mixed-valence lists such as di-zirconium(ii,iv).
    numeral = r"(?:viii|vii|iii|vi|iv|ii|ix|x|v|i|0|o)"
    pattern = rf"(\b[a-zA-Z-]+\b)\(\s*({numeral}(?:\s*,\s*{numeral})*)\s*\)"

    matches = list(re.finditer(pattern, chemical_name, re.IGNORECASE))
    for match in matches:
        metal, ox_state = match.groups()
        ox_state_map = {
            "0": 0,
            "o": 0,
            "i": 1,
            "ii": 2,
            "iii": 3,
            "iv": 4,
            "v": 5,
            "vi": 6,
            "vii": 7,
            "viii": 8,
            "ix": 9,  # [IrO4]+
            "x": 10,  # theoretically possible
        }
        states = [ox_state_map[state.strip().lower()] for state in ox_state.split(",")]
        if len(states) == 1:
            oxidation_states.append((match.group(0), states[0]))
        else:
            # Each numeral describes one metal; consume the multiplicity prefix
            # here so downstream expansion does not duplicate every state.
            metal = re.sub(
                rf"\b(?:{prefixes})-?([a-z]+)$", r"\1", metal, flags=re.IGNORECASE
            )
            oxidation_states.extend((metal, state) for state in states)

    # Bare metal names carry composition information even without an OS.
    # Keep a directly attached multiplicity, but exclude ligand prefixes.
    names = "|".join(
        re.escape(elemdatabase.elementname[symbol].lower())
        for symbol in sorted(TRANSITION_METALS)
    )
    bare_pattern = rf"\b(?:(?:{prefixes})-?)?(?:{names})\b"
    for match in re.finditer(bare_pattern, chemical_name, re.IGNORECASE):
        if any(
            explicit.start() <= match.start() < explicit.end() for explicit in matches
        ):
            continue
        oxidation_states.append((match.group(0), None))

    if not oxidation_states:
        return None
    return oxidation_states


def parse_moiety(moiety: str) -> tuple[str, float | str, int, str]:
    """Return (formula, ratio, charge, type) from a moiety string.

    Parse numeric multiplicities and trailing charges. Use an empty ratio for
    symbolic multiplicities such as n(...); classify transition-metal formulas as complex."""
    ratio: float | str = 1.0
    formula_with_charge = moiety.strip()

    # Symbolic multiplicities have no numeric ratio.
    x_match = re.match(r"^[^\d\W]\w*\((.*?)\)$", moiety)
    if x_match:
        formula_with_charge = x_match.group(1)
        ratio = ""

    # Numeric multiplicities, such as 0.5(...) or 2(...).
    match = re.match(r"([0-9\.]+)\((.*?)\)", moiety)
    if match:
        ratio = float(match.group(1))
        formula_with_charge = match.group(2)

    # Parse trailing integer charges, such as 1+ or 2-.
    charge_match = re.search(r"(\d+[+-])$", formula_with_charge)
    if charge_match:
        charge_str = charge_match.group(1)
        charge = int(charge_str[:-1]) * (1 if charge_str[-1] == "+" else -1)
        formula = formula_with_charge.replace(charge_str, "").strip()
    else:
        charge = 0
        formula = formula_with_charge.strip()

    is_complex = any(re.search(rf"{metal}\d*", formula) for metal in TRANSITION_METALS)
    compound_type = "complex" if is_complex else "molecule"

    return (formula, ratio, charge, compound_type)


def expand_formula_symbols(formula: str) -> list[str]:
    """Expand integer formula counts into a list of element symbols."""
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    result = []
    for element, count in tokens:
        n = int(count) if count else 1
        result.extend([element] * n)
    return result


def extract_cif_metadata(cif_file_path):
    """Return the chemical name, reported metal states, and formula moiety records."""

    chemical_name = extract_chemical_name(cif_file_path)
    if chemical_name is None:
        reported_metal_os = None
        logger.info("No _chemical_name_systematic found in CIF")
        logger.info("No metal oxidation state reported in CIF")
    else:
        reported_metal_os = extract_metal_oxidation_states(chemical_name)
        logger.info("_chemical_name_systematic in CIF: %s", chemical_name)
        logger.info("Reported oxidation states in CIF: %s", reported_metal_os)

    moiety_dicts = extract_formula_moieties(cif_file_path)
    if moiety_dicts is None:
        logger.info("No moiety information found in CIF")
    else:
        logger.info(
            "Number of moieties extracted from CIF: %d",
            len(moiety_dicts) if moiety_dicts else 0,
        )
        logger.debug("Moiety dictionaries:")
        for i, moiety in enumerate(moiety_dicts):
            logger.debug("  %d: %s", i, moiety)

    return chemical_name, reported_metal_os, moiety_dicts


def extract_cell_atoms(structure: Atoms):
    """Extract element symbols and Cartesian/fractional coordinates, marking deuterium as D."""
    wrap_keywords = {"pbc": True, "center": (0.5, 0.5, 0.5)}
    cell_labels = []
    for label, n, m in zip(
        structure.get_chemical_symbols(),
        structure.get_atomic_numbers(),
        structure.get_masses(),
    ):
        if n == 1 and (m > 2 or m == 2.01355):  # Deuterium
            cell_labels.append("D")
        else:
            cell_labels.append(label)

    cell_pos = structure.get_positions(wrap=True, **wrap_keywords)
    cell_fracs = structure.get_scaled_positions()
    cell_pos = cell_pos.tolist()
    cell_fracs = cell_fracs.tolist()

    return cell_labels, cell_pos, cell_fracs


def extract_cell_parameters(structure: Atoms):
    """Extract lattice vectors, cell parameters, and available symmetry operations from ASE."""

    cell_vector = structure.cell.array
    cell_param = structure.cell.cellpar()
    space_group = structure.info.get("spacegroup")
    sym_ops = space_group.get_op() if space_group else None

    return cell_vector, cell_param, sym_ops


def check_cif_formula_mismatch(
    moiety_dicts: list[MoietyRecord] | None, refmoleclist: list[Molecule] | None
) -> bool | None:
    """Check CIF moiety formulas against reference molecule formulas.

    Store reported charges on exact matches. Return True for formula differences,
    False for agreement, or None when either input is unavailable."""

    disagree_with_cif_formula = None

    if not moiety_dicts:
        logger.info("No _chemical_formula_moiety information found in the CIF file.")
        return disagree_with_cif_formula
    if not refmoleclist:
        logger.info("No reference molecules found in the refcell.")
        return disagree_with_cif_formula

    formulas_from_refcell = [ref.formula for ref in refmoleclist]

    formulas_from_cif = [
        symbols2formula(expand_formula_symbols(moiety["formula"]))
        for moiety in moiety_dicts
    ]
    ratios_from_cif = [moiety["ratio"] for moiety in moiety_dicts]
    charges_from_cif = [moiety["charge"] for moiety in moiety_dicts]
    matches = find_closest_formula_matches(formulas_from_refcell, formulas_from_cif)

    logger.debug("Formulas from CIF: %s", formulas_from_cif)
    logger.debug("Ratios from CIF: %s", ratios_from_cif)
    logger.debug("Charges from CIF: %s", charges_from_cif)
    logger.debug("Formulas from refcell: %s", formulas_from_refcell)

    disagree_with_cif = []
    for ref_idx, info in matches.items():
        ref = refmoleclist[ref_idx]

        if len(info["diff_dict"]) == 0:
            logger.debug(
                "%s %s: Exact match found. %s", ref_idx, info["ref"], info["match"]
            )
            try:
                cif_idx = formulas_from_cif.index(info["match"])
                ref.totcharge_cif = charges_from_cif[cif_idx]
            except ValueError:
                logger.warning(
                    "Failed to assign charge to refcell molecule %s matched with CIF %s",
                    info["ref"],
                    info["match"],
                )
        else:
            logger.debug(
                "%s %s: Closest match found. %s with difference of %s",
                ref_idx,
                info["ref"],
                info["match"],
                info["diff_dict"],
            )
            alkali_idxs = get_alkali_alkaline_earth_metal_idxs(
                list(info["diff_dict"].keys())
            )
            if alkali_idxs:
                logger.warning(
                    "%s %s: Discrepancy due to alkali/alkaline earth metals.",
                    ref_idx,
                    info["ref"],
                )
            disagree_with_cif.append(ref_idx)

    if disagree_with_cif:
        logger.info("Discrepancies found between refcell and CIF")
        logger.info("This will cause errors in the charge prediction!")
        disagree_with_cif_formula = True

        try:
            cif_totals = sum_formula_counts(formulas_from_cif, ratios_from_cif)
            ref_totals = sum_formula_counts(formulas_from_refcell)
            if cif_totals is not None and ref_totals is not None:
                df_compare, all_match = compare_element_totals(
                    cif_totals, ref_totals, atol=1e-9
                )
                if not all_match:
                    logger.info("Element totals differ between refcell and CIF")
                    logger.debug("Element comparison (CIF vs refcell):")
                    logger.debug(" Element     CIF   Refcell   Delta    OK")
                    logger.debug(" ---------------------------------------")
                    for _, row in df_compare.iterrows():
                        logger.debug(
                            " %-8s %6.1f %9.1f %7.1f %s",
                            row["Element"],
                            row["CIF_total"],
                            row["Refcell_total"],
                            row["Delta (CIF-Ref)"],
                            row["OK"],
                        )
                    logger.debug("Possible causes of discrepancies:")
                    logger.debug(
                        " - Missing atoms in the crystal structure (CIF moiety mismatch)"
                    )
                    logger.debug(
                        " - Connectivity changes due to different adjacency matrices"
                    )
        except Exception:
            logger.warning(
                "Can not calculate element totals from CIF. This may be due to non-float ratios in moieties in CIF."
            )
    else:
        logger.info("No discrepancies found between formulas from refcell and CIF.")
        disagree_with_cif_formula = False

    return disagree_with_cif_formula
