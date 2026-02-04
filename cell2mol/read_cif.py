from pathlib import Path
import numpy as np
import re
import networkx as nx
import pandas as pd
from typing import Tuple, Dict
from collections import Counter
from cell2mol.elementdata import ElementData
from cell2mol.element_utils import (
    labels2formula,
    get_alkali_alkaline_earth_metal_idxs,
    TRANSITION_METALS,
    ALKALI_METALS,
    ALKALINE_EARTH_METALS,
    LANTHANIDES,
    ACTINIDES,
    POST_TRANSITION_METALS,
)

import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


def detect_cif_issues(cif_file_path):
    radical = False
    disorder = False
    notfound_atom = False

    with open(cif_file_path, "r") as ciffile:
        file_content = ciffile.read()
        if "radical" in file_content:
            radical = True
        elif "_atom_site_fract_x" not in file_content:
            notfound_atom = True
        elif "?" in file_content:
            if (
                "_diffrn_ambient_temperature ?" not in file_content
                and "_chemical_melting_point ?" not in file_content
            ):
                disorder = True
            else:
                num_greps = file_content.count("?")
                if num_greps > 1:
                    disorder = True

    moiety_dicts = extract_moiety(cif_file_path)
    if not moiety_dicts:
        polymeric = False
    else:
        polymeric = any("n(" in moiety["formula"] for moiety in moiety_dicts)

    return radical, disorder, notfound_atom, polymeric


def prefilter_cif(cif_file_path):
    """
    Pre-filters a CIF file to check for certain conditions.
    Returns True if the file is ready for processing, otherwise returns False.
    """
    # Check for radical, disorder, 3D fractional coordinates, and polymeric structure
    radical, disorder, notfound_atom, polymeric = detect_cif_issues(cif_file_path)

    messages = []

    if any([radical, disorder, notfound_atom, polymeric]):
        if radical:
            messages.append("Radical found in .cif file.")
        if disorder:
            messages.append("Disorder found in .cif file.")
        if notfound_atom:
            messages.append("No fractional coordinates found in .cif file.")
        if polymeric:
            messages.append("Polymeric structure found in .cif file.")
        return False, "\n".join(messages)
    else:
        return True, ""


def get_geom_bond(cif_file_path):
    """Get geometry bond information from a CIF file."""
    full_text = Path(cif_file_path).read_text()
    loop_sections = re.split(r"\nloop_\n", full_text)

    geom_bond_data = []

    for section in loop_sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        # === _geom_bond block ===
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
                        dist = float(parts[header_map["_geom_bond_distance"]])
                        geom_bond_data.append((atom_1, atom_2, dist))
                    except (KeyError, ValueError, IndexError):
                        continue

    # --- Build connectivity graph and extract moieties ---
    G = nx.Graph()
    for atom1, atom2, _ in geom_bond_data:
        G.add_edge(atom1, atom2)

    moieties = list(nx.connected_components(G))
    moiety_list = [sorted(list(group)) for group in moieties]

    # logger.debug("Bond data (first 5): %s", geom_bond_data[:5])
    # logger.debug("Moieties: %s", moiety_list)

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


def get_wyckoff_positions(cif_file_path):
    """Extract Wyckoff positions from a CIF file."""
    full_text = Path(cif_file_path).read_text()
    loop_sections = re.split(r"\nloop_\n", full_text)

    atom_site_data = []

    for section in loop_sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        # === _atom_site block ===
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

    # --- Separate parsed data ---
    atom_site_labels = [entry[0] for entry in atom_site_data]
    ref_labels = [entry[1] for entry in atom_site_data]
    ref_fracs = [[entry[2], entry[3], entry[4]] for entry in atom_site_data]

    UNSPECIFIED_MARKERS = ("?", "*")

    if any(
        marker in (label or "")
        for label in atom_site_labels
        for marker in UNSPECIFIED_MARKERS
    ):
        logger.warning("Unspecified atoms found in Wyckoff positions.")
        atom_site_labels, ref_labels, ref_fracs = remove_unspecified_atoms(
            atom_site_labels, ref_labels, ref_fracs
        )

    return atom_site_labels, ref_labels, ref_fracs


def remove_unspecified_atoms(atom_site_labels, ref_labels, ref_fracs):
    """
    Remove atoms with unspecified positions from the Wyckoff sites.
    """

    if not (len(atom_site_labels) == len(ref_labels) == len(ref_fracs)):
        raise ValueError("Input lists must have the same length")

    UNSPECIFIED_MARKERS = ("?", "*")

    new_atom_site_labels = []
    new_ref_labels = []
    new_ref_fracs = []

    for atom_site_label, ref_label, ref_frac in zip(
        atom_site_labels, ref_labels, ref_fracs
    ):
        if any(marker in atom_site_label for marker in UNSPECIFIED_MARKERS):
            logger.info(
                "Removing unspecified atom: %s (%s)",
                atom_site_label,
                ref_label,
            )
        else:
            new_atom_site_labels.append(atom_site_label)
            new_ref_labels.append(ref_label)
            new_ref_fracs.append(ref_frac)

    return new_atom_site_labels, new_ref_labels, new_ref_fracs


def remove_unspecified_atoms(atom_site_labels, ref_labels, ref_fracs):
    """
    Remove atoms with unspecified positions from the Wyckoff sites.
    Args:
        atom_site_labels (list): List of atom site labels
            (e.g. ["O1", "H1", "H2"]).
        ref_labels (list): List of reference labels
            (e.g. ["O", "H", "H"]).
        ref_fracs (list): List of fractional coordinates.
    Returns:
        tuple: Filtered lists of atom site labels,
            reference labels, and fractional coordinates.
    """

    substring_to_remove = ["?", "*"]

    # Build new filtered lists
    new_atom_site_labels = []
    new_ref_labels = []
    new_ref_fracs = []

    for atom_site_label, ref_label, ref_frac in zip(
        atom_site_labels, ref_labels, ref_fracs
    ):
        if any(sub in atom_site_label for sub in substring_to_remove):
            logger.info("  Removing unspecified atom: %s", atom_site_label)
        else:
            new_atom_site_labels.append(atom_site_label)
            new_ref_labels.append(ref_label)
            new_ref_fracs.append(ref_frac)

    # Optionally overwrite originals
    atom_site_labels = new_atom_site_labels
    ref_labels = new_ref_labels
    ref_fracs = new_ref_fracs

    return atom_site_labels, ref_labels, ref_fracs


def extract_moiety(file_path: str) -> list:
    """Extracts the moiety information from a CIF file"""
    uploaded_file_path = Path(file_path)
    with uploaded_file_path.open("r", encoding="utf-8") as file:
        cif_data_uploaded = file.read()
    if "_chemical_formula_moiety" in cif_data_uploaded:
        # Extract the moiety block
        moiety_match_uploaded = re.search(
            r"_chemical_formula_moiety\s*;\s*(.*?)\s*;", cif_data_uploaded, re.DOTALL
        )
        moiety_string_uploaded = (
            moiety_match_uploaded.group(1) if moiety_match_uploaded else ""
        )

        # # Split and parse moieties
        moieties_uploaded = moiety_string_uploaded.split(",")
        moieties_uploaded = [moiety.replace("\n", "") for moiety in moieties_uploaded]
        moiety_tuples = [parse_moiety(m.strip()) for m in moieties_uploaded]
        moiety_dicts = [
            {"formula": f, "ratio": s, "charge": c, "type": t}
            for f, s, c, t in moiety_tuples
        ]
    else:
        return None
    return moiety_dicts


def combine_formulas(formulas, ratios=None):
    """
    Sum element counts across a list of formula strings.
    If ratios is provided, multiply each formula's counts by the corresponding ratio.
    Returns a dict {element: float_count}.
    """
    if ratios is None:
        ratios = [1.0] * len(formulas)
    if len(ratios) != len(formulas):
        return None
        # raise ValueError("ratios and formulas must have the same length")
    if any(not isinstance(r, (int, float)) for r in ratios):
        return None
        # raise ValueError("ratios contains non-float values")
    total = Counter()
    for f, r in zip(formulas, ratios):
        c = parse_formula_string(f)
        for el, n in c.items():
            total[el] += n * float(r)
    # Convert to plain dict of floats
    return {el: float(n) for el, n in total.items()}


def compare_totals(cif_totals: dict, ref_totals: dict, atol=1e-8):
    """
    Build a comparison DataFrame with CIF totals, Refcell totals, and Delta = CIF - Ref.
    Also returns whether all elements match within the tolerance.
    """
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


def parse_formula_string(formula_str: str) -> Counter:
    # One lowercase letter max (Cl, Ti, Fe, etc.)
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula_str)
    #     tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    return Counter({el: int(cnt) if cnt else 1 for el, cnt in tokens})


def formula_diff_dict(f1: str, f2: str) -> dict:
    c1 = parse_formula_string(f1)
    c2 = parse_formula_string(f2)
    all_elements = set(c1) | set(c2)
    return {el: abs(c1[el] - c2[el]) for el in all_elements if c1[el] != c2[el]}


def _nonH_signature(counter: Counter):
    # Canonical signature ignoring hydrogens (order-independent)
    return tuple(sorted((el, cnt) for el, cnt in counter.items() if el != "H"))


def find_closest_matches(reference, target):
    # Pre-parse targets once
    parsed_targets = [(tgt, parse_formula_string(tgt)) for tgt in target]
    matches = {}

    for i, ref in enumerate(reference):
        c_ref = parse_formula_string(ref)

        # 1) Exact match on full composition
        for tgt, c_tgt in parsed_targets:
            if c_tgt == c_ref:
                matches[i] = {"ref": ref, "match": tgt, "diff_dict": {}}
                break
        else:
            # 2) H-insensitive signature match (same non-H composition)
            sig_ref = _nonH_signature(c_ref)
            hinsensitive = [
                (tgt, c_tgt)
                for tgt, c_tgt in parsed_targets
                if _nonH_signature(c_tgt) == sig_ref
            ]

            if hinsensitive:
                # Choose the one with minimal |ΔH|
                best_tgt, best_ct = min(
                    hinsensitive,
                    key=lambda x: abs(x[1].get("H", 0) - c_ref.get("H", 0)),
                )
                matches[i] = {
                    "ref": ref,
                    "match": best_tgt,
                    "diff_dict": formula_diff_dict(ref, best_tgt),
                }
            else:
                # 3) Weighted fallback: penalize non-H strongly, H lightly
                def weighted_score(c_tgt: Counter):
                    elems = set(c_ref) | set(c_tgt)
                    score = 0.0
                    for el in elems:
                        diff = abs(c_ref.get(el, 0) - c_tgt.get(el, 0))
                        if el == "H":
                            score += 0.1 * diff  # hydrogens are cheap
                        else:
                            score += 10.0 * diff  # non-H differences matter a lot
                    return score

                best_tgt, best_ct = min(
                    parsed_targets, key=lambda x: weighted_score(x[1])
                )
                matches[i] = {
                    "ref": ref,
                    "match": best_tgt,
                    "diff_dict": formula_diff_dict(ref, best_tgt),
                }

    return matches


def extract_chemical_name(cif_file_path, tag="_chemical_name_systematic"):
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
            # Try to get value on the same line: e.g.,
            # _chemical_name_systematic 'Some name'
            after = line.split(tag, 1)[1].strip()
            if after:  # value is on same line
                # strip matching quotes if present
                if (after[0] in "'\"" and after[-1:] == after[0]) and len(after) >= 2:
                    val = after[1:-1]
                else:
                    val = after
                # normalize whitespace
                return " ".join(val.split())

            # Otherwise value should be on following lines
            i += 1
            # Expect a semicolon in column 1 starting the text block
            if i < n and lines[i].startswith(";"):
                i += 1
                buf = []
                while i < n:
                    # End of block only if semicolon is in column 1
                    if lines[i].startswith(";"):
                        break
                    # Keep exact line content except trailing newline
                    buf.append(lines[i].rstrip("\n"))
                    i += 1
                # Join without forcing spaces between lines, then normalize
                text = "".join(buf)
                # Collapse any excessive whitespace to single spaces
                text = " ".join(text.split())
                return text or None
            else:
                # Fallback: next non-empty line is the value (rare but possible)
                while i < n and not lines[i].strip():
                    i += 1
                if i < n:
                    val = lines[i].strip()
                    return " ".join(val.split()) if val else None
                return None
        i += 1
    return None


def extract_metal_oxidation_state(chemical_name):
    oxidation_states = []

    # Regex pattern to capture any format like "iron(iii)"
    pattern = r"(\b[a-zA-Z-]+\b)\((iii|ii|iv|v|vi|vii|viii|ix|x|i|0|o)\)"

    matches = re.findall(pattern, chemical_name, re.IGNORECASE)
    for metal, ox_state in matches:
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
        oxidation_state = ox_state_map.get(
            ox_state.lower(), ox_state
        )  # Retain text if needed
        oxidation_states.append((metal, oxidation_state))

    if not oxidation_states:
        return None
    return oxidation_states


def parse_moiety(moiety: str) -> Tuple[str, float, int, str]:
    """Parse a moiety string and return its formula, ratio (float), charge, and type"""
    # Default values
    ratio = 1.0
    formula_with_charge = moiety.strip()

    # Handle 'x('
    x_match = re.match(r"^[^\d\W]\w*\((.*?)\)$", moiety)
    if x_match:
        formula_with_charge = x_match.group(1)
        ratio = ""

    # Check for ratio in the unit cell with parentheses, e.g., 0.5(...) or 2(...)
    match = re.match(r"([0-9\.]+)\((.*?)\)", moiety)
    if match:
        ratio = float(match.group(1))
        formula_with_charge = match.group(2)

    # Extract charge at the end, e.g., "1+", "2-"
    charge_match = re.search(r"(\d+[+-])$", formula_with_charge)
    if charge_match:
        charge_str = charge_match.group(1)
        charge = int(charge_str[:-1]) * (1 if charge_str[-1] == "+" else -1)
        formula = formula_with_charge.replace(charge_str, "").strip()
    else:
        charge = 0
        formula = formula_with_charge.strip()

    # Check if it contains any transition metal
    is_complex = any(re.search(rf"{metal}\d*", formula) for metal in TRANSITION_METALS)
    compound_type = "complex" if is_complex else "molecule"

    return (formula, ratio, charge, compound_type)


def cifformula_to_list(formula: str) -> list:
    # Match element symbols with optional numbers (e.g., 'C4', 'H2', 'N1')
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    result = []
    for element, count in tokens:
        n = int(count) if count else 1
        result.extend([element] * n)
    return result


def classify_metals(formula: str) -> Dict[str, Dict[str, int]]:
    # Initialize category results
    result = {
        "transition_metals": {},
        "alkali_metals": {},
        "alkaline_earth_metals": {},
        "post_transition_metals": {},
        "lanthanides": {},
        "actinides": {},
    }

    # Split and parse
    parts = formula.strip().split()
    for part in parts:
        match = re.match(r"([A-Z][a-z]*)(\d*)", part)
        if match:
            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1

            if element in TRANSITION_METALS:
                result["transition_metals"][element] = (
                    result["transition_metals"].get(element, 0) + count
                )
            elif element in ALKALI_METALS:
                result["alkali_metals"][element] = (
                    result["alkali_metals"].get(element, 0) + count
                )
            elif element in ALKALINE_EARTH_METALS:
                result["alkaline_earth_metals"][element] = (
                    result["alkaline_earth_metals"].get(element, 0) + count
                )
            elif element in POST_TRANSITION_METALS:
                result["post_transition_metals"][element] = (
                    result["post_transition_metals"].get(element, 0) + count
                )
            elif element in LANTHANIDES:
                result["lanthanides"][element] = (
                    result["lanthanides"].get(element, 0) + count
                )
            elif element in ACTINIDES:
                result["actinides"][element] = (
                    result["actinides"].get(element, 0) + count
                )

    return result


def metal_info(moiety_dicts: list) -> dict:
    """
    Extracts metal information from moiety dictionaries.
    """
    metal_data = {}
    for moiety in moiety_dicts:
        formula = moiety["formula"]
        ratio = moiety["ratio"]
        charge = moiety["charge"]
        compound_type = moiety["type"]

        # Classify metals
        classified_metals = classify_metals(formula)
        metal_data[formula] = {
            "ratio": ratio,
            "charge": charge,
            "type": compound_type,
            "classified_metals": classified_metals,
        }
    return metal_data


def flatten_metal_info(metal_info_dict, refcode):
    flat_list = []

    for i, (formula, info) in enumerate(metal_info_dict.items()):
        base = {
            "refcode": refcode,  # Placeholder for refcode
            "index": i,
            "formula": formula,
            "ratio": info["ratio"],
            "charge": info["charge"],
            "type": info["type"],
        }

        found = False
        classified = info["classified_metals"]
        for category, elements in classified.items():
            for element, count in elements.items():
                row = base.copy()
                row["metal_category"] = category
                row["metal"] = element
                row["count"] = count
                flat_list.append(row)
                found = True

        if not found:
            row = base.copy()
            row["metal_category"] = None
            row["metal"] = None
            row["count"] = 0
            flat_list.append(row)

    return flat_list


def extract_info_from_cif(cif_file_path):
    """Extract chemical name, metal oxidation state, and moiety information from the CIF file."""

    chemical_name = extract_chemical_name(cif_file_path)
    if chemical_name is None:
        reported_metal_os = None
        logger.info("No _chemical_name_systematic found in CIF")
        logger.info("No metal oxidation state reported in CIF")
    else:
        reported_metal_os = extract_metal_oxidation_state(chemical_name)
        logger.info("_chemical_name_systematic in CIF: %s", chemical_name)
        logger.info("Reported oxidation states in CIF: %s", reported_metal_os)

    moiety_dicts = extract_moiety(cif_file_path)
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


def get_cell_atoms(structure):
    """Extracts atom labels, positions, and fractional coordinates from ase.Atoms object."""
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


def get_cell_parameters(structure):
    """Extracts cell parameters and symmetry operations from ase.Atoms object."""

    cell_vector = structure.cell.array
    cell_param = structure.cell.cellpar()
    space_group = structure.info.get("spacegroup")
    sym_ops = space_group.get_op() if space_group else None

    return cell_vector, cell_param, sym_ops


def compare_cif_with_reference(moiety_dicts, refmoleclist):
    """Compare reference molecules with chemical name,
    metal oxidation state, and moiety information extracted from the CIF file."""

    # Default value
    disagree_with_cif_formula = None

    if not moiety_dicts:
        logger.info("No _chemical_formula_moiety information found in the CIF file.")
        return disagree_with_cif_formula
    if not refmoleclist:
        logger.info("No reference molecules found in the refcell.")
        return disagree_with_cif_formula

    formulas_from_refcell = [ref.formula for ref in refmoleclist]

    formulas_from_cif = [
        labels2formula(cifformula_to_list(moiety["formula"])) for moiety in moiety_dicts
    ]
    ratios_from_cif = [moiety["ratio"] for moiety in moiety_dicts]
    charges_from_cif = [moiety["charge"] for moiety in moiety_dicts]
    matches = find_closest_matches(formulas_from_refcell, formulas_from_cif)

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
            cif_totals = combine_formulas(formulas_from_cif, ratios_from_cif)
            if cif_totals is not None:
                ref_totals = combine_formulas(formulas_from_refcell)
                df_compare, all_match = compare_totals(
                    cif_totals, ref_totals, atol=1e-9
                )
                if not all_match:
                    logger.info("Element totals differ between refcell and CIF")
                    # logger.debug("Element comparison table:\n%s", df_compare.to_string())
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
