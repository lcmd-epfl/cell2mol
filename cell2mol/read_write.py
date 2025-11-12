#!/usr/bin/env python

import numpy as np
import pickle
import sys
import re
from collections import defaultdict
import os
import traceback
from typing import Tuple
from collections import Counter
from ase.io import read
from pathlib import Path
from typing import Dict
import pandas as pd
import networkx as nx

transition_metals = {
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


#######################
def screening_cif(cif_file_path):
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
    if len(moiety_dicts) == 0:
        polymeric = False
    else:
        polymeric = any("n(" in moiety["formula"] for moiety in moiety_dicts)

    return radical, disorder, notfound_atom, polymeric


#################
def prefilter_cif(input_path):
    """
    Pre-filters a CIF file to check for certain conditions.
    Returns True if the file is ready for processing, otherwise returns False.
    """
    # Check for radical, disorder, 3D fractional coordinates, and polymeric structure
    radical, disorder, notfound_atom, polymeric = screening_cif(input_path)

    message = ""

    if any([radical, disorder, notfound_atom, polymeric]):
        # print(f"{radical=}, {disorder=}, {notfound_atom=}, {polymeric=}")
        if radical:
            message += "\nRadical found in .cif file."
        if disorder:
            message += "\nDisorder found in .cif file."
        if notfound_atom:
            message += "\nNo fractional coordinates found in .cif file."
        if polymeric:
            message += "\nPolymeric structure found in .cif file."
        return False, message
    else:
        print("Cif file is ready for processing")
        return True, message


#######################
def get_geom_bond(file_path):
    full_text = Path(file_path).read_text()
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
            # print("Bond header map:", header_map)

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

    # print("Bond data (first 5):", geom_bond_data[:5])
    # print("Moieties:", moiety_list)

    return geom_bond_data, moiety_list


######################
def get_wyckoff_positions(file_path):
    full_text = Path(file_path).read_text()
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
            # print("Atom site header map:", header_map)

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

    # --- Preview output ---
    # print("Atom site data (first 5):", atom_site_data[:5])
    # print("Atom labels:", atom_site_labels[:5])
    # print("Element types:", ref_labels[:5])
    # print("Fractional coords:", ref_fracs[:5])

    return atom_site_labels, ref_labels, ref_fracs


##################
def get_moiety_indices_from_labels(atom_site_labels, moiety_list):
    flat_list = [atom for moiety in moiety_list for atom in moiety]

    atom_site_labels = np.array(atom_site_labels)  # ensure it's a numpy array
    moiety_indices = [
        np.where(np.isin(atom_site_labels, moiety))[0].tolist()
        for moiety in moiety_list
    ]
    for i, atom in enumerate(atom_site_labels):
        if atom not in flat_list:
            moiety_indices.append([i])

    return moiety_indices


#######################
def exit_with_error_input(message):
    """Logs the error message to a file and exits the program."""
    error_log_path = os.path.join(os.getcwd(), "error_input.out")
    with open(error_log_path, "w") as error_log:
        error_log.write(f"Error: {message}\n")
    sys.exit(message)


#######################
def exit_with_error_exception(e):
    """Logs the error details to a file and exits the program."""
    error_details = traceback.format_exc()
    error_log_path = os.path.join(os.getcwd(), f"error_{type(e).__name__}.out")

    # Write the full error details to the log file
    with open(error_log_path, "w") as error_log:
        error_log.write(f"Error message: {type(e).__name__} - {str(e)}\n")
        error_log.write(f"Error details:\n{error_details}")

    # Print the error details to the console
    print(f"An error occurred. Details have been logged to {error_log_path}")
    print(f"Error details:\n{error_details}")

    sys.exit(e)


#######################
def sum_formulas(formulas, ratios=None):
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


#######################
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


#######################
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


#######################
def extract_chemical_name(file_path, tag="_chemical_name_systematic"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
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


#######################
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

    return oxidation_states


#######################
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
    is_complex = any(re.search(rf"{metal}\d*", formula) for metal in transition_metals)
    compound_type = "complex" if is_complex else "molecule"

    return (formula, ratio, charge, compound_type)


################################
def cifformula_to_list(formula: str) -> list:
    # Match element symbols with optional numbers (e.g., 'C4', 'H2', 'N1')
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    result = []
    for element, count in tokens:
        n = int(count) if count else 1
        result.extend([element] * n)
    return result


#########################
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

        # print("moiety_dicts=",moiety_dicts)
    else:
        print("Error parsing moiety information from CIF file.")
        moiety_dicts = []
    return moiety_dicts


##########################
def classify_metals(formula: str) -> Dict[str, Dict[str, int]]:
    # Define categories
    transition_metals = {
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

    alkali_metals = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
    alkaline_earth_metals = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}

    post_transition_metals = {"Al", "Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi"}

    lanthanides = {
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

    actinides = {
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

            if element in transition_metals:
                result["transition_metals"][element] = (
                    result["transition_metals"].get(element, 0) + count
                )
            elif element in alkali_metals:
                result["alkali_metals"][element] = (
                    result["alkali_metals"].get(element, 0) + count
                )
            elif element in alkaline_earth_metals:
                result["alkaline_earth_metals"][element] = (
                    result["alkaline_earth_metals"].get(element, 0) + count
                )
            elif element in post_transition_metals:
                result["post_transition_metals"][element] = (
                    result["post_transition_metals"].get(element, 0) + count
                )
            elif element in lanthanides:
                result["lanthanides"][element] = (
                    result["lanthanides"].get(element, 0) + count
                )
            elif element in actinides:
                result["actinides"][element] = (
                    result["actinides"].get(element, 0) + count
                )

    return result


##########################
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


##########################
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


##########################
def compare_formula_xyz_vs_cif(xyzfile: str, formula_str_from_cif: str) -> dict:
    """
    Compare the elements from CIF with the elements in the XYZ file.
    """
    element_pattern = r"([A-Z][a-z]*)(\d*)"
    parsed_formula_cif = dict()
    for element, count in re.findall(element_pattern, formula_str_from_cif):
        parsed_formula_cif[element] = int(count) if count else 1
    mol = read(xyzfile)
    element_list = mol.get_chemical_symbols()
    element_list_count = dict(Counter(element_list))
    comparison_result = {
        "in_formula_cif": parsed_formula_cif,
        "in_list_xyz": element_list_count,
        "missing_in_xyz": {
            k: v - element_list_count.get(k, 0)
            for k, v in parsed_formula_cif.items()
            if element_list_count.get(k, 0) < v
        },
        "extra_in_xyz": {
            k: v - parsed_formula_cif.get(k, 0)
            for k, v in element_list_count.items()
            if parsed_formula_cif.get(k, 0) < v
        },
    }
    return comparison_result


#######################
def save_binary(variable, pathfile, backup: bool = False):
    try:
        file = open(pathfile, "wb")
        pickle.dump(variable, file)
        file.close()
    except Exception as exc:
        print("Error Saving Binary for pathfile:", pathfile)
        print(exc)


#######################
def load_binary(pathfile):
    with open(pathfile, "rb") as pickle_file:
        binary = pickle.load(pickle_file)
    return binary


##############
def readxyz(file):
    labels = []
    pos = []
    xyz = open(file, "r")
    n_atoms = xyz.readline()
    title = xyz.readline()
    for line in xyz:
        line_data = line.split()
        if len(line_data) == 4:
            label, x, y, z = line.split()
            pos.append([float(x), float(y), float(z)])
            labels.append(label)
        else:
            print("I can't read the xyz. It has =/ than 4 columns")
    xyz.close()

    return labels, pos


################################
def printxyz(labels, pos):
    print(len(labels))
    print("")
    for idx, l in enumerate(labels):
        print("%s  %.6f  %.6f  %.6f" % (l, pos[idx][0], pos[idx][1], pos[idx][2]))


##############
def writexyz(fdir, fname, labels, pos, charge: int = 0, spin: int = 1, info: str = ""):
    """Writes an XYZ file with given labels and positions."""
    os.makedirs(fdir, exist_ok=True)

    fullname = os.path.join(fdir, fname)
    natoms = len(labels)

    with open(fullname, "w") as fil:
        print(natoms, file=fil)
        print(f"{charge=} {spin=} {info}", file=fil)
        for label, (x, y, z) in zip(labels, pos):
            fil.write(f"{label:<2}  {x:15.8f}  {y:15.8f}  {z:15.8f}\n")


######################################################
def get_cell_parameters(structure):
    """Extracts cell parameters and symmetry operations from structure."""
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
    cell_vector = structure.cell.array
    cell_param = structure.cell.cellpar()
    space_group = structure.info.get("spacegroup")
    sym_ops = space_group.get_op() if space_group else None
    # print(f"Cell parameters: {cell_param}")
    # print(f"Cell vectors: {cell_vector}")
    # print(f"Space group: {space_group if space_group else 'N/A'}")
    # print("Symmetry operations:", sym_ops if sym_ops else "No symmetry operations found")

    return cell_labels, cell_pos, cell_fracs, cell_vector, cell_param, sym_ops


######################################################
def print_refmoleclist(cell):
    for i, ref in enumerate(cell.refmoleclist):
        ref_info = f"Reference Molecule {i}: {ref.formula} "
        if ref.iscomplex:
            ref_info += "(TM Complex) "
        if ref.has_IA_IIA:
            ref_info += "(Complex with Alkali or Alkaline metals) "
        if ref.has_post_transition_metal:
            ref_info += "(Complex with Post-Transition metals) "
        if (
            not ref.iscomplex
            and not ref.has_IA_IIA
            and not ref.has_post_transition_metal
        ):
            ref_info += "(Non-complex)"
        ref_info += "\n"
        if ref.totcharge is not None:
            ref_info += f"totcharge={ref.totcharge} "
        if ref.totcharge_cif is not None:
            ref_info += f"totcharge_cif={ref.totcharge_cif} "
        if ref.smiles is not None:
            ref_info += f"smiles={ref.smiles}"
        print(ref_info)

        if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
            for met in ref.metals:
                met_info = f"\t{met.formula} ({met.subtype}) atom_site_label={met.atom_site_label}"
                if met.charge is not None:
                    met_info += f" metal_OS={met.charge}"
                elif met.possible_cs is not None:
                    met_info += f" metal_possible_OS={met.possible_cs}"
                print(met_info)

                if met.coord_sphere_formula is not None:
                    print(
                        f"\t|--Coordination information coord_sphere_formula={met.coord_sphere_formula}"
                    )

                if all(
                    hasattr(met, attr)
                    for attr in ["coord_nr", "coord_geometry", "geom_deviation"]
                ):
                    print(
                        f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}"
                    )

                if all(
                    hasattr(met, attr)
                    for attr in [
                        "coord_nr_with_metal_bonds",
                        "coord_geometry_with_metal_bonds",
                        "geom_deviation_with_metal_bonds",
                        "metals",
                    ]
                ):
                    bonded_metals = [m.label for m in met.metals]
                    if len(bonded_metals) > 0:
                        print(
                            f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}"
                        )

            for lig in ref.ligands:
                lig_info = f"\t{lig.formula} ({lig.subtype})"
                for attr in ["smiles", "denticity", "totcharge"]:
                    if hasattr(lig, attr):
                        lig_info += f" {attr}={getattr(lig, attr)}"

                if lig.possible_cs is not None and lig.totcharge is None:
                    if len(lig.possible_cs) > 0:
                        lig_info += " lig.possible_cs Exists"
                    else:
                        lig_info += " lig.possible_cs Does not exist"

                print(lig_info)
                if lig.groups is not None:
                    for group in lig.groups:
                        group_info = f"\t|--(group) {group.labels}"
                        for attr in ["denticity"]:
                            if hasattr(group, attr):
                                group_info += f" {attr}={getattr(group, attr)}"
                        for attr in ["is_haptic", "haptic_type"]:
                            if hasattr(group, attr):
                                if group.is_haptic:
                                    group_info += f" {attr}={getattr(group, attr)}"
                        if group.metals is not None:
                            group_info += f" connected_metals={[m.atom_site_label for m in group.metals]} "
                        # if group.closest_metal is not None:
                        #     group_info += f" closest_metal.label={group.closest_metal.label}"
                        print(group_info)


######################################################
def print_unique_species(cell):
    unique_species = getattr(cell, "unique_species", None)
    if not unique_species:
        print("\nNo unique species found in the cell object.")
        return

    print(f"\nUnique Species in {cell.subtype}:")
    for specie in unique_species:
        parts = [f"{specie.unique_index=}", f"{specie.formula}", f"({specie.subtype})"]

        if specie.subtype == "metal":
            parts.append(f"{specie.coord_sphere_formula=}")
            if getattr(specie, "charge", None) is not None:
                parts.append(f"{specie.charge=}")
        elif specie.subtype == "ligand":
            parts.append(f"{specie.denticity=}")
            if specie.is_haptic:
                parts.append(f"{specie.haptic_type=}")
            if getattr(specie, "smiles", None) is not None:
                parts.append(f"{specie.smiles=}")
            if getattr(specie, "totcharge", None) is not None:
                parts.append(f"{specie.totcharge=}")
            if getattr(specie, "groups", None) is not None:
                parts.append(f"groups={[group.formula for group in specie.groups]}")
        else:
            if getattr(specie, "smiles", None) is not None:
                parts.append(f"{specie.smiles=}")
            if getattr(specie, "totcharge", None) is not None:
                parts.append(f"{specie.totcharge=}")

        print("\t" + " ".join(parts))


######################################################
def print_possible_charges(cell, debug=0):
    """
    Print the possible charges for each species in the cell object.
    """
    if cell.species_list is not None:
        print(f"\nPossible charges of species in {cell.subtype}:")
        for specie in cell.species_list:
            if specie.possible_cs is not None:
                if specie.subtype == "metal":
                    print(
                        f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} {specie.possible_cs=}"
                    )
                else:
                    print(
                        f"\t{specie.unique_index=} {specie.formula} ({specie.subtype})\n\t{specie.possible_cs=}"
                    )
            else:
                if specie.subtype == "metal":
                    print(
                        f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} No possible cs"
                    )
                else:
                    print(
                        f"\t{specie.unique_index=} {specie.formula}, {specie.subtype}  No possible cs"
                    )  # [p.subtype for p in specie.parents])
    else:
        print("\nNo species list found in the cell object.")
    print("")


######################################################
def print_moleclist(cell):
    if cell.moleclist is not None:
        print(f"\nMolecules in {cell.subtype}:")
        for i, mol in enumerate(cell.moleclist):
            if mol.totcharge is not None:
                if mol.iscomplex:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (TM complex)"
                    )
                elif mol.has_IA_IIA:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (Complex with Alkali or Alkaline metals)"
                    )
                elif mol.has_post_transition_metal:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (Complex with Post-Transition metals)"
                    )
                else:
                    if mol.smiles is not None:
                        print(
                            f"Unitcell Molecule {i} : {mol.formula} {mol.totcharge=} (Non-complex) {mol.smiles=}"
                        )
                    else:
                        print(
                            f"Unitcell Molecule {i} : {mol.formula} {mol.totcharge=} (Non-complex)"
                        )
            else:
                if mol.iscomplex:
                    print(f"Unitcell Molecule {i}: {mol.formula} (TM complex)")
                elif mol.has_IA_IIA:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} (Complex with Alkali or Alkaline metals)"
                    )
                elif mol.has_post_transition_metal:
                    print(
                        f"Unitcell Molecule {i}: {mol.formula} (Complex with Post-Transition metals)"
                    )
                else:
                    print(f"Unitcell Molecule {i} : {mol.formula} (Non-complex)")

            if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
                for met in mol.metals:
                    met_info = f"\t{met.formula} ({met.subtype}) atom_site_label={met.atom_site_label}"

                    if met.charge is not None:
                        met_info += f" metal_OS={met.charge}"
                    elif met.possible_cs is not None:
                        met_info += f" metal_possible_OS={met.possible_cs}"
                    print(met_info)

                    for attr in ["coord_sphere_formula"]:
                        if hasattr(met, attr):
                            print(
                                f"\t|--Coordination information {attr}={getattr(met, attr)}"
                            )

                    if all(
                        hasattr(met, attr)
                        for attr in ["coord_nr", "coord_geometry", "geom_deviation"]
                    ):
                        print(
                            f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}"
                        )

                    if all(
                        hasattr(met, attr)
                        for attr in [
                            "coord_nr_with_metal_bonds",
                            "coord_geometry_with_metal_bonds",
                            "geom_deviation_with_metal_bonds",
                            "metals",
                        ]
                    ):
                        bonded_metals = [m.label for m in met.metals]
                        if len(bonded_metals) > 0:
                            print(
                                f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}"
                            )
                print("")

                for lig in mol.ligands:
                    lig_info = f"\t{lig.formula} ({lig.subtype})"
                    for attr in ["smiles", "denticity", "totcharge"]:
                        if hasattr(lig, attr):
                            lig_info += f" {attr}={getattr(lig, attr)}"
                    print(lig_info)
                    if lig.groups is not None:
                        for group in lig.groups:
                            group_info = f"\t|--(group){group.labels}"
                            for attr in ["denticity"]:
                                if hasattr(group, attr):
                                    group_info += f" {attr}={getattr(group, attr)}"
                            for attr in ["is_haptic", "haptic_type"]:
                                if hasattr(group, attr):
                                    if group.is_haptic:
                                        group_info += f" {attr}={getattr(group, attr)}"
                            if group.metals is not None:
                                group_info += f" connected_metals={[m.atom_site_label for m in group.metals]}"
                            # if group.closest_metal is not None:
                            #     group_info += f" closest_metal.label={group.closest_metal.label}"
                            print(group_info)

                            # Optional: print group-metals connectivity
                            # for met in group.metals:
                            #     print(f"\t|--(group.metals){met.label} {met.mconnec=}")
    else:
        print("\nNo molecules found in the cell object.")


######################################################
def print_molecule(mol):
    if mol is not None:
        print("\nMolecule:")
        if mol.totcharge is not None:
            if mol.iscomplex:
                print(f"{mol.formula} {mol.totcharge=} (TM complex)")
            elif mol.has_IA_IIA:
                print(
                    f"{mol.formula} {mol.totcharge=} (Complex with Alkali or Alkaline metals)"
                )
            elif mol.has_post_transition_metal:
                print(
                    f"{mol.formula} {mol.totcharge=} (Complex with Post-Transition metals)"
                )
            else:
                if mol.smiles is not None:
                    print(f"{mol.formula} {mol.totcharge=} (Non-complex) {mol.smiles=}")
                else:
                    print(f"{mol.formula} {mol.totcharge=} (Non-complex)")
        else:
            if mol.iscomplex:
                print(f"{mol.formula} (TM complex)")
            elif mol.has_IA_IIA:
                print(f"{mol.formula} (Complex with Alkali or Alkaline metals)")
            elif mol.has_post_transition_metal:
                print(f"{mol.formula} (Complex with Post-Transition metals)")
            else:
                print(f"{mol.formula} (Non-complex)")

        if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
            for met in mol.metals:
                met_info = f"\t{met.formula} ({met.subtype}) atom_site_label={met.atom_site_label}"

                if met.charge is not None:
                    met_info += f" metal_OS={met.charge}"
                elif met.possible_cs is not None:
                    met_info += f" metal_possible_OS={met.possible_cs}"
                print(met_info)

                for attr in ["coord_sphere_formula"]:
                    if hasattr(met, attr):
                        print(
                            f"\t|--Coordination information {attr}={getattr(met, attr)}"
                        )

                if all(
                    hasattr(met, attr)
                    for attr in ["coord_nr", "coord_geometry", "geom_deviation"]
                ):
                    print(
                        f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}"
                    )

                if all(
                    hasattr(met, attr)
                    for attr in [
                        "coord_nr_with_metal_bonds",
                        "coord_geometry_with_metal_bonds",
                        "geom_deviation_with_metal_bonds",
                        "metals",
                    ]
                ):
                    bonded_metals = [m.label for m in met.metals]
                    if len(bonded_metals) > 0:
                        print(
                            f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}"
                        )
            print("")

            for lig in mol.ligands:
                lig_info = f"\t{lig.formula} ({lig.subtype})"
                for attr in ["smiles", "denticity", "totcharge"]:
                    if hasattr(lig, attr):
                        lig_info += f" {attr}={getattr(lig, attr)}"
                print(lig_info)
                if lig.groups is not None:
                    for group in lig.groups:
                        group_info = f"\t|--(group){group.labels}"
                        for attr in ["denticity"]:
                            if hasattr(group, attr):
                                group_info += f" {attr}={getattr(group, attr)}"
                        for attr in ["is_haptic", "haptic_type"]:
                            if hasattr(group, attr):
                                if group.is_haptic:
                                    group_info += f" {attr}={getattr(group, attr)}"
                        if group.metals is not None:
                            group_info += f" connected_metals={[m.atom_site_label for m in group.metals]}"

                        print(group_info)
    else:
        print("\nNo molecule object.")
