"""Standardize the reported metal oxidation states for a single entry using plain lists.

Inputs are given directly as lists (or as their string representation):

    tms      = ["Ni"]                 or  "['Ni']"
    reported = [("nickel", 2)]        or  "[('nickel', 2)]"

    -> matched = ["Ni_2"], confidence = 1.0
"""

import ast
import re
from collections import defaultdict, Counter

from cell2mol.elementdata import ElementData
import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)

# ===============================================================
# Prefix map (extendable)
# ===============================================================
PREFIX_MAP = {
    "mono": 1,
    "bi": 2,
    "di": 2,
    "tri": 3,
    "tetra": 4,
    "penta": 5,
    "hexa": 6,
    "hepta": 7,
    "octa": 8,
    "nona": 9,
    "deca": 10,
}

# ===============================================================
# Canonical metal root fragments (for fuzzy match)
# ===============================================================
metal_roots = {
    "Sc": [],
    "Ti": ["titan"],
    "V": ["vanad"],
    "Cr": ["chrom"],
    "Mn": ["mangan"],
    "Fe": ["ferr"],
    "Co": ["cobal"],
    "Ni": [],
    "Cu": ["cupra"],
    "Zn": [],
    "Y": ["yttr"],
    "Zr": ["zirc"],
    "Nb": ["niob"],
    "Mo": ["molyb"],
    "Tc": ["techn"],
    "Ru": ["ruthen"],
    "Rh": ["rhod"],
    "Pd": ["pallad"],
    "Ag": ["silve", "argent"],
    "Cd": ["cadm"],
    "Hf": ["hafn"],
    "Ta": ["tantal"],
    "W": ["tungs"],
    "Re": ["rhen"],
    "Os": ["osm"],
    "Ir": ["irid"],
    "Pt": ["plat"],
    "Au": ["aura"],
    "Hg": ["merc"],
}


# ===============================================================
# Special reported names that pin the oxidation state directly
# ===============================================================
# Some reported names encode the oxidation state in the ligand name itself,
# independent of any parsed number. Order matters: more specific (longer)
# fragments must come first, because "ferrocen" is a substring of "ferrocenium".
METAL_OS_SPECIAL_CASES = [
    ("ferrocenium", ("Fe", 3)),  # ferrocenium cation -> Fe(III)
    ("ferrocen", ("Fe", 2)),  # ferrocene / ferrocenyl -> Fe(II)
]


def detect_special_case(name: str):
    """Return (symbol, ox) if the reported name is a known special case, else None."""
    n = name.lower()
    for frag, (sym, ox) in METAL_OS_SPECIAL_CASES:
        if frag in n:
            return sym, ox
    return None


# ===============================================================
# Extract last word from a name
# ===============================================================
def get_last_word(name: str):
    parts = re.split(r"[-\s,]+", name.lower())
    if not parts:
        return ""
    return parts[-1]


# ===============================================================
# Match metal from last word (priority)
# ===============================================================
def match_metal_symbol_from_last_word(last_word: str, metal_list):
    # 1) exact canonical element name
    for metal in metal_list:
        canonical = elemdatabase.elementname[metal].lower()
        if last_word == canonical:
            return metal

    # 2) canonical substring match
    for metal in metal_list:
        canonical = elemdatabase.elementname[metal].lower()
        if canonical in last_word:
            return metal

    # 3) root/fragment match
    for metal in metal_list:
        for frag in metal_roots.get(metal, []):
            if frag in last_word:
                return metal

    return None


# ===============================================================
# Full multi-metal extraction from entire name
# ===============================================================
def find_all_metal_symbols(name: str, metal_list):
    """Identify metals mentioned in reported name.
    Priority is last-word-based OS carrier detection."""

    name_lower = name.lower()

    # ---- 1. Try last-word match first ----
    last = get_last_word(name)
    last_match = match_metal_symbol_from_last_word(last, metal_list)
    if last_match:
        return [last_match]  # OS carrier identified

    # ---- 2. Else: find all possible metals by scanning entire string ----
    found = []

    # canonical scanning
    for metal in metal_list:
        canonical = elemdatabase.elementname[metal].lower()
        idx = name_lower.find(canonical)
        if idx != -1:
            found.append((idx, metal))

    # fragment scanning
    for metal in metal_list:
        for frag in metal_roots.get(metal, []):
            idx = name_lower.find(frag)
            if idx != -1:
                found.append((idx, metal))

    if not found:
        return None

    # sort by appearance
    found.sort(key=lambda x: x[0])

    # deduplicate
    ordered = []
    for _, m in found:
        if m not in ordered:
            ordered.append(m)

    return ordered


# ===============================================================
# Extract numeric prefix count
# ===============================================================
def extract_prefix_count(name: str):
    n = name.lower()
    for prefix, count in PREFIX_MAP.items():
        if n.startswith(prefix + "-") or n.startswith(prefix):
            return count
    return 1


# ===============================================================
# Expand reported list into per-metal entries
# ===============================================================
def expand_reported_list(reported_list, metal_list):
    """Take raw reported entries like [('tri-iron-molybdenum', 4)] and expand
    using prefix count, assigning OS to the last metal."""

    expanded = []

    for name, ox in reported_list:
        metals_found = find_all_metal_symbols(name, metal_list)
        if not metals_found:
            continue

        count = extract_prefix_count(name)

        # case A: single metal mentioned
        if len(metals_found) == 1:
            sym = metals_found[0]
            for _ in range(count):
                expanded.append((sym, ox))
            continue

        # case B: multiple metals: OS last
        prefix_metals = metals_found[:-1]
        os_carrier = metals_found[-1]

        # assign None to prefix metals, repeated count times
        for m in prefix_metals:
            for _ in range(count):
                expanded.append((m, None))

        # single OS to carrier (once)
        expanded.append((os_carrier, ox))

    return expanded


# ===============================================================
# Single-entry matching / scoring (list inputs, no row dict)
# ===============================================================
def _as_list(value):
    """Accept a real list, its string form ("['Ni']"), or None; return a list.

    None (e.g. a CIF with no reported metal oxidation states) yields an empty
    list rather than raising.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return list(value)


def match_reported_to_metals(tms, reported):
    """Same logic as extract_metal_ox_v2.match_reported_to_metals, but taking
    the two lists directly instead of a row dict."""
    metal_list = _as_list(tms)
    reported_list = _as_list(reported)

    logger.info(f"Matching reported metal OS {reported_list} to metals {metal_list}")

    # Special cases: the reported name pins the oxidation state directly
    # (e.g. "ferrocene" -> Fe_2, "ferrocenium" -> Fe_3). Override the parsed OS
    # for those entries, then let the normal pipeline resolve the symbol.
    overridden = []
    for name, ox in reported_list:
        special = detect_special_case(name)
        overridden.append((name, special[1]) if special else (name, ox))
    reported_list = overridden

    expanded = expand_reported_list(reported_list, metal_list)
    if not expanded:
        return [f"{m}_unknown" for m in metal_list]

    # More than one reported metal (complex cases)
    if len(reported_list) > 1:
        os_queue = defaultdict(list)
        for sym, ox in expanded:
            if sym in metal_list and ox is not None:
                os_queue[sym].append(ox)

        final_assignments = []
        for m, needed in Counter(metal_list).items():
            ox_list = os_queue.get(m, [])
            if len(ox_list) == 0:
                final_assignments.extend([f"{m}_unknown"] * needed)
            elif len(set(ox_list)) == 1:
                final_assignments.extend([f"{m}_{ox_list[0]}"] * needed)
            elif len(ox_list) >= needed:
                final_assignments.extend([f"{m}_{ox}" for ox in ox_list[:needed]])
            else:
                padded = ox_list + [ox_list[-1]] * (needed - len(ox_list))
                final_assignments.extend([f"{m}_{ox}" for ox in padded])
        return final_assignments

    # Single reported metal
    if len(expanded) == len(metal_list):
        result = []
        for (sym, ox), m in zip(expanded, metal_list):
            if sym == m:
                result.append(f"{m}_{ox if ox is not None else 'unknown'}")
            else:
                result.append(f"{m}_unknown")
        return result

    # Fewer expanded than metals
    result = []
    for m in metal_list:
        assigned = None
        for sym, ox in expanded:
            if sym == m:
                assigned = ox
                break
        result.append(f"{m}_unknown" if assigned is None else f"{m}_{assigned}")
    return result


def scoring_function(matched):
    """Fraction of assignments that carry a known oxidation state."""
    if not matched:
        return 0.0
    known = sum(1 for m in matched if "_unknown" not in m)
    return known / len(matched)


def standardize_reported_metal_os(tms, reported):
    """Standardize the reported metal oxidation states into ``Metal_state``
    tokens (e.g. "Ni_2") so they can be compared against the cell2mol-assigned
    metal OS.

    Returns (standardized_metal_os, confidence) for a single entry.
    """
    matched = match_reported_to_metals(tms, reported)
    confidence = scoring_function(matched)
    return matched, confidence


if __name__ == "__main__":
    tms = ["Ni"]
    reported = [("nickel", 2)]

    matched, confidence = standardize_reported_metal_os(tms, reported)

    print("TMs                     :", tms)
    print("reported_metal_os:", reported)
    print("matched_metal_os        :", matched)
    print("confidence              :", confidence)
