"""Match reported metal names to element symbols and oxidation-state tokens.

Accept lists or their literal string forms. For example, ["Ni"] and
[("nickel", 2)] produce (["Ni_2"], 1.0).
"""

import ast
import re
from collections import defaultdict, Counter

from cell2mol.elementdata import ElementData
import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)

# Multiplicity prefixes in reported names.
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

# Alternate element-name fragments used for substring matching.
METAL_NAME_ROOTS = {
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


# Check specific fragments first: ferrocen is also contained in ferrocenium.
METAL_OS_SPECIAL_CASES = [
    ("ferrocenium", ("Fe", 3)),  # ferrocenium cation -> Fe(III)
    ("ferrocen", ("Fe", 2)),  # ferrocene / ferrocenyl -> Fe(II)
]


def match_special_metal_state(name: str):
    """Return a name-encoded (element symbol, oxidation state), or None."""
    n = name.lower()
    for frag, (sym, ox) in METAL_OS_SPECIAL_CASES:
        if frag in n:
            return sym, ox
    return None


def extract_last_name_token(name: str):
    """Extract the final lowercase name token, splitting on hyphens, spaces, or commas."""
    name = re.sub(r"\((?:[ivx]+|0|o)\)$", "", name, flags=re.IGNORECASE)
    parts = re.split(r"[-\s,]+", name.lower())
    if not parts:
        return ""
    return parts[-1]


def match_metal_symbol_from_last_word(last_word: str, metal_list):
    """Match a name token by exact element name, name substring, then root fragment."""
    for metal in metal_list:
        canonical = elemdatabase.elementname[metal].lower()
        if last_word == canonical:
            return metal

    for metal in metal_list:
        canonical = elemdatabase.elementname[metal].lower()
        if canonical in last_word:
            return metal

    for metal in metal_list:
        for frag in METAL_NAME_ROOTS.get(metal, []):
            if frag in last_word:
                return metal

    return None


def find_metal_symbols(name: str, metal_list):
    """Match the final name token first; otherwise scan the full name.

    Return distinct matching symbols in appearance order, or None."""

    name_lower = name.lower()

    last = extract_last_name_token(name)
    last_match = match_metal_symbol_from_last_word(last, metal_list)
    if last_match:
        return [last_match]

    found = []

    for metal in metal_list:
        canonical = elemdatabase.elementname[metal].lower()
        idx = name_lower.find(canonical)
        if idx != -1:
            found.append((idx, metal))

    for metal in metal_list:
        for frag in METAL_NAME_ROOTS.get(metal, []):
            idx = name_lower.find(frag)
            if idx != -1:
                found.append((idx, metal))

    if not found:
        return None

    found.sort(key=lambda x: x[0])

    ordered = []
    for _, m in found:
        if m not in ordered:
            ordered.append(m)

    return ordered


def extract_prefix_count(name: str):
    """Read multiplicity on the final metal token, defaulting to one.

    A ligand prefix such as ``tetra`` in ``tetracyano-iron`` does not
    describe the number of metal atoms; ``di-iron`` does.
    """
    name = re.sub(r"\((?:[ivx]+|0|o)\)$", "", name, flags=re.IGNORECASE)
    parts = re.split(r"[-\s,]+", name.lower().strip())
    n = parts[-1]
    if len(parts) > 1 and parts[-2] in PREFIX_MAP:
        return PREFIX_MAP[parts[-2]]
    for prefix, count in PREFIX_MAP.items():
        if n.startswith(prefix):
            return count
    return 1


def expand_reported_metal_states(reported_list, metal_list):
    """Expand reported names into (element symbol, oxidation state) entries.

    Repeat matches by prefix count. When several metals match, assign the
    reported state to the last and leave preceding states unknown."""

    expanded = []

    for name, ox in reported_list:
        metals_found = find_metal_symbols(name, metal_list)
        if not metals_found:
            continue

        count = extract_prefix_count(name)

        # Repeat a single-metal match by its multiplicity prefix.
        if len(metals_found) == 1:
            sym = metals_found[0]
            for _ in range(count):
                expanded.append((sym, ox))
            continue

        # The final matching metal carries the reported oxidation state.
        prefix_metals = metals_found[:-1]
        os_carrier = metals_found[-1]

        # Preceding metals have unknown states.
        for m in prefix_metals:
            for _ in range(count):
                expanded.append((m, None))

        expanded.append((os_carrier, ox))

    return expanded


def _as_list(value):
    """Return a list from an iterable or literal string; None becomes an empty list."""
    if value is None:
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return list(value)


def match_reported_metal_states(tms, reported):
    """Match reported names and states to the metals present.

    Return element_state tokens, using element_unknown for unresolved entries.
    Name-encoded special states override the reported numeric value."""
    metal_list = _as_list(tms)
    reported_list = _as_list(reported)

    logger.info(f"Matching reported metal OS {reported_list} to metals {metal_list}")

    # Apply name-encoded states before matching element symbols.
    overridden = []
    for name, ox in reported_list:
        special = match_special_metal_state(name)
        overridden.append((name, special[1]) if special else (name, ox))
    reported_list = overridden

    expanded = expand_reported_metal_states(reported_list, metal_list)
    if not expanded:
        return [f"{m}_unknown" for m in metal_list]

    # A name-encoded state describes only that named moiety. In particular,
    # ferrocene's Fe(II) must not propagate to other iron atoms in a complex.
    if all(match_special_metal_state(name) for name, _ in reported_list):
        states = defaultdict(list)
        for sym, ox in expanded:
            states[sym].append(ox)
        result = []
        for metal in metal_list:
            ox = states[metal].pop(0) if states[metal] else None
            result.append(f"{metal}_{ox if ox is not None else 'unknown'}")
        return result

    # Distribute multiple reports across occurrences of each element.
    if len(reported_list) > 1:
        os_queue = defaultdict(list)
        for sym, ox in expanded:
            if sym in metal_list:
                os_queue[sym].append(ox)

        final_assignments = []
        for m, needed in Counter(metal_list).items():
            ox_list = os_queue.get(m, [])
            if len(ox_list) == 0:
                final_assignments.extend([f"{m}_unknown"] * needed)
            elif len(set(ox_list)) == 1:
                state = ox_list[0] if ox_list[0] is not None else "unknown"
                final_assignments.extend([f"{m}_{state}"] * needed)
            elif len(ox_list) >= needed:
                final_assignments.extend(
                    [
                        f"{m}_{ox if ox is not None else 'unknown'}"
                        for ox in ox_list[:needed]
                    ]
                )
            else:
                padded = ox_list + [ox_list[-1]] * (needed - len(ox_list))
                final_assignments.extend(
                    [f"{m}_{ox if ox is not None else 'unknown'}" for ox in padded]
                )
        return final_assignments

    # Use positional assignments when expanded and target counts agree.
    if len(expanded) == len(metal_list):
        result = []
        for (sym, ox), m in zip(expanded, metal_list):
            if sym == m:
                result.append(f"{m}_{ox if ox is not None else 'unknown'}")
            else:
                result.append(f"{m}_unknown")
        return result

    # Otherwise reuse the first matching state for each element.
    result = []
    for m in metal_list:
        assigned = None
        for sym, ox in expanded:
            if sym == m:
                assigned = ox
                break
        result.append(f"{m}_unknown" if assigned is None else f"{m}_{assigned}")
    return result


def compute_known_state_fraction(matched):
    """Return the fraction of tokens with a known oxidation state; zero for no tokens."""
    if not matched:
        return 0.0
    known = sum(1 for m in matched if "_unknown" not in m)
    return known / len(matched)


def standardize_reported_metal_oxidation_states(tms, reported):
    """Return matched element_state tokens and the fraction with known oxidation states."""
    matched = match_reported_metal_states(tms, reported)
    confidence = compute_known_state_fraction(matched)
    return matched, confidence


if __name__ == "__main__":
    tms = ["Ni"]
    reported = [("nickel", 2)]

    matched, confidence = standardize_reported_metal_oxidation_states(tms, reported)

    print("TMs                     :", tms)
    print("reported_metal_os:", reported)
    print("matched_metal_os        :", matched)
    print("confidence              :", confidence)
