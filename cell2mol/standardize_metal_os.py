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

# Multipliers that apply to a parenthesised group, as in bis(ferrocenyl).
GROUP_MULTIPLIER_MAP = {
    "bis": 2,
    "tris": 3,
    "tetrakis": 4,
    "pentakis": 5,
    "hexakis": 6,
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


# Metallocene names that fix the metal state. Check specific fragments first:
# ferrocen is also contained in ferrocenium. Group 4 and 5 names such as
# zirconocene or vanadocene describe a Cp2M fragment whose state depends on
# the other ligands, so they are not listed.
METAL_OS_SPECIAL_CASES = [
    ("ferrocenium", ("Fe", 3)),  # ferrocenium cation -> Fe(III)
    ("ferricen", ("Fe", 3)),  # ferricenium / ferricenyl -> Fe(III)
    ("ferrocen", ("Fe", 2)),  # ferrocene / ferrocenyl -> Fe(II)
    ("cobaltocenium", ("Co", 3)),
    ("cobaltocen", ("Co", 2)),
    ("rhodocenium", ("Rh", 3)),
    ("ruthenocenium", ("Ru", 3)),
    ("ruthenocen", ("Ru", 2)),
    ("osmocenium", ("Os", 3)),
    ("osmocen", ("Os", 2)),
    ("nickelocenium", ("Ni", 3)),
    ("nickelocen", ("Ni", 2)),
    ("chromocenium", ("Cr", 3)),
    ("chromocen", ("Cr", 2)),
    ("manganocenium", ("Mn", 3)),
    ("manganocen", ("Mn", 2)),
]

EXPLICIT_NUMERAL = re.compile(r"\((?:[ivx]+|0|o)\)$", re.IGNORECASE)


def match_special_metal_state(name: str):
    """Return a name-encoded (element symbol, oxidation state), or None."""
    n = name.lower()
    for frag, (sym, ox) in METAL_OS_SPECIAL_CASES:
        if frag in n:
            return sym, ox
    return None


def special_unit_states(name: str, ox: int | None) -> list[int | None]:
    """Return one oxidation state per metallocene unit of a special entry.

    The prefix counts units: tri-ferrocen and bis(ferrocen give three and two.
    Without numerals, a bi- prefix on a cation such as biferrocenium is the
    mixed-valence [Fc-Fc]+ cation, whereas di-ferrocenium names two separate
    ferrocenium units.
    """
    n = name.lower()
    explicit = EXPLICIT_NUMERAL.search(n)
    base = n[: explicit.start()] if explicit else n
    fragment = next(frag for frag, _ in METAL_OS_SPECIAL_CASES if frag in base)
    prefix = base[: base.rfind(fragment)]

    count = 1
    if prefix.endswith("(") and prefix[:-1] in GROUP_MULTIPLIER_MAP:
        count = GROUP_MULTIPLIER_MAP[prefix[:-1]]
    elif prefix.rstrip("-") in PREFIX_MAP:
        count = PREFIX_MAP[prefix.rstrip("-")]

    cation = fragment.endswith("ium") or fragment == "ferricen"
    if not explicit and prefix.rstrip("-") == "bi" and cation and ox is not None:
        return [ox - 1, ox]
    return [ox] * count


def is_metallocene(metal) -> bool:
    """Return True for a metal bound only to two eta5 rings, as in ferrocene.

    Connectivity cannot tell ferrocene from ferrocenium; the name does. A Cp2M
    fragment with further ligands, such as Cp2ZrCl2, is not a metallocene here.
    """
    groups = metal.groups or []
    return len(groups) == 2 and all(
        group.is_haptic and len(group.atoms or []) == 5 for group in groups
    )


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


def _fit_unit_states(units, n_sites):
    """Spread named unit states over n_sites, or return None if ambiguous.

    A single state fills every site, since names such as
    bis(4-(2-ferrocenyl)ethenyl)... mention the moiety once for several units.
    Mixed states fit only when the counts agree.
    """
    if n_sites == 0:
        return []
    if len(units) == n_sites:
        return list(units)
    if len(set(units)) == 1:
        return [units[0]] * n_sites
    return None


def _match_with_metallocenes(metal_list, reported_list, metallocene):
    """Give name-encoded states to metallocene metals and match the rest by name.

    Each element is fitted separately, so cobaltocenium states never reach a
    ferrocene Fe. A metallocene of an element the name does not encode is
    matched with the other metals.
    """
    special = [(n, ox) for n, ox in reported_list if match_special_metal_state(n)]
    others = [(n, ox) for n, ox in reported_list if not match_special_metal_state(n)]

    units = defaultdict(list)
    for name, ox in special:
        sym = match_special_metal_state(name)[0]
        units[sym].extend(special_unit_states(name, ox))

    result = [None] * len(metal_list)
    for sym, sym_units in units.items():
        sites = [
            i
            for i, (m, flag) in enumerate(zip(metal_list, metallocene))
            if flag and m == sym
        ]
        states = _fit_unit_states(sym_units, len(sites))
        if states is None and len(sym_units) == metal_list.count(sym):
            # Exact counts still pin the states when a unit carries another
            # ligand, as in iodo-biruthenocenium(ii,iv).
            sites = [i for i, m in enumerate(metal_list) if m == sym]
            states = list(sym_units)
        if states is None:
            logger.info(
                f"Named metallocene states {sym_units} do not fit "
                f"{len(sites)} metallocene {sym}"
            )
            states = [None] * len(sites)
        for i, ox in zip(sites, states):
            result[i] = f"{sym}_{ox if ox is not None else 'unknown'}"

    rest = [i for i, token in enumerate(result) if token is None]
    rest_metals = [metal_list[i] for i in rest]
    rest_tokens = (
        match_reported_metal_states(rest_metals, others)
        if others
        else [f"{m}_unknown" for m in rest_metals]
    )
    for i, token in zip(rest, rest_tokens):
        result[i] = token
    return result


def match_reported_metal_states(tms, reported, metallocene=None):
    """Match reported names and states to the metals present.

    Return element_state tokens, using element_unknown for unresolved entries.
    Name-encoded special states override the reported numeric value unless
    the fragment carries its own numerals. ``metallocene`` flags, parallel to
    ``tms``, mark the Fe atoms that name-encoded states describe; without
    them those states are matched by element only."""
    metal_list = _as_list(tms)
    reported_list = _as_list(reported)

    logger.info(f"Matching reported metal OS {reported_list} to metals {metal_list}")

    # Apply name-encoded states before matching element symbols.
    overridden = []
    for name, ox in reported_list:
        special = match_special_metal_state(name)
        if special and not EXPLICIT_NUMERAL.search(name):
            ox = special[1]
        overridden.append((name, ox))
    reported_list = overridden

    if metallocene is not None and any(
        match_special_metal_state(name) for name, _ in reported_list
    ):
        return _match_with_metallocenes(metal_list, reported_list, metallocene)

    expanded = expand_reported_metal_states(reported_list, metal_list)
    if not expanded:
        return [f"{m}_unknown" for m in metal_list]

    # Names such as bis(ferrocenyl) mention the moiety once for several units.
    # With only name-encoded reports, every atom of that element belongs to
    # those moieties, so the last state is reused; another iron mentioned in
    # the name is reported separately and takes the branch below.
    if all(match_special_metal_state(name) for name, _ in reported_list):
        states = defaultdict(list)
        for name, ox in reported_list:
            sym = match_special_metal_state(name)[0]
            states[sym].extend(special_unit_states(name, ox))
        result = []
        for metal in metal_list:
            queue = states[metal]
            ox = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else None
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


def standardize_reported_metal_oxidation_states(tms, reported, metallocene=None):
    """Return matched element_state tokens and the fraction with known oxidation states."""
    matched = match_reported_metal_states(tms, reported, metallocene)
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
