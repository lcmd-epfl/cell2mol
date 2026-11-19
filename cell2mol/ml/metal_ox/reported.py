"""
Score predicted metal oxidation states against the CIF-reported ones
(`refcell.reported_metal_os_matched`, e.g. ['Fe_2', 'Fe_3', 'Pd_0']).

Comparison is per refcode and assignment-free, the same way cell2mol's own
m_ox_agree check works (compare.py): predictions become "<element>_<ox>" tags
and are compared to the reported tags as multisets. Pairing a specific atom
with a specific reported token is not reliable: match_reported_metal_states
orders its output differently depending on the case (one report, several
reports grouped by element, metallocene/special names), and for two atoms of
one element with different reported states (mixed valence) the CIF text
does not say which atom is which.

  n_correct  size of the multiset intersection = the best achievable number
             of correctly predicted metals over all atom<->token pairings
             (only same-element pairings can match)
  exact      the multisets are equal (every metal right)

As in cell2mol, a refcode is scored only when every metal has a known
reported state (metal_os_match_confidence == 1.0) and the token count equals
the metal count.
"""

from __future__ import annotations

from collections import Counter


def parse_tag(token: str) -> tuple[str, int | None]:
    """"Pd_2" -> ("Pd", 2); "Pd_-2" -> ("Pd", -2); "Pd_unknown" -> ("Pd", None)."""
    element, _, charge_text = token.rpartition("_")
    try:
        return element, int(charge_text)
    except ValueError:
        return element, None


def reported_tags(refcell, n_metals: int) -> tuple[list[str] | None, str]:
    """(tags, "ok") if the refcode can be scored, else (None, reason)."""
    matched = getattr(refcell, "reported_metal_os_matched", None)
    confidence = getattr(refcell, "metal_os_match_confidence", None)

    if not getattr(refcell, "reported_metal_os", None) or not matched:
        return None, "no_reported_os"
    if confidence is None or confidence < 1.0:
        return None, "unknown_reported_os"
    if len(matched) != n_metals:
        return None, "count_mismatch"
    if any(parse_tag(t)[1] is None for t in matched):
        return None, "unknown_reported_os"
    return list(matched), "ok"


def tags(labels: list[str], states: list[int | None]) -> list[str]:
    return [f"{label}_{state}" for label, state in zip(labels, states)]


def score(predicted: list[str], reported: list[str]) -> tuple[int, bool]:
    """(n_correct, exact) -- see module docstring."""
    p, r = Counter(predicted), Counter(reported)
    return sum((p & r).values()), p == r


def unambiguous_reported(label: str, reported: list[str] | None) -> int | None:
    """The reported state of `label` if all its tokens agree (always true for
    a single atom of that element), else None."""
    if not reported:
        return None
    states = {parse_tag(t)[1] for t in reported if parse_tag(t)[0] == label}
    return states.pop() if len(states) == 1 else None
