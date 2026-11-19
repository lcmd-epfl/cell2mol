"""Determine metal and complex spin multiplicities from rules and model features."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import joblib
import numpy as np
from numpy.typing import NDArray

from cell2mol import __file__
from cell2mol.coordination_sphere import shape_structure_references_simplified
from cell2mol.elementdata import ElementData
from cell2mol.my_types import Spin

if TYPE_CHECKING:
    from cell2mol.classes.metal import Metal
    from cell2mol.classes.molecule import Molecule

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def predict_metal_spin(metal: Metal) -> Spin | None:
    """Predict metal spin multiplicity (2S + 1) from its assigned charge.

    Use electron-count rules for 3d d0/d1/d9/d10; use the Random Forest
    model for the remaining 3d d2–d8 cases, correcting predictions that are
    impossible for the electron count. For heavier d-block metals,
    use 1 for even and 2 for odd valence electron counts.

    Return None if spin cannot be determined.
    Raise ValueError if required data is missing.
    Store the valence electron count and return the spin multiplicity.
    """
    charge = metal.charge
    if charge is None:
        raise ValueError(f"Metal {metal.label} has no assigned charge")
    valence_elec = metal.compute_valence_electron_count(charge)
    period = elemdatabase.elementperiod[metal.label]
    block = elemdatabase.elementblock[metal.label]

    if period == 4 and block == "d":
        if valence_elec in [0, 10]:
            return 1
        if valence_elec in [1, 9]:
            return 2
        if valence_elec in [2, 3, 4, 5, 6, 7, 8]:
            if metal.coord_geometry is None or metal.coord_geometry == "Undefined":
                logger.error(
                    "Cannot determine spin: geometry of %s is undefined", metal.label
                )
                return None

            feature = build_spin_feature_vector(metal)
            model = load_spin_model("TM-GSspinPlus_RandomForest.joblib")
            predicted = int(model.predict(feature)[0])
            spin, label = correct_spin(
                predicted, valence_elec, model.predict_proba(feature)[0], model.classes_
            )
            logger.info(
                "Metal %s spin predicted as %s (%s) by the Random Forest model "
                "(raw prediction %s)",
                metal.label,
                spin,
                label,
                predicted,
            )
            return spin

        logger.error(
            "Cannot determine spin: unsupported valence count %s", valence_elec
        )
        return None

    if period > 4 and block == "d":
        return 1 if valence_elec % 2 == 0 else 2
    return None


@lru_cache(maxsize=None)
def load_spin_model(filename: str):
    """Load a spin model from the models directory once per process."""
    return joblib.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", filename)
    )


def valid_spins(valence_elec: int) -> list[int]:
    """Allowed multiplicities (2S + 1) from LS to HS for a valence electron count."""
    unpaired = min(valence_elec, 10 - valence_elec)
    return list(range(valence_elec % 2 + 1, max(unpaired, 0) + 2, 2))


def spin_type(spin: int, valence_elec: int) -> str:
    """LS / IS / HS label of a multiplicity, or 'invalid' if it is not allowed."""
    allowed = valid_spins(valence_elec)
    if spin not in allowed:
        return "invalid"
    if spin == allowed[0]:
        return "LS"
    if spin == allowed[-1]:
        return "HS"
    return "IS"


def correct_spin(
    spin: int,
    valence_elec: int,
    proba: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> tuple[int, str]:
    """Replace an impossible multiplicity with the corresponding valid spin type.

    Above HS (e.g. d2 quartet) -> HS; below LS -> LS; wrong parity between LS
    and HS -> the neighbouring valid multiplicity with the larger predicted
    probability (lower spin on ties or without probabilities).
    Return (spin, spin type).
    """
    allowed = valid_spins(valence_elec)
    if spin in allowed:
        return spin, spin_type(spin, valence_elec)
    if spin > allowed[-1]:
        new = allowed[-1]
    elif spin < allowed[0]:
        new = allowed[0]
    else:
        lower, upper = spin - 1, spin + 1
        new = lower
        if proba is not None and classes is not None:
            p = dict(zip(classes.tolist(), proba))
            if p.get(upper, 0.0) > p.get(lower, 0.0):
                new = upper
    return new, spin_type(new, valence_elec)


def predict_corrected(
    model, X: NDArray[np.float64]
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Predict multiplicities and correct the impossible ones; d_elec is X[:, 2].

    Return (raw predictions, corrected predictions).
    """
    pred = model.predict(X)
    probs = model.predict_proba(X)
    corrected = np.array(
        [
            correct_spin(int(s), int(n), p, model.classes_)[0]
            for s, n, p in zip(pred, X[:, 2], probs)
        ]
    )
    return pred, corrected


def determine_complex_spin(mol: Molecule) -> Spin | None:
    """Determine complex multiplicity from the constituent metal spins.

    Assign missing d-block metal spins and check for nitrosyl ligands.
    Return 1 if all metals are singlets, or the spin of the only non-singlet metal.
    Return None for nitrosyls, missing spins, or more than one non-singlet metal.
    """
    metals = mol.metals or []
    ligands = mol.ligands or []
    for metal in metals:
        if metal.spin is None and elemdatabase.elementblock[metal.label] == "d":
            metal.assign_spin()
    for ligand in ligands:
        if ligand.is_nitrosyl is None:
            ligand.detect_nitrosyl()

    metal_spins = [metal.spin for metal in metals]
    logger.info("Metal spin multiplicities: %s", metal_spins)
    if any(ligand.is_nitrosyl for ligand in ligands) or any(
        spin is None for spin in metal_spins
    ):
        return None

    nonsinglet_spins = [spin for spin in metal_spins if spin is not None and spin != 1]
    if not nonsinglet_spins:
        return 1
    if len(nonsinglet_spins) == 1:
        return nonsinglet_spins[0]
    return None


def build_spin_feature_vector(metal: Metal) -> NDArray[np.float64]:
    """Build a float array for the metal spin prediction model.

    Features follow training order: atomic number, charge, valence electron
    count, coordination number, coordination geometry index, relative metal
    radius, and hapticity flag. Raise ValueError for missing charge or
    coordination data; an unknown geometry name raises KeyError.
    """
    charge = metal.charge
    geometry = metal.coord_geometry
    if charge is None:
        raise ValueError(f"Metal {metal.label} has no assigned charge")
    if geometry is None or geometry == "Undefined":
        raise ValueError(f"Metal {metal.label} has no defined coordination geometry")
    if metal.coord_nr is None or metal.rel_metal_radius is None:
        raise ValueError(f"Metal {metal.label} has incomplete coordination features")

    groups = metal.get_connected_groups() or []
    feature = np.array(
        [
            [
                elemdatabase.elementnr[metal.label],
                charge,
                metal.compute_valence_electron_count(charge),
                metal.coord_nr,
                build_geometry_index()[geometry],
                metal.rel_metal_radius,
                int(any(group.is_haptic for group in groups)),
            ]
        ],
        dtype=np.float64,
    )
    logger.info("Spin feature vector: %s", feature)
    return feature


def build_geometry_index() -> dict[str, int]:
    """Map geometry names to integer indices used by the spin prediction model.

    Preserve this order to keep feature encoding consistent with training.
    """
    indices: dict[str, int] = {}
    count = 0
    for shapes in shape_structure_references_simplified.values():
        for geometry in np.array(shapes)[:, 3]:
            indices[str(geometry)] = count
            count += 1
    return indices
