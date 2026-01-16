import numpy as np
import pickle
import joblib
import os
from cell2mol import __file__
from cell2mol.coordination_sphere import shape_structure_references_simplified
from cell2mol.elementdata import ElementData
import logging

logger = logging.getLogger(__name__)
elemdatabase = ElementData()


def predict_ox_state(metal: object) -> None:
    model = ""
    feature = generate_feature_vector(metal, target_prop="m_ox")
    path_rf = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    ramdom_forest = pickle.load(open(path_rf, "rb"))
    predictions = ramdom_forest.predict(feature)
    m_ox_rf = predictions[0]
    return m_ox_rf


def assign_spin_metal(metal: object) -> None:
    """Assigns spin multiplicity of the transition metal."""
    valence_elec = metal.get_valence_elec(metal.charge)
    period = elemdatabase.elementperiod[metal.label]
    block = elemdatabase.elementblock[metal.label]
    if period == 4 and block == "d":  # 3d transition metals
        if valence_elec in [0, 10]:
            return 1
        elif valence_elec in [1, 9]:
            return 2
        elif valence_elec in [2, 3] and not metal.get_parent("molecule").is_haptic:
            return valence_elec + 1
        elif valence_elec in [4, 5, 6, 7, 8] or (
            valence_elec in [2, 3] and metal.get_parent("molecule").is_haptic
        ):
            if metal.coord_geometry is not None and metal.coord_geometry != "Undefined":
                # Predict spin multiplicity of metal based on Random Forest model
                feature = generate_feature_vector(metal, target_prop="spin")
                # path_rf = os.path.join(
                #     os.path.dirname(os.path.abspath(__file__)),
                #     "models",
                #     "TM-GSspin_RandomForest.pkl",
                # )
                # ramdom_forest = pickle.load(open(path_rf, "rb"))
                path_rf = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "models",
                    "TM-GSspinPlus_RandomForest.joblib",
                )
                ramdom_forest = joblib.load(path_rf)
                predictions = ramdom_forest.predict(feature)
                spin_rf = predictions[0]
                logger.info(
                    "ASSIGN_SPIN_METAL: Spin multiplicity of the metal %s is predicted as %s using Random Forest model",
                    metal.label,
                    spin_rf,
                )
                return spin_rf
            else:
                logger.error(
                    "Cannot assign spin multiplicity! Coordination geometry of the metal %s is not defined.",
                    metal.label,
                )
                return None
        else:
            logger.error(
                "Cannot assign spin multiplicity! valence electrons of metal: %s",
                valence_elec,
            )
            return None
    elif period > 4 and block == "d":  # 4d and 5d transition metals
        if valence_elec % 2 == 0:
            return 1
        else:
            return 2
    else:  # other metals
        return None


def assign_spin_complexes(mol: object) -> None:
    """Assigns spin multiplicity of the transition metal complexes."""
    for metal in mol.metals:
        if metal.spin is None and (elemdatabase.elementblock[metal.label] == "d"):
            metal.get_spin()
    for ligand in mol.ligands:
        if ligand.is_nitrosyl is None:
            ligand.evaluate_as_nitrosyl()

    metals_spin = [metal.spin for metal in mol.metals if metal.spin is not None]
    logger.info("Spin multiplicity of metals: %s", metals_spin)

    if any([ligand.is_nitrosyl for ligand in mol.ligands]):
        return None
    else:
        if None in metals_spin:
            return None
        elif len(metals_spin) == 1:
            return metals_spin[0]  # Mononuclear complex
        else:  # Polynuclear complex
            metals_idx_not_singlet = [
                idx for idx, spin in enumerate(metals_spin) if spin != 1
            ]
            if len(metals_idx_not_singlet) == 0:
                return 1
            elif len(metals_idx_not_singlet) == 1:
                return metals_spin[metals_idx_not_singlet[0]]
            else:
                return None


def generate_feature_vector(metal: object, target_prop: str) -> np.ndarray:
    """Generate feature vector for a given transition metal coordination complex
    Args:
        metal (obj): metal atom object
        target_prop (str): target property to predict, either "m_ox" for metal oxidation state or "spin" for spin multiplicity
    Returns:
        feature (np.ndarray): feature vector
    """
    elem_nr = elemdatabase.elementnr[metal.label]

    coord_group = metal.get_connected_groups()
    coord_nr = metal.coord_nr
    geom_nr = make_geom_list()[metal.coord_geometry]
    rel_metal_radius = metal.rel_metal_radius

    coord_hapticty = [group.is_haptic for group in coord_group]
    if any(coord_hapticty):
        hapticity = 1
    else:
        hapticity = 0

    if target_prop == "m_ox":
        feature = np.array([[elem_nr, coord_nr, geom_nr, rel_metal_radius, hapticity]])
        logger.info("feature_vector: %s", feature)
    elif target_prop == "spin":
        m_ox = metal.charge
        valence_elec = metal.get_valence_elec(metal.charge)
        feature = np.array(
            [
                [
                    elem_nr,
                    m_ox,
                    valence_elec,
                    coord_nr,
                    geom_nr,
                    rel_metal_radius,
                    hapticity,
                ]
            ]
        )
        logger.info("feature_vector: %s", feature)

    return feature


def make_geom_list():
    geom_list = {}
    count = 0
    for i in shape_structure_references_simplified.values():
        for geom in np.array(i)[:, 3]:
            geom_list[geom] = count
            count += 1
    return geom_list
