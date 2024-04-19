import numpy as np
import pickle
import os
from cell2mol import __file__
from cell2mol.coordination_sphere import shape_structure_references_simplified
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()


#######################################################
def predict_ox_state (metal:object, debug: int=0) -> None:
    model = ""
    feature = generate_feature_vector (metal, target_prop = "m_ox", debug=debug)
    path_rf = os.path.join( os.path.abspath(os.path.dirname(__file__)), model)
    ramdom_forest = pickle.load(open(path_rf, 'rb'))
    predictions = ramdom_forest.predict(feature)
    m_ox_rf = predictions[0]
    return m_ox_rf

#######################################################
def assign_spin_metal (metal:object, debug: int=0) -> None:
    """ Assigns spin multiplicity of the transition metal.
    """
    valence_elec = metal.get_valence_elec(metal.charge)
    period = elemdatabase.elementperiod[metal.label]

    if period == 4:  # 3d transition metals
        if valence_elec in [0, 10]:                                        return 1
        elif valence_elec in [1, 9]:                                       return 2
        elif valence_elec in [2, 3] and metal.get_parent("molecule").is_haptic == False :         return (valence_elec + 1)
        elif valence_elec in [4, 5, 6, 7, 8] or (valence_elec in [2, 3] and metal.get_parent("molecule").is_haptic == True) :
            # Predict spin multiplicity of metal based on Random forest model
            feature = generate_feature_vector (metal, target_prop="spin", debug=debug)
            path_rf = os.path.join( os.path.abspath(os.path.dirname(__file__)), "total_spin_3131.pkl")
            ramdom_forest = pickle.load(open(path_rf, 'rb'))
            predictions = ramdom_forest.predict(feature)
            spin_rf = predictions[0]
            print(f"ASSIGN_SPIN_METAL: Spin multiplicity of the metal {metal.label} is predicted as {spin_rf} using Random Forest model")
            return spin_rf
        else :
            print("ASSIGN_SPIN_METAL: Error! Spin multiplicity could not be assigned to the metal with valence electrons: ", valence_elec)
            return None
    else :      # 4d and 5d transition metals
        if valence_elec % 2 == 0:   return 1
        else:                       return 2

#######################################################
def assign_spin_complexes (mol:object, debug: int=0) -> None:
    """ Assigns spin multiplicity of the transition metal complexes.
    """
    for metal in mol.metals:
        if not hasattr(metal,"spin"): metal.get_spin(debug=debug)
    metals_spin = [metal.spin for metal in mol.metals]
    if debug >=2: print(f"ASSIGN_SPIN_COMPLEXES: {metals_spin=}")

    if any(ligand.is_nitrosyl for ligand in mol.ligands):       return None
    else :
        if None in metals_spin :                        return None
        elif len(metals_spin) == 1:                     return metals_spin[0]       # Mononuclear complex
        else :                                                                      # Polynuclear complex                       
            metals_idx_not_singlet = [idx for idx, spin in enumerate(metals_spin) if spin != 1]
            if len(metals_idx_not_singlet) == 0 :        return 1
            elif len(metals_idx_not_singlet) == 1 :      return metals_spin[metals_idx_not_singlet[0]]
            else :                                       return None          


#######################################################
def generate_feature_vector (metal: object, target_prop: str, debug: int = 0) -> np.ndarray:
    """ Generate feature vector for a given transition metal coordination complex
    Args:
        metal (obj): metal atom object
    Returns:
        feature (np.ndarray): feature vector
    """
    if debug > 1: print(f"GENERATE_feature_vector: {metal.label}")

    elem_nr = elemdatabase.elementnr[metal.label]
    m_ox = metal.charge
    valence_elec = metal.get_valence_elec(metal.charge)
    if debug > 1: print(f"GENERATE_feature_vector: {elem_nr=} {m_ox=} {valence_elec=}")
    
    coord_group = metal.get_connected_groups()
    coord_nr = metal.coord_nr
    geom_nr = make_geom_list()[metal.coord_geometry]
    if debug > 1: print(f"GENERATE_feature_vector: {metal.coord_nr=} {metal.coord_geometry=} {geom_nr=}")

    rel_metal_radius = metal.rel_metal_radius
    if debug > 1: print(f"GENERATE_feature_vector: {metal.rel_metal_radius=}")

    coord_hapticty = [ group.is_haptic for group in coord_group ]
    if any(coord_hapticty) :    hapticity = 1
    else :                      hapticity = 0
    if debug > 1: print(f"GENERATE_feature_vector: {hapticity=}")
    
    if target_prop == "m_ox":
        feature = np.array([[elem_nr, coord_nr, geom_nr, rel_metal_radius, hapticity]])
        if debug > 1: print(f"GENERATE_feature_vector: {feature=}")
    elif target_prop == "spin":
        feature = np.array([[elem_nr, m_ox, valence_elec, coord_nr, geom_nr, rel_metal_radius, hapticity]])
        if debug > 1: print(f"GENERATE_feature_vector: {feature=}")
    
    return feature

#######################################################
def make_geom_list ():

    geom_list = {}
    count = 0
    for i in shape_structure_references_simplified.values():
    #     print(np.array(i)[:,3])
        for geom in np.array(i)[:,3]:
            geom_list[geom] = count
            count +=1
    return geom_list

#######################################################