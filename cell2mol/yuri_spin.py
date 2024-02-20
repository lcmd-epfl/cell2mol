import numpy as np
import pickle
import os
from cell2mol import __file__
import sklearn
from cell2mol.coordination_sphere import *
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()

#######################################################
def assign_spin_metal (metal:object) -> None:
    """ Assigns spin multiplicity of the transition metal.
    """
    valence_elec = metal.count_valence_elec (metal.totcharge)
    period = elemdatabase.elementperiod[metal.label]

    if period == 4:  # 3d transition metals
        if valence_elec in [0, 10]:                                        return 1
        elif valence_elec in [1, 9]:                                       return 2
        elif valence_elec in [2, 3] and metal.hapticity == False :         return (valence_elec + 1)
        elif valence_elec in [4, 5, 6, 7, 8] or (valence_elec in [2, 3] and metal.hapticity == True) :
            # Predict spin multiplicity of metal based on Random forest model
            feature = generate_feature_vector (metal)
            path_rf = os.path.join( os.path.abspath(os.path.dirname(__file__)), "total_spin_3131.pkl")
            ramdom_forest = pickle.load(open(path_rf, 'rb'))
            predictions = ramdom_forest.predict(feature)
            spin_rf = predictions[0]
            return spin_rf
        else :
            print("Error: Spin multiplicity could not be assigned to the metal with valence electrons: ", valence_elec)
            return None
    else :      # 4d and 5d transition metals
        if valence_elec % 2 == 0:   return 1
        else:                       return 2

#######################################################
def assign_spin_complexes (mol:object) -> None:
    """ Assigns spin multiplicity of the transition metal complexes.
    """
    for metal in mol.metals:
        if not hasattr(metal,"spin"): metal.get_spin()
    metals_spin = [metal.spin for metal in mol.metals]

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
def generate_feature_vector (metal: object, debug: int = 0) -> np.ndarray:
    """ Generate feature vector for a given transition metal coordination complex
    Args:
        metal (obj): metal atom object
    Returns:
        feature (np.ndarray): feature vector
    """
    elem_nr = elemdatabase.elementnr[metal.label]
    m_ox = metal.totcharge
    valence_elec = metal.count_valence_elec (metal.totcharge)

    coord_group = metal.get_connected_groups()
    coord_nr = len(coord_group)

    coord_geometry = get_coordination_geometry(coord_group, debug = debug)
    geom_nr = make_geom_list()[coord_geometry]

    rel_metal_radius = metal.get_relative_metal_radius(debug = debug)
    
    coord_hapticty = [ group.hapticity for group in coord_group ]
    if any(coord_hapticty) :    hapticity = 1
    else :                      hapticity = 0

    feature = np.array([[elem_nr, m_ox, valence_elec, coord_nr, geom_nr, rel_metal_radius, hapticity]])
    
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