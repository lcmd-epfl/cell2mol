import numpy as np
import os
from cell2mol import __file__
import yaml
from cell2mol.other import *
from cell2mol.connectivity import add_atom, get_adjmatrix, get_adjmatrix_from_cif_bonds, is_single_ring
from cell2mol.elementdata import ElementData
from scipy.optimize import linear_sum_assignment      # Hungarian algorithm
from scipy.stats import special_ortho_group           # more evenly distributed 
from scipy.linalg import svd
elemdatabase = ElementData()

#######################################################
# Load YAML file
path = os.path.join( os.path.abspath(os.path.dirname(__file__)), "ideal_structures_center.yaml")
with open(path, "r") as file:
    data = yaml.safe_load(file)
ideal_shapes_from_cosymlib = {key: np.array(value) for key, value in data.items()}
#######################################################
###     Define coordination geometry from groups    ### 
#######################################################
def define_coordination_geometry (metal: object, coord_group: list, debug: int=0) -> object:
    
    symbols = []
    positions = []
    coord_haptic_type = []

    symbols.append(metal.label)
    positions.append(metal.coord)
    if debug >= 2 : print(f"METAL.DEFINE_coordination_geometry: {metal.label} {metal.coord}")
    if debug >= 1 :print(f"METAL.DEFINE_coordination_geometry: coord_group formula {[group.formula for group in coord_group]}")
    if debug >= 1 :print(f"METAL.DEFINE_coordination_geometry: coord_group hapticity {[group.is_haptic if group.subtype != 'metal' else False for group in coord_group]}")
    if debug >= 1 :print(f"METAL.DEFINE_coordination_geometry: coord_group atoms{[[a.label for a in group.atoms] if group.subtype != 'metal' else [group.label] for group in coord_group]}")

    count = 0
    for group in coord_group:
        if group.subtype == 'metal':
            symbols.append(group.label)
            positions.append(group.coord)
            count += 1    
        elif group.is_haptic == False:
            for atom in group.atoms:
                symbols.append(atom.label)
                positions.append(atom.coord)
                count += 1
                if debug >= 2 : print("METAL.DEFINE_coordination_geometry:", atom.label, atom.coord)
        else :
            if debug >= 2 : print(f"METAL.DEFINE_coordination_geometry: {group.haptic_type=}")
            #if debug >= 2 : print(f"METAL.DEFINE_coordination_geometry: {[atom.coord for atom in group.atoms]}")
            haptic_center_coord = compute_centroid(np.array([atom.coord for atom in group.atoms]))
            symbols.append(str(group.haptic_type))
            positions.append(haptic_center_coord.tolist())
            count += 1      
            if debug >= 2 : print(f"mid point of {group.haptic_type=}", haptic_center_coord)      
            coord_haptic_type.append(group.haptic_type)             

    posgeom_dev = shape_measure(symbols, positions, debug=debug)
    coord_nr = count
    if len(posgeom_dev) > 0:
        coordination_geometry=min(posgeom_dev, key=posgeom_dev.get)
        geom_deviation=min(posgeom_dev.values())
    else :
        coordination_geometry = "Undefined"
        geom_deviation = "Undefined"

    if debug >= 2 :
        # for haptic ligands, it's the mid point of haptic ligands
        print(f"METAL.DEFINE_coordination_geometry: The number of coordinating points: {coord_nr}")
        print(f"METAL.DEFINE_coordination_geometry: {posgeom_dev}")
        print(f"METAL.DEFINE_coordination_geometry: The type of hapticity : {coord_haptic_type}")
    
    if debug >= 1 : 
        print(f"METAL.DEFINE_coordination_geometry: The most likely geometry is '{coordination_geometry}' with deviation value {geom_deviation}")

    # return coordination_geometry
    return coord_nr, coordination_geometry, geom_deviation
#######################################################
def shape_measure (symbols: list, positions: list, debug: int=0) -> dict:
    # Get shape measure of a set of coordinates

    if debug >= 2:print(f"SHAPE_MEASURE: {symbols=}")
    if debug >= 2:print(f"SHAPE_MEASURE: {positions=}")

    cn = len(symbols)-1 # coordination number of metal center
    if debug >= 2: print(f"SHAPE_MEASURE: coordination number of metal center {cn}")
    
    if cn == 0 : 
        posgeom_dev = {}
    elif cn == 1 :
        posgeom_dev = {'Linear' : 0.0}
    else :
        posgeom_dev={}
        try :
            ref_geom = np.array(shape_structure_references_simplified['{} Vertices'.format(cn)], dtype=object)
            ideal_shapes = {}
            for idx, rg in enumerate(ref_geom[:,0]):
                geom = ref_geom[:,3][idx]
                ideal_shapes[geom]=ideal_shapes_from_cosymlib[rg]
            print(f"SHAPE_MEASURE: Ideal_shapes: {ideal_shapes.keys()}")
            for geom, ideal_shape in ideal_shapes.items():
                chsm = calc_cshm_fast(positions, ideal_shape)
                posgeom_dev[geom]=round(float(chsm), 3)
        except:
            print(f"SHAPE_MEASURE: {cn} Vertices not found in shape_structure_references")

    return posgeom_dev
#######################################################
def shape_measure_old (symbols: list, positions: list, debug: int=0) -> dict:
    from cosymlib import Geometry
    # Get shape measure of a set of coordinates

    if debug >= 2:print(f"SHAPE_MEASURE: {symbols=}")
    if debug >= 2:print(f"SHAPE_MEASURE: {positions=}")

    cn = len(symbols)-1 # coordination number of metal center

    connectivity= [[1, i] for i in range(2, cn+2)]
    if debug >= 2: print(f"SHAPE_MEASURE: coordination number of metal center {cn}")
    if debug >= 2: print(f"SHAPE_MEASURE: connectivity of metal center(1) {connectivity}")
    geometry = Geometry(positions=positions, 
                        symbols=symbols, 
                        connectivity=connectivity)            
    
    if cn == 0 : 
        posgeom_dev = {}
    elif cn == 1 :
        posgeom_dev = {'Linear' : 0.0}
    else :
        posgeom_dev={}
        try :
            ref_geom = np.array(shape_structure_references_simplified['{} Vertices'.format(cn)], dtype=object)
            for idx, rg in enumerate(ref_geom[:,0]):
                shp_measure = geometry.get_shape_measure(rg, central_atom=1)
                geom = str(ref_geom[:,3][idx])
                posgeom_dev[geom]=round(shp_measure, 3)      
        except:
            print(f"SHAPE_MEASURE: {cn} Vertices not found in shape_structure_references")

    return posgeom_dev

#######################################################
shape_structure_references_simplified = {'2 Vertices': [['L-2', 1, 'Dinfh', 'Linear'],
                                        ['vT-2', 2, 'C2v', 'Bent (V-shape, 109.47°)'],
                                        ['vOC-2', 3, 'C2v', 'Bent (L-shape, 90°)']],

                        '3 Vertices': [['TP-3', 1, 'D3h', 'Trigonal planar'],
                                       ['fvOC-3', 3, 'C3v', 'fac-Trivacant octahedron'],
                                       ['mvOC-3', 4, 'C2v', 'T-shaped']],

                        '4 Vertices': [['T-4', 2, 'Td', 'Tetrahedral'],
                                        ['SP-4', 1, 'D4h', 'Square planar'],
                                        ['SS-4', 3, 'C2v', 'Seesaw']],

                        '5 Vertices': [['PP-5', 1, 'D5h', 'Pentagon'],
                                        ['TBPY-5', 3, 'D3h', 'Trigonal bipyramidal'],
                                        ['SPY-5', 4, 'C4v', 'Square pyramidal']],

                        '6 Vertices': [['HP-6', 1, 'D6h', 'Hexagon'],
                                        ['PPY-6', 2, 'C5v', 'Pentagonal pyramidal'],
                                        ['OC-6', 3, 'Oh', 'Octahedral'],
                                        ['TPR-6', 4, 'D3h', 'Trigonal prismatic']],

                        '7 Vertices': [['HP-7', 1, 'D7h', 'Heptagon'],
                                        ['HPY-7', 2, 'C6v', 'Hexagonal pyramidal'],
                                        ['PBPY-7', 3, 'D5h', 'Pentagonal bipyramidal'],
                                        ['CTPR-7', 5, 'C2v', 'Capped trigonal prismatic']],

                        '8 Vertices': [['OP-8', 1, 'D8h', 'Octagon'],
                                        ['HPY-8', 2, 'C7v', 'Heptagonal pyramidal'],
                                        ['HBPY-8', 3, 'D6h', 'Hexagonal bipyramidal'],
                                        ['CU-8', 4, 'Oh', 'Cube'],
                                        ['SAPR-8', 5, 'D4d', 'Square antiprismatic'],
                                        ['TDD-8', 6, 'D2d', 'Dodecahedral']],

                        '9 Vertices': [['EP-9', 1, 'D9h', 'Enneagon'],
                                        ['OPY-9', 2, 'C8v', 'Octagonal pyramid'],
                                        ['HBPY-9', 3, 'D7h', 'Heptagonal bipyramid'],
                                        ['JTC-9', 4, 'C3v', 'Johnson triangular cupola J3'],
                                        ['JCCU-9', 5, 'C4v', 'Capped cube J8'],
                                        ['CCU-9', 6, 'C4v', 'Spherical-relaxed capped cube'],
                                        ['JCSAPR-9', 7, 'C4v', 'Capped square antiprism J10'],
                                        ['CSAPR-9', 8, 'C4v', 'Spherical capped square antiprism'],
                                        ['JTCTPR-9', 9, 'D3h', 'Tricapped trigonal prism J51'],
                                        ['TCTPR-9', 10, 'D3h', 'Spherical tricapped trigonal prism'],
                                        ['JTDIC-9', 11, 'C3v', 'Tridiminished icosahedron J63'],
                                        ['HH-9', 12, 'C2v', 'Hula-hoop'],
                                        ['MFF-9', 13, 'Cs', 'Muffin']],

                        '10 Vertices': [['DP-10', 1, 'D10h', 'Decagon'],
                                        ['EPY-10', 2, 'C9v', 'Enneagonal pyramid'],
                                        ['OBPY-10', 3, 'D8h', 'Octagonal bipyramid'],
                                        ['PPR-10', 4, 'D5h', 'Pentagonal prism'],
                                        ['PAPR-10', 5, 'D5d', 'Pentagonal antiprism'],
                                        ['JBCCU-10', 6, 'D4h', 'Bicapped cube J15'],
                                        ['JBCSAPR-10', 7, 'D4d', 'Bicapped square antiprism J17'],
                                        ['JMBIC-10', 8, 'C2v', 'Metabidiminished icosahedron J62'],
                                        ['JATDI-10', 9, 'C3v', 'Augmented tridiminished icosahedron J64'],
                                        ['JSPC-10', 10, 'C2v', 'Sphenocorona J87'],
                                        ['SDD-10', 11, 'D2', 'Staggered Dodecahedron (2:6:2)'],
                                        ['TD-10', 12, 'C2v', 'Tetradecahedron (2:6:2)'],
                                        ['HD-10', 13, 'D4h', 'Hexadecahedron (2:6:2) or (1:4:4:1)']],
                        '11 Vertices': [['HP-11', 1, 'D11h', 'Hendecagon'],
                                        ['DPY-11', 2, 'C10v', 'Decagonal pyramid'],
                                        ['EBPY-11', 3, 'D9h', 'Enneagonal bipyramid'],
                                        ['JCPPR-11', 4, 'C5v', 'Capped pentagonal prism J9'],
                                        ['JCPAPR-11', 5, 'C5v', 'Capped pentagonal antiprism J11'],
                                        ['JAPPR-11', 6, 'C2v', 'Augmented pentagonal prism J52'],
                                        ['JASPC-11', 7, 'Cs', 'Augmented sphenocorona J87']],
                        '12 Vertices': [['DP-12', 1, 'D12h', 'Dodecagon'],
                                        ['HPY-12', 2, 'C11v', 'Hendecagonal pyramid'],
                                        ['DBPY-12', 3, 'D10h', 'Decagonal bipyramid'],
                                        ['HPR-12', 4, 'D6h', 'Hexagonal prism'],
                                        ['HAPR-12', 5, 'D6d', 'Hexagonal antiprism'],
                                        ['TT-12', 6, 'Td', 'Truncated tetrahedron'],
                                        ['COC-12', 7, 'Oh', 'Cuboctahedron'],
                                        ['ACOC-12', 8, 'D3h', 'Anticuboctahedron J27'],
                                        ['IC-12', 9, 'Ih', 'Icosahedron'],
                                        ['JSC-12', 10, 'C4v', 'Johnson square cupola J4'],
                                        ['JEPBPY-12', 11, 'D6h', 'Johnson elongated pentagonal bipyramid J16'],
                                        ['JBAPPR-12', 12, 'C2v', 'Biaugmented pentagonal prism J53'],
                                        ['JSPMC-12', 13, 'Cs', 'Sphenomegacorona J88']],
                        '20 Vertices': [['DD-20', 1, 'Ih', 'Dodecahedron']],
                        '24 Vertices': [['TCU-24', 1, 'Oh', 'Truncated cube'],
                                        ['TOC-24', 2, 'Oh', 'Truncated octahedron']],
                        '48 Vertices': [['TCOC-48', 1, 'Oh', 'Truncated cuboctahedron']],
                        '60 Vertices': [['TRIC-60', 1, 'Ih', 'Truncated icosahedron (fullerene)']]}

########################################################
# From https://github.com/radi0sus/cshm-cc/blob/main/cshm-cc.py
def normalize_structure(coordinates):
    # center and normalize the structure for CShM calculations
    centered_coords = coordinates - np.mean(coordinates, axis=0)
    norm = np.sqrt(np.mean(np.sum(centered_coords**2, axis=1)))
    return centered_coords / norm

########################################################
# From https://github.com/radi0sus/cshm-cc/blob/main/cshm-cc.py
def calc_cshm_fast(coordinates, ideal_shape, num_trials=100):
    # faster Hungarian algorithm optimization
    # check number of trials, if it is to low, it calculates the
    # local and not the global minimum
    input_structure = normalize_structure(coordinates)
    ideal_sq_norms = np.sum(ideal_shape**2)
    # try different rotations first, then optimize assignment
    min_cshm = float('inf')
    
    # generate some initial rotations to avoid local minima
    for trial in range(num_trials):
        if trial == 0:
            # first trial with identity rotation
            R_init = np.eye(3)
        else:
            # random rotation matrix for subsequent trials
            # generate a random rotation matrix 
            
            R_init = special_ortho_group.rvs(3)  
            
            # Ensure it's a proper rotation (det=1)
            if np.linalg.det(R_init) < 0:
                R_init[:, 0] *= -1
                
        # apply initial rotation to ideal shape
        rotated_ideal_init = np.dot(ideal_shape, R_init)
        
        # compute cost matrix based on squared Euclidean distances
        cost_matrix = np.linalg.norm(input_structure[:, None, :] - rotated_ideal_init[None, :, :], axis=2)
        
        # solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # rearrange ideal_shape based on optimal assignment
        permuted_ideal = ideal_shape[col_ind]

        # compute optimal rotation using SVD
        H = np.dot(input_structure.T, permuted_ideal)
        U, _, Vt = svd(H)
        R = np.dot(Vt.T, U.T)

        rotated_ideal = np.dot(permuted_ideal, R)
        scale = np.sum(input_structure * rotated_ideal) / ideal_sq_norms
        cshm = np.mean(np.sum((input_structure - scale * rotated_ideal) ** 2, axis=1))
        
        min_cshm = min(min_cshm, cshm)

    return min_cshm * 100

#######################################################    
def coordination_correction_for_nonhaptic(group: object, debug: int=0):

    if debug > 0: print("Entering COORD_CORR_NONHAPTIC:")
    if group.metals is None: group.get_connected_metals()
    if debug > 1: print(f"group: {[atom.label for atom in group.atoms]}")
    # Pair each atom with its index in the original list
    indexed_atoms = list(enumerate(group.atoms))

    # Sort the indexed list of atoms, prioritizing hydrogen atoms
    sorted_indexed_atoms = sorted(indexed_atoms, key=lambda x: (x[1].label != "H", x[1].label))
    if debug > 2: print("sorted_indexed_atoms:", sorted_indexed_atoms)
    # Extract the sorted atoms and their original indices into separate lists
    sorted_atoms = [atom[1] for atom in sorted_indexed_atoms]
    original_indices = [atom[0] for atom in sorted_indexed_atoms]

    ## First Correction (former verify_connectivity)
    conn_idx = []
    # conn_idx_by_metal = {met.atom_site_label : [] for met in group.metals}
    conn_idx_by_metal = {jdx : [] for jdx, met in enumerate(group.metals)}
    final_ligand_indices = []
    good_atoms = []
    removed_idx = []
    for idx, atom in zip(original_indices, sorted_atoms):
        if debug > 0: print(f"\tCoordinating atom label={atom.label} with mconnec={atom.mconnec}, original group index {idx}")
        isremoved = False
        ## Now there is an extra loop for each metal of the group. For bridging ligands
        #print(f"\t{[met in for met in group.metals]=}")
        for jdx, met in enumerate(group.metals):
            if isremoved: continue
            lig     = group.get_parent("ligand")
            ligand_idx = atom.get_parent_index("ligand")
            if debug > 0: print(f"\tevaluating coordination with metal {met.label}")
            if debug > 2: print(f"\n{met}")

            tmplabels = [atom.label, met.label]
            tmpcoord = [atom.coord, met.coord]            
            
            refcell = atom.get_parent("reference")
            if atom.atom_site_label is not None and met.atom_site_label is not None:
                atom_site_labels = [atom.atom_site_label, met.atom_site_label]
            else :
                atom_site_labels = None
            if refcell is not None and getattr(refcell, "exist_cif_bond_moiety", False):
                isconnected, tmpadjmat, tmpadjnum = get_adjmatrix_from_cif_bonds(tmplabels, tmpcoord, atom_site_labels, refcell.geom_bond_cif, metal_only=True)
            else:
                isconnected, tmpadjmat, tmpadjnum = get_adjmatrix(tmplabels, tmpcoord, metal_only=True)

            if isconnected and any(tmpadjnum) > 0: 
                if debug > 0 : 
                    print(f"\tAtom {atom.label} is connected to metal {met.label} (atom {ligand_idx=}) (metal group.metals index {jdx=})")
                
                if refcell is not None and getattr(refcell, "exist_cif_bond_moiety", False):
                    isadded  = True
                    if debug > 0: print(f"\tConnectivity verified for atom {atom.label} with ligand index {ligand_idx} based on CIF bonds")
                else:
                    isadded, newlab, newcoord = add_atom(lig.labels, lig.coord, ligand_idx, lig, list([met]), "H", removed_idx, debug=debug)
                    if debug >= 2:print(f"{removed_idx=}")
                
                if isadded:
                    if debug > 0: print(f"\tConnectivity verified for atom {atom.label} with ligand index {ligand_idx}")
                    conn_idx.append(idx)
                    final_ligand_indices.append(atom.get_parent_index("ligand"))
                    good_atoms.append(atom)
                    conn_idx_by_metal[jdx].append(idx)
                else:
                    if debug > 0: print(f"\tCORRECT mconnec of atom {atom.label} with ligand index {ligand_idx}")
                    isremoved = True
                    removed_idx.append(ligand_idx)
                    ### Reset Connectivity of the atom and the parents
                    atom.reset_mconnec(met, debug=1)
                    met.get_coord_sphere()
                    met.get_coord_sphere_formula()
            else:
                if debug > 0 : print(f"\tAtom {atom.label} is not connected to metal {met.label} (atom {ligand_idx=}) (metal group.metals index {jdx=})")

    
    print(f"conn_idx before set: {conn_idx=}")
    conn_idx = sorted(list(set(conn_idx)))
    split_groups = []
    for jdx, indices in conn_idx_by_metal.items():
        metal = group.metals[jdx]
        if indices:
            print(f"metal {metal.label} ({metal.atom_site_label}) connected to {[group.atoms[i].atom_site_label for i in indices]}")
            new_group = [i for i in indices]
            split_groups.append(new_group)
    print(f"conn_idx: {conn_idx=}")
    print(f"split_groups: {split_groups=}")
    final_group_indices = extract_final_indices(original_indices, split_groups)
    print("original_indices:", original_indices)
    print(f"final_group_indices: {final_group_indices=}")
    
    return group, final_group_indices, final_ligand_indices

#######################################################
def extract_final_indices(initial_list, intermediate_list):
    result = []
    seen = set()

    # Normalize flat list to nested list
    if intermediate_list and isinstance(intermediate_list[0], int):
        intermediate_list = [[i] for i in intermediate_list]

    for sublist in intermediate_list:
        group = []
        for idx in sublist:
            if idx not in seen:
                group.append(idx)
                # group.append(initial_list[idx])
                seen.add(idx)
        if group:
            result.append(group)

    return result

#######################################################
def coordination_correction_for_haptic(group: object, debug: int=0):
    add_factor = 0.45
    if debug > 0: print("Entering COORD_CORR_HAPTIC:")
    single_ring = is_single_ring(group.labels, group.coord)
    if debug > 0: print(f"Is single ring: {single_ring}")
    conn_idx = []
    conn_idx_by_metal = {jdx : [] for jdx, met in enumerate(group.metals)}
    for idx, atom in enumerate(group.atoms):
        for jdx, met in enumerate(group.metals):
            lig     = group.get_parent("ligand")
            ligand_idx = atom.get_parent_index("ligand")
            if debug > 2: print(f"\tevaluating coordination with metal {met.label} ({met.atom_site_label})")
            if debug > 2: print(f"\n{met}")

            tmplabels = [atom.label, met.label]
            tmpcoord = [atom.coord, met.coord]            
            
            refcell = atom.get_parent("reference")
            if atom.atom_site_label is not None and met.atom_site_label is not None:
                atom_site_labels = [atom.atom_site_label, met.atom_site_label]
            else :
                atom_site_labels = None
            if refcell is not None and getattr(refcell, "exist_cif_bond_moiety", False):
                isconnected, tmpadjmat, tmpadjnum = get_adjmatrix_from_cif_bonds(tmplabels, tmpcoord, atom_site_labels, refcell.geom_bond_cif, metal_only=True)
            else:
                isconnected, tmpadjmat, tmpadjnum = get_adjmatrix(tmplabels, tmpcoord, metal_only=True)

            if isconnected and any(tmpadjnum) > 0: 
                if debug > 0 : 
                    print(f"\tAtom {atom.label} ({ligand_idx=}) is connected to metal {met.label} ({met.atom_site_label}, group.metals index {jdx=})")
                conn_idx.append(idx)
                conn_idx_by_metal[jdx].append(idx)
            else:
                if debug > 0 : print(f"\tAtom {atom.label} ({ligand_idx=}) is not connected to metal {met.label} ({met.atom_site_label}, group.metals index {jdx=})")
    print(f"conn_idx before set: {conn_idx=}")
    conn_idx = sorted(list(set(conn_idx)))
    split_groups = []
    final_ligand_indices_by_metal = {jdx : [] for jdx, met in enumerate(group.metals)}
    for jdx, indices in conn_idx_by_metal.items():
        metal = group.metals[jdx]
        if indices:
            print(f"metal {metal.label} ({metal.atom_site_label}) connected to {[group.atoms[i].atom_site_label for i in indices]}")
            if single_ring:
                # For single ring, we need to check distances
                print(f"Checking distances for a single ring")
                distances = [get_dist(metal.coord, group.atoms[i].coord) for i in indices]
                mean = round(np.mean(distances), 3)
                std_dev = round(float(np.std(distances)), 3)
                z_scores = (distances - mean) / std_dev

                print(f"Distances: {distances}, Mean: {mean}, Std Dev: {std_dev}, Z-scores: {z_scores}")
                if std_dev > 0.1 :
                    new_group = []
                    for idx, z in zip(indices, z_scores):
                        if abs(z) < 1.0:  # Using a threshold of 1.0 for Z-score
                            print(f"Distance {distances[idx]} has a z-score {z_scores[idx]} below 1.0, adding to conn_idx")
                            new_group.append(idx)
                        else:
                            print(f"Distance {distances[idx]} has a z-score {z_scores[idx]} above 1.0, resetting mconnec for atom {group.atoms[idx].label}")
                            group.atoms[idx].reset_mconnec(metal, debug=1)
                    print(f"New group after distance check: {new_group}")
                else:
                    print(f"std_dev is too low ({std_dev}), adding all indices to conn_idx")
                    new_group = [i for i in indices]
            else :
                new_group = [i for i in indices]
            split_groups.append(new_group)
    
    for jdx, indices in enumerate(split_groups):
        for idx in indices:
            atom = group.atoms[idx]
            if atom.get_parent_index("ligand") is not None:
                final_ligand_indices_by_metal[jdx].append(atom.get_parent_index("ligand"))

    print(f"conn_idx: {conn_idx=}")
    print(f"split_groups: {split_groups=}")
    final_group_indices = split_groups
    print(f"final_group_indices: {final_group_indices=}")
    return group, final_group_indices, final_ligand_indices_by_metal

#######################################################
def coordination_correction_for_haptic_old (group: object, debug: int=0):
    add_factor = 0.45
    if debug > 0: print("Entering COORD_CORR_HAPTIC:")

    distances = []
    for idx, atom in enumerate(group.atoms):
        metal = atom.get_closest_metal()
        dist = get_dist(atom.coord, metal.coord)
        thres = (metal.radii + atom.radii) + add_factor
        #ratio_list.append(round(dist/thres,3))
        distances.append(round(dist, 3))
        if debug >= 2 : 
            print(f"\tAtom {idx} :", atom.label, f"\tMetal :", metal.label, "\tdistance :", round(dist, 3), "\tthres :", thres)
    mean = np.mean(distances)
    std_dev = round(float(np.std(distances)), 3)
    if debug >= 2 : print(f"\t{distances=} {mean=} {std_dev=}")

    is_ring = is_single_ring(group.labels, group.coord)
    conn_idx = []
    final_ligand_indices = []
    for idx, (atom, dist) in enumerate(zip(group.atoms, distances)) :
        if atom.label == "H" : 
            if debug >=1 : print(f"\t!!! Wrong metal-coordination assignment for Atom", idx, atom.label , get_dist(atom.coord, metal.coord), "due to H")
            if debug >=1 : print(atom.label)
            atom.reset_mconnec(metal, debug=1)  
        elif is_ring:
            if std_dev > 0.1 :
                if dist < mean - std_dev:
                    conn_idx.append(idx)
                    final_ligand_indices.append(atom.get_parent_index("ligand"))
                else:
                    atom.reset_mconnec(metal, debug=1) 
            else:
                conn_idx.append(idx)
                final_ligand_indices.append(atom.get_parent_index("ligand")) 
        else:
            conn_idx.append(idx)
            final_ligand_indices.append(atom.get_parent_index("ligand"))

    conn_idx = sorted(list(set(conn_idx)))
    conn_idx = [conn_idx]
    print(f"conn_idx: {conn_idx=}")
    return group, conn_idx, final_ligand_indices
    # final_group_indices = extract_final_indices(range(len(group.atoms)), conn_idx)
    # print(f"final_group_indices: {final_group_indices=}")
    # return group, final_group_indices, final_ligand_indices

#######################################################