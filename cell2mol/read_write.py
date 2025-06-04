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
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
}

#######################
def screening_cif(cif_file_path):
    
    radical = False
    disorder = False
    notfound_atom = False

    with open(cif_file_path, 'r') as ciffile:
        file_content = ciffile.read()
        if 'radical' in file_content:
            radical = True
        elif '_atom_site_fract_x' not in file_content:
            notfound_atom = True
        elif '?' in file_content:
            if "_diffrn_ambient_temperature ?" not in file_content and "_chemical_melting_point ?" not in file_content:
                disorder = True
            else:
                num_greps = file_content.count('?')
                if num_greps > 1:
                    disorder = True
    
    moiety_dicts = extract_moiety(cif_file_path)
    if len(moiety_dicts) == 0:
        polymeric = False
    else:
        polymeric = any("n(" in moiety['formula'] for moiety in moiety_dicts)

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
        #print(f"{radical=}, {disorder=}, {notfound_atom=}, {polymeric=}")    
        if radical:
            message += "\nRadical found in .cif file."        
        if disorder:
            message += "\nDisorder found in .cif file."
        if notfound_atom:
            message += "\nNo fractional coordinates found in .cif file."
        if polymeric:
            message += "\nPolymeric structure found in .cif file."
        return False, message
    else :
        print("Cif file is ready for processing")
        return True, message
        
#######################
def get_wyckoff_positions_old (file_path):
    # Open and read the CIF file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Flags and storage for parsing
    start_parsing = False
    data = []

    # Iterate through each line
    for line in lines:
        # Check if line contains the loop declaration for atomic sites
        if '_atom_site_fract_z' in line:
            start_parsing = True
            continue

        # Stop parsing if another loop starts or if data is done
        if 'loop_' in line and start_parsing:
            break

        # Parse the atomic position data
        if start_parsing:
            if line.strip():  # Check for non-empty line
                parts = line.split()
                if len(parts) >= 4:  # Ensure there are enough parts to parse
                    # Clean and extract the coordinates before parentheses
                    x = parts[2].split('(')[0]
                    y = parts[3].split('(')[0]
                    z = parts[4].split('(')[0]
                    # Append label, type, and cleaned fractional coordinates
                    data.append((parts[0], parts[1], float(x), float(y), float(z)))

    ref_labels = [entry[1] for entry in data]
    ref_fracs = [[entry[2], entry[3], entry[4]] for entry in data]

    # print(f"{len(ref_labels)=} {ref_labels}=")
    # print(f"{len(ref_fracs)=} {ref_fracs}=")

    return ref_labels, ref_fracs
#######################
def get_geom_bond (file_path):
    full_text = Path(file_path).read_text()
    loop_sections = re.split(r'\nloop_\n', full_text)

    geom_bond_data = []

    for section in loop_sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        # === _geom_bond block ===
        if lines[0].startswith('_geom_bond'):
            headers = []
            start_idx = -1

            for i, line in enumerate(lines):
                if line.strip().startswith('_geom_bond'):
                    headers.append(line.strip())
                else:
                    start_idx = i
                    break

            required = {'_geom_bond_atom_site_label_1',
                        '_geom_bond_atom_site_label_2',
                        '_geom_bond_distance'}
            if not required.issubset(headers):
                continue

            header_map = {h: idx for idx, h in enumerate(headers)}
            # print("Bond header map:", header_map)

            for line in lines[start_idx:]:
                if line.strip().startswith('_') or line.strip() == 'loop_':
                    break
                parts = line.split()
                if len(parts) >= len(headers):
                    try:
                        atom_1 = parts[header_map['_geom_bond_atom_site_label_1']]
                        atom_2 = parts[header_map['_geom_bond_atom_site_label_2']]
                        dist = float(parts[header_map['_geom_bond_distance']])
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
def get_wyckoff_positions (file_path):

    full_text = Path(file_path).read_text()
    loop_sections = re.split(r'\nloop_\n', full_text)

    atom_site_data = []

    for section in loop_sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        # === _atom_site block ===
        if lines[0].startswith('_atom_site'):
            headers = []
            start_idx = -1

            for i, line in enumerate(lines):
                if line.strip().startswith('_atom_site'):
                    headers.append(line.strip())
                else:
                    start_idx = i
                    break

            required = {'_atom_site_label', '_atom_site_type_symbol',
                        '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'}
            if not required.issubset(headers):
                continue

            header_map = {h: idx for idx, h in enumerate(headers)}
            #print("Atom site header map:", header_map)

            for line in lines[start_idx:]:
                if line.strip().startswith('_') or line.strip() == 'loop_':
                    break
                parts = line.split()
                if len(parts) >= len(headers):
                    try:
                        label = parts[header_map['_atom_site_label']]
                        symbol = parts[header_map['_atom_site_type_symbol']]
                        x = float(parts[header_map['_atom_site_fract_x']].split('(')[0])
                        y = float(parts[header_map['_atom_site_fract_y']].split('(')[0])
                        z = float(parts[header_map['_atom_site_fract_z']].split('(')[0])
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
# Helper to convert string like "H13-C5-N2" to Counter {'H':13, 'C':5, 'N':2}
def parse_formula_string(formula_str):
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    return Counter({el: int(cnt) if cnt else 1 for el, cnt in tokens})

def formula_diff_dict(f1, f2):
    c1 = parse_formula_string(f1)
    c2 = parse_formula_string(f2)
    all_elements = set(c1) | set(c2)
    diff = {el: abs(c1[el] - c2[el]) for el in all_elements if c1[el] != c2[el]}
    return diff

def find_closest_matches(reference, target):
    matches = {}
    for i, ref in enumerate(reference):
        if ref in target:
            matches[i] = {'ref': ref, 'match': ref, 'diff_dict': {}}
        else:
            diffs = [(tgt, formula_diff_dict(ref, tgt)) for tgt in target]
            # Select the one with the smallest total difference
            best_match, best_diff = min(diffs, key=lambda x: sum(x[1].values()))
            matches[i] = {'ref': ref, 'match': best_match, 'diff_dict': best_diff}
    return matches
#######################
def extract_chemical_name(file_path):
    try:
        with open(file_path, 'r') as file:
            start_reading = False
            chemical_name = ""

            for line in file:
                # Check for line starting with '_chemical_name_systematic'
                if line.startswith('_chemical_name_systematic'):
                    start_reading = True
                    continue  

                if start_reading:
                    chemical_name += line.strip()

                    if chemical_name.count(';') >= 2:
                        # Extract content between the first and second ';'
                        # Reformat without extra line breaks or extra spaces
                        chemical_name = ' '.join(chemical_name.split(';')[1].strip().split())
                        break

            return chemical_name

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

#######################
def extract_metal_oxidation_state(chemical_name):

    oxidation_states = []

    # Regex pattern to capture any format like "iron(iii)"
    pattern = r'(\b[a-zA-Z-]+\b)\((iii|ii|iv|v|vi|vii|i|0|o)\)'

    matches = re.findall(pattern, chemical_name, re.IGNORECASE)
    for metal, ox_state in matches:
        ox_state_map = {'0': 0, 'o': 0, 'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7}
        oxidation_state = ox_state_map.get(ox_state.lower(), ox_state)  # Retain text if needed
        oxidation_states.append((metal, oxidation_state))

    return oxidation_states

#######################
def parse_moiety(moiety: str) -> Tuple[str, float, int, str]:
    """ Parse a moiety string and return its formula, ratio (float), charge, and type """
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
    charge_match = re.search(r'(\d+[+-])$', formula_with_charge)
    if charge_match:
        charge_str = charge_match.group(1)
        charge = int(charge_str[:-1]) * (1 if charge_str[-1] == '+' else -1)
        formula = formula_with_charge.replace(charge_str, '').strip()
    else:
        charge = 0
        formula = formula_with_charge.strip()

    # Check if it contains any transition metal
    is_complex = any(re.search(rf'{metal}\d*', formula) for metal in transition_metals)
    compound_type = "complex" if is_complex else "molecule"

    return (formula, ratio, charge, compound_type)

################################
def cifformula_to_list(formula: str) -> list:
    # Match element symbols with optional numbers (e.g., 'C4', 'H2', 'N1')
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    result = []
    for element, count in tokens:
        n = int(count) if count else 1
        result.extend([element] * n)
    return result

#########################
def extract_moiety(file_path: str) -> list:
    """ Extracts the moiety information from a CIF file """
    uploaded_file_path = Path(file_path)
    with uploaded_file_path.open("r", encoding="utf-8") as file:
        cif_data_uploaded = file.read()
    if "_chemical_formula_moiety" in cif_data_uploaded:
        # Extract the moiety block
        moiety_match_uploaded = re.search(r"_chemical_formula_moiety\s*;\s*(.*?)\s*;", cif_data_uploaded, re.DOTALL)
        moiety_string_uploaded = moiety_match_uploaded.group(1) if moiety_match_uploaded else ""

        # # Split and parse moieties
        moieties_uploaded = moiety_string_uploaded.split(',')
        moieties_uploaded = [moiety.replace('\n', '') for moiety in moieties_uploaded]

        moiety_tuples = [parse_moiety(m.strip()) for m in moieties_uploaded]
        moiety_dicts = [ {'formula': f, 'ratio': s, 'charge': c, 'type': t} for f, s, c, t in moiety_tuples ]

        # print("moiety_dicts=",moiety_dicts)
    else:
        print("Error parsing moiety information from CIF file.")
        moiety_dicts = []
    return moiety_dicts

##########################
def classify_metals(formula: str) -> Dict[str, Dict[str, int]]:
    # Define categories
    transition_metals = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
    }

    alkali_metals = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'}
    alkaline_earth_metals = {'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'}

    post_transition_metals = {
        'Al', 'Ga', 'Ge', 'In', 'Sn', 'Tl', 'Pb', 'Bi'
    }

    lanthanides = {
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
        'Ho', 'Er', 'Tm', 'Yb', 'Lu'
    }

    actinides = {
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        'Es', 'Fm', 'Md', 'No', 'Lr'
    }

    # Initialize category results
    result = {
        'transition_metals': {},
        'alkali_metals': {},
        'alkaline_earth_metals': {},
        'post_transition_metals': {},
        'lanthanides': {},
        'actinides': {}
    }

    # Split and parse
    parts = formula.strip().split()
    for part in parts:
        match = re.match(r"([A-Z][a-z]*)(\d*)", part)
        if match:
            element = match.group(1)
            count = int(match.group(2)) if match.group(2) else 1

            if element in transition_metals:
                result['transition_metals'][element] = result['transition_metals'].get(element, 0) + count
            elif element in alkali_metals:
                result['alkali_metals'][element] = result['alkali_metals'].get(element, 0) + count
            elif element in alkaline_earth_metals:
                result['alkaline_earth_metals'][element] = result['alkaline_earth_metals'].get(element, 0) + count
            elif element in post_transition_metals:
                result['post_transition_metals'][element] = result['post_transition_metals'].get(element, 0) + count
            elif element in lanthanides:
                result['lanthanides'][element] = result['lanthanides'].get(element, 0) + count
            elif element in actinides:
                result['actinides'][element] = result['actinides'].get(element, 0) + count

    return result

##########################
def metal_info (moiety_dicts: list) -> dict:
    """
    Extracts metal information from moiety dictionaries.
    """
    metal_data = {}
    for moiety in moiety_dicts:
        formula = moiety['formula']
        ratio = moiety['ratio']
        charge = moiety['charge']
        compound_type = moiety['type']

        # Classify metals
        classified_metals = classify_metals(formula)
        metal_data[formula] = {
            'ratio': ratio,
            'charge': charge,
            'type': compound_type,
            'classified_metals': classified_metals
        }
    return metal_data

##########################
def flatten_metal_info(metal_info_dict, refcode):
    flat_list = []

    for i, (formula, info) in enumerate(metal_info_dict.items()):
        base = {
            'refcode': refcode,  # Placeholder for refcode
            'index': i,
            'formula': formula,
            'ratio': info['ratio'],
            'charge': info['charge'],
            'type': info['type']
        }

        found = False
        classified = info['classified_metals']
        for category, elements in classified.items():
            for element, count in elements.items():
                row = base.copy()
                row['metal_category'] = category
                row['metal'] = element
                row['count'] = count
                flat_list.append(row)
                found = True

        if not found:
            row = base.copy()
            row['metal_category'] = None
            row['metal'] = None
            row['count'] = 0
            flat_list.append(row)

    return flat_list

##########################
def compare_formula_xyz_vs_cif(xyzfile: str, formula_str_from_cif: str) -> dict:
    """
    Compare the elements from CIF with the elements in the XYZ file.
    """
    element_pattern = r'([A-Z][a-z]*)(\d*)'
    parsed_formula_cif = dict()
    for (element, count) in re.findall(element_pattern, formula_str_from_cif):
        parsed_formula_cif[element] = int(count) if count else 1
    mol = read(xyzfile)
    element_list = mol.get_chemical_symbols()
    element_list_count = dict(Counter(element_list))
    comparison_result = {
        'in_formula_cif': parsed_formula_cif,
        'in_list_xyz': element_list_count,
        'missing_in_xyz': {k: v-element_list_count.get(k, 0) for k, v in parsed_formula_cif.items() if element_list_count.get(k, 0) < v},
        'extra_in_xyz': {k: v-parsed_formula_cif.get(k, 0) for k, v in element_list_count.items() if parsed_formula_cif.get(k, 0) < v}
    }
    return comparison_result

#######################
def save_binary(variable, pathfile, backup: bool=False):
    try:
        file = open(pathfile,'wb')
        pickle.dump(variable,file)
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
def writexyz(fdir, fname, labels, pos, charge: int=0, spin: int=1):
    if fdir[-1] != "/":
        fdir = fdir + "/"
    natoms = len(labels)
    fullname = fdir + fname
    with open(fullname, "w") as fil:
        print(natoms, file=fil)
        print(charge, spin, file=fil)
        for label, (x, y, z) in zip(labels, pos):
            print(f"{label:<2}\t{x: .6f}\t{y: .6f}\t{z: .6f}", file=fil)
            # print("%s\t%.6f\t%.6f\t%.6f" % (l, pos[idx][0], pos[idx][1], pos[idx][2]),file=fil)

##############
def search_string_in_file(file_name, string_to_search):
    line_number = 0
    list_of_results = []
    with open(file_name, "r") as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                list_of_results.append((line_number))

    return list_of_results

##############
def readinfo(filepath):

    info = open(filepath, "r")
    lines = list(info.readlines())
    info.close()

    strings = [
        "Lattice parameters:",
        "Representative sites :",
        "OUTPUT CELL INFORMATION",
        "Bravais lattice vectors :",
        "All sites, (cartesian coordinates):",
        "Unit cell volume",
    ]
    lint = np.zeros((len(strings)))

    for l, line in enumerate(lines):
        for stx, string in enumerate(strings):
            if string in line:
                lint[stx] = l


    latparamsone = int(lint[0] + 2)
    latparamstwo = int(lint[1] - 1)
    fracstart = int(lint[1] + 2)
    fracend = int(lint[2] - 2)
    cellvecstart = int(lint[3] + 1)
    cellvecend = int(lint[4] - 1)
    coordstart = int(lint[4] + 2)
    coordend = int(lint[5] - 2)

    lfracs = []
    labels = []
    pos = []
    fracs = []
    cellvec = []
    cellparam = []

    for l, line in enumerate(lines):
        # reads lattice parameters
        if (l == latparamsone) or (l == latparamstwo):
            a, b, c = line.split()
            cellparam += [float(a), float(b), float(c)]

        # reads fractional coordinates
        if (l >= fracstart) and (l <= fracend):
            line_data = line.split()
            if len(line_data) == 4:
                label, x, y, z = line.split()
            elif len(line_data) == 5:  #sometimes, an occupation value is also given.
                label, x, y, z, occ = line.split()
            fracs.append([float(x), float(y), float(z)])
            lfracs.append(label)

        # reads cell vectors
        if (l >= cellvecstart) and (l <= cellvecend):
            v1, v2, v3 = line.split()
            cellvec.append([float(v1), float(v2), float(v3)])

        # reads cartesian coordinates
        if (l >= coordstart) and (l <= coordend):
            line_data = line.split()
            if len(line_data) == 4:
                label, x, y, z = line.split()
            elif len(line_data) == 5:  #sometimes, an occupation value is also given.
                label, x, y, z, occ = line.split()
            pos.append([float(x), float(y), float(z)])
            labels.append(label)


    return labels, pos, lfracs, fracs, cellvec, cellparam

##############
def readcif(filepath):

    info = open(filepath, "r")
    lines = list(info.readlines())
    info.close()

    strings = [
        "_journal_name_full",
        "_chemical_name_systematic",
        "_cell_volume",
        "_atom_type_radius_bond",
        "_atom_site_label",
        "_chemical_name_common",
    ]
    lint = np.zeros((len(strings)))

    for l, line in enumerate(lines):
        for stx, string in enumerate(strings):
            if string in line:
                lint[stx] = l

    common = int(lint[5])
    iscommon = False
    if common != 0:
        iscommon = True

    journalname_line = int(lint[0])

    radius_start = int(lint[3] + 1)
    radius_end = int(lint[4] - 2)

    if not iscommon:
        chemname_start = int(lint[1] + 2)
        chemname_end = int(lint[2] - 2)
    if iscommon:
        chemname_start = int(lint[1] + 2)
        chemname_end = int(lint[5] - 2)

    labels = []
    radii = []
    chemname = ""
    for l, line in enumerate(lines):
        # reads journal name
        if l == journalname_line:
            journal = line.split("'")[1]
        # reads fractional coordinates
        if (l >= chemname_start) and (l <= chemname_end):
            chemname += line.split("\n")[0]
        # reads cell vectors
        if (l >= radius_start) and (l <= radius_end):
            label, radius = line.split()
            labels.append(label)
            radii.append(radius)

    return journal, chemname, labels, radii

###########
def print_molecule(mol, name, ext, folder):
    filename = str(folder) + "/" + str(name) + "." + str(ext)

    if ext == "xyz" or ext == "txt":
        with open(filename, "w") as fil:

            # XYZ
            if ext == "xyz":
                print(mol.natoms, file=fil)
                if mol.totcharge is not None and mol.spin is not None:
                    print(mol.totcharge, mol.spin, file=fil)
                if mol.totcharge is not None and mol.spin is None:
                    print(mol.totcharge, "SPIN", file=fil)
                # if not hasattr(mol, 'totcharge') and not hasattr(mol, 'spin'):
                else:
                    print("", file=fil)

                for a in mol.atoms:
                    print("%s   %.6f   %.6f   %.6f" % (a.label, a.coord[0], a.coord[1], a.coord[2]),file=fil)

            # TXT
            elif ext == "txt":
                print(vars(mol), file=fil)

    elif ext == "gmol" or ext == "mol" or ext == "npy" or ext == "dict":
        with open(filename, "wb") as fil:

            # GMOL
            if ext == "gmol":
                pickle.dump(mol, fil)

            # MOL
            elif ext == "mol":
                pickle.dump(mol.object, fil)

            # NPY
            elif ext == "npy":
                np.save(filename, mol)

            # DICT
            elif ext == "dict":
                mydict = vars(mol)
                pickle.dump(mydict, fil)

    else:
        print(ext, "not found as a valid print extension in print_molecule")

#############
def savemolecules(moleclist, output_dir, print_types, option_print_repeated=True):

    # DEFAULTS
    print_xyz = True
    print_gmol = True
    print_npy = False
    print_mol = False
    print_txt = False
    print_dict = False

    if "xyz" not in print_types:
        print_xyz = False
    if "gmol" not in print_types:
        print_gmol = False
    if "npy" in print_types:
        print_npy = True
    if "mol" in print_types:
        print_mol = True
    if "txt" in print_types:
        print_txt = True
    if "dict" in print_types:
        print_dict = True

    printedmolecs = []

    for mol in moleclist:
        shalliprint = False

        if any((mol.elemcountvec == pmol.elemcountvec).all() for pmol in printedmolecs):
            shalliprint = False
        else:
            shalliprint = True
        if option_print_repeated:  # Overwrites decision if the user decides so
            shalliprint = True

        if shalliprint:
            printedmolecs.append(mol)

            if print_xyz:
                print_molecule(mol, mol.name, "xyz", output_dir)
            if print_gmol:
                print_molecule(mol, mol.name, "gmol", output_dir)
            if print_npy:
                print_molecule(mol, mol.name, "npy", output_dir)
            if print_txt:
                print_molecule(mol, mol.name, "txt", output_dir)
            if print_dict:
                print_molecule(mol, mol.name, "dict", output_dir)
            if mol.object is not None:
                if print_mol:
                    print_molecule(mol, mol.name, "mol", output_dir)

###########
def print_unit_cell(cell, output_dir):

    # Print Original cell
    print("Original_Cell.xyz")
    writexyz(output_dir, "Original_Cell.xyz", cell.labels, cell.pos)

    # Print Full Unit Cell
    cellatoms = 0
    for mol in cell.moleclist:
        cellatoms += mol.natoms

    print("{}_Full_Cell.xyz".format(cell.refcode))
    cell_fname = output_dir + "/" + cell.refcode + "_Full_Cell.xyz"

    with open(cell_fname, "w") as fil:
        print(cellatoms, file=fil)
        print(" ", file=fil)
        for mol in cell.moleclist:
            for a in mol.atoms:
                print(a.label, a.coord[0], a.coord[1], a.coord[2], file=fil)

############
def extract_chemical_formula_moiety(file_path):
    with open(file_path, 'r') as file:
        cif_content = file.read()
    
    # Find the chemical formula moiety using regex
    pattern = r"_chemical_formula_moiety\s+['\";]([^;'\"]+)['\";]"
    match = re.search(pattern, cif_content)
    if not match:
        raise ValueError("Chemical formula moiety not found.")
    formula = match.group(1).strip()
    return formula

def parse_formula_with_quantity(formula: str):
    element_pattern = r"(\d*)\(?([A-Za-z0-9\s]+)\)?(\d*)(\d+[+-]?)"
    moieties_info = []

    for match in re.findall(element_pattern, formula):
        quantity = int(match[0]) if match[0] else 1
        elements = defaultdict(int)
        for element, count in re.findall(r"([A-Z][a-z]*)(\d*)", match[1]):
            elements[element] += int(count) if count else 1

        charge_str = match[3]
        charge = int(charge_str[:-1]) * (1 if charge_str.endswith("+") else -1)

        moieties_info.append({
            "elements": dict(elements),
            "quantity": quantity,
            "charge": charge
        })
    return moieties_info

######################################################
def print_refmoleclist (cell):
    
    for i, ref in enumerate(cell.refmoleclist):
        if ref.totcharge is not None:
            if ref.iscomplex:
                print(f"Reference Molecule {i}: {ref.formula} {ref.totcharge=} (TM complex)")
            elif ref.has_IA_IIA:
                print(f"Reference Molecule {i}: {ref.formula} {ref.totcharge=} (Complex with Alkali or Alkaline metals)")
            elif ref.has_post_transition_metal:
                print(f"Reference Molecule {i}: {ref.formula} {ref.totcharge=} (Complex with Post-Transition metals)")
            else:
                if ref.smiles is not None:
                    print(f"Reference Molecule {i} : {ref.formula} {ref.totcharge=} (Non-complex) {ref.smiles=}")
                else:
                    print(f"Reference Molecule {i} : {ref.formula} {ref.totcharge=} (Non-complex)")
        elif ref.totcharge_cif is not None:
            if ref.iscomplex:
                print(f"Reference Molecule {i}: {ref.formula} {ref.totcharge_cif=} (TM complex)")
            elif ref.has_IA_IIA:
                print(f"Reference Molecule {i}: {ref.formula} {ref.totcharge_cif=} (Complex with Alkali or Alkaline metals)")
            elif ref.has_post_transition_metal:
                print(f"Reference Molecule {i}: {ref.formula} {ref.totcharge_cif=} (Complex with Post-Transition metals)")
            else:
                print(f"Reference Molecule {i} : {ref.formula} {ref.totcharge_cif=} (Non-complex)")        
        else:
            if ref.iscomplex:
                print(f"Reference Molecule {i}: {ref.formula} (TM complex)")
            elif ref.has_IA_IIA:
                print(f"Reference Molecule {i}: {ref.formula} (Complex with Alkali or Alkaline metals)")
            elif ref.has_post_transition_metal:
                print(f"Reference Molecule {i}: {ref.formula} (Complex with Post-Transition metals)")
            else:
                print(f"Reference Molecule {i} : {ref.formula} (Non-complex)")

        if ref.iscomplex or ref.has_IA_IIA or ref.has_post_transition_metal:
            for met in ref.metals:
                met_info = f"\t{met.formula} ({met.subtype})"
                if met.charge is not None:
                    met_info += f" metal_OS={met.charge}"
                elif met.possible_cs is not None:
                    met_info += f" metal_possible_OS={met.possible_cs}"
                print(met_info)

                if met.coord_sphere_formula is not None:
                    print(f"\t|--Coordination information coord_sphere_formula={met.coord_sphere_formula}")

                if all(hasattr(met, attr) for attr in ["coord_nr", "coord_geometry", "geom_deviation"]):
                    print(f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}")

                if all(hasattr(met, attr) for attr in ["coord_nr_with_metal_bonds", "coord_geometry_with_metal_bonds", "geom_deviation_with_metal_bonds", "metals"]):
                    bonded_metals = [m.label for m in met.metals]
                    print(f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}")

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
                            group_info += f" connected_metals={[m.label for m in group.metals]}"
                        # if group.closest_metal is not None:
                        #     group_info += f" closest_metal.label={group.closest_metal.label}"
                        print(group_info)
                    
######################################################
def print_unique_species (cell):
    if cell.unique_species is not None:
        print(f"\nUnique Species in {cell.subtype}:")
        for specie in cell.unique_species:
            if specie.subtype == "metal":
                if specie.charge is not None:
                    print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} {specie.charge=}")
                else:
                    print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=}")
            else:
                if specie.totcharge is not None and specie.smiles is not None:
                    print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.smiles=} {specie.totcharge=}")
                elif specie.totcharge is not None:
                    print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.totcharge=}")
                else:
                    print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype})")
    else:
        print("\nNo unique species found in the cell object.")

######################################################
def print_possible_charges (cell, debug=0):
    """
    Print the possible charges for each species in the cell object.
    """
    if cell.species_list is not None:
        print(f"\nPossible charges of species in {cell.subtype}:")
        for specie in cell.species_list:
            if specie.possible_cs is not None:
                if specie.subtype == "metal":
                    print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} {specie.possible_cs=}") 
                else:
                    print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype})\n\t{specie.possible_cs=}")
                    if debug > 0 : print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype})\n\t{specie.possible_cs=}")
            else:
                if specie.subtype == "metal":
                    print(f"\t{specie.unique_index=} {specie.formula} ({specie.subtype}) {specie.coord_sphere_formula=} No possible cs")
                else:
                    print(f"\t{specie.unique_index=} {specie.formula}, {specie.subtype}  No possible cs") #[p.subtype for p in specie.parents])    
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
                    print(f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (TM complex)")
                elif mol.has_IA_IIA:
                    print(f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (Complex with Alkali or Alkaline metals)")
                elif mol.has_post_transition_metal:
                    print(f"Unitcell Molecule {i}: {mol.formula} {mol.totcharge=} (Complex with Post-Transition metals)")
                else:
                    if mol.smiles is not None:
                        print(f"Reference Molecule {i} : {mol.formula} {mol.totcharge=} (Non-complex) {mol.smiles=}")
                    else:
                        print(f"Reference Molecule {i} : {mol.formula} {mol.totcharge=} (Non-complex)")
            else:
                if mol.iscomplex:
                    print(f"Unitcell Molecule {i}: {mol.formula} (TM complex)")
                elif mol.has_IA_IIA:
                    print(f"Unitcell Molecule {i}: {mol.formula} (Complex with Alkali or Alkaline metals)")
                elif mol.has_post_transition_metal:
                    print(f"Unitcell Molecule {i}: {mol.formula} (Complex with Post-Transition metals)")
                else:
                    print(f"Unitcell Molecule {i} : {mol.formula} (Non-complex)")

            if mol.iscomplex or mol.has_IA_IIA or mol.has_post_transition_metal:
                for met in mol.metals:
                    met_info = f"\t{met.formula} ({met.subtype})"

                    if met.charge is not None:
                        met_info += f" metal_OS={met.charge}"
                    elif met.possible_cs is not None:
                        met_info += f" metal_possible_OS={met.possible_cs}"
                    print(met_info)

                    for attr in ["coord_sphere_formula"]:
                        if hasattr(met, attr):
                            print(f"\t|--Coordination information {attr}={getattr(met, attr)}")

                    if all(hasattr(met, attr) for attr in ["coord_nr", "coord_geometry", "geom_deviation"]):
                        print(f"\t|--coord_nr={met.coord_nr} coord_geometry={met.coord_geometry} geom_deviation={met.geom_deviation}")

                    if all(hasattr(met, attr) for attr in ["coord_nr_with_metal_bonds", "coord_geometry_with_metal_bonds", "geom_deviation_with_metal_bonds", "metals"]):
                        bonded_metals = [m.label for m in met.metals]
                        print(f"\t|--bonded metals={bonded_metals} coord_nr_with_metal_bonds={met.coord_nr_with_metal_bonds} coord_geometry_with_metal_bonds={met.coord_geometry_with_metal_bonds} geom_deviation_with_metal_bonds={met.geom_deviation_with_metal_bonds}")
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
                                group_info += f" connected_metals={[m.label for m in group.metals]}"
                            # if group.closest_metal is not None:
                            #     group_info += f" closest_metal.label={group.closest_metal.label}"
                            print(group_info)

                            # Optional: print group-metals connectivity
                            # for met in group.metals:
                            #     print(f"\t|--(group.metals){met.label} {met.mconnec=}")
    else:
        print("\nNo molecules found in the cell object.")

