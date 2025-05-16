import os
import sys
import logging
from contextlib import redirect_stdout
from ase.io import read
from cell2mol.classes import molecule


# Constants
VERSION = "2.0"
COV_FACTOR = 1.3
METAL_FACTOR = 1.0

def get_molecule (input_path, name, current_dir, debug=2):

    molec_fname = os.path.join(current_dir, f"Molecule_{name}.mol")
    output_fname = os.path.join(current_dir, "cell2mol.out")

    with open(output_fname, "w") as output:
        with redirect_stdout(output):
            print(f"cell2mol version {VERSION}")
            print(f"INITIATING cell object from input path: {input_path}")
            print(f"Debug level: {debug}")

            structure = read(input_path)
            labels = structure.get_chemical_symbols()
            coords = structure.get_positions()
            newmolec = molecule.from_positional(labels, coords)
            newmolec.set_adjacency_parameters(cov_factor=COV_FACTOR, metal_factor=METAL_FACTOR)
            newmolec.set_atoms(create_adjacencies=True, debug=debug)
            if newmolec.iscomplex: 
                newmolec.split_complex()
                newmolec.get_hapticity(debug=debug)
                for lig in newmolec.ligands:
                    lig.get_denticity(debug=debug)
                for met in newmolec.metals:                         
                    met.get_coordination_geometry(debug=debug)
                    met.get_coord_sphere_formula()
            
            newmolec.save(molec_fname)
    return newmolec

# Run the main function
if __name__ == "__main__":

    input = sys.argv[1]
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)

    # Example usage, replace with actual arguments
    get_molecule(input_path, name, current_dir, debug=1)