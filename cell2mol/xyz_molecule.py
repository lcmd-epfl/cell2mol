import os
import sys
from contextlib import redirect_stdout
from ase.io import read
from cell2mol.classes import Molecule
from cell2mol.connectivity import labels2formula, split_species
from cell2mol.read_write import print_molecule, writexyz

# Constants
VERSION = "2.0"
COV_FACTOR = 1.0
METAL_FACTOR = 1.0


def get_molecule(input_path, name, input_charge, current_dir, debug=2):
    molec_fname = os.path.join(current_dir, f"Molecule_{name}.mol")
    output_fname = os.path.join(current_dir, f"mol_{name}.out")

    with open(output_fname, "w") as output:
        with redirect_stdout(output):
            print(f"cell2mol version {VERSION}")
            print(f"INITIATING molceule object from input path: {input_path}")
            print(f"Debug level: {debug}")

            structure = read(input_path)
            labels = structure.get_chemical_symbols()
            coords = structure.get_positions()

            blocklist = split_species(labels, coords, cov_factor=COV_FACTOR)
            print(f"The number of molecules in the xyz : blocklist={len(blocklist)}")
            if len(blocklist) > 1:
                print("Input file includes more than one molecules. Stopping")
                for i, block in enumerate(blocklist):
                    block_labels = [labels[i] for i in block]
                    block_coords = [coords[i] for i in block]
                    print(
                        f"Found block {i}",
                        labels2formula(block_labels),
                        len(block_labels),
                        "atoms",
                    )
                    writexyz(
                        os.getcwd(), f"Block_{name}_{i}.xyz", block_labels, block_coords
                    )
                return
            if len(blocklist) == 0:
                print("No molecule found from the input file")
                return

            newmolec = Molecule.from_positional(labels, coords)

            newmolec.set_adjacency_parameters(
                cov_factor=COV_FACTOR, metal_factor=METAL_FACTOR
            )
            newmolec.set_atoms(create_adjacencies=True, debug=debug)

            if newmolec.iscomplex:
                newmolec.split_complex()
            elif newmolec.has_IA_IIA:
                newmolec.split_IA_IIA()
            elif newmolec.has_post_transition_metal:
                print(f"GETREFS: {newmolec.formula} has post-transition metal")
                newmolec.split_post_transition_metal()
            else:
                newmolec.add_parent(newmolec, indices=[*range(0, newmolec.natoms, 1)])

            if newmolec.iscomplex:
                if debug >= 0:
                    print(
                        f"GET_MOLECULE: working with {newmolec.formula} with transition metals"
                    )
                newmolec.get_hapticity(debug=debug)
                if len(newmolec.ligands) == 0:
                    print(f"GET_MOLECULE: {newmolec.formula} is a metal cluster")
                else:
                    for lig in newmolec.ligands:
                        lig.get_denticity(debug=debug)
                for met in newmolec.metals:
                    met.get_connected_metals(debug=debug)
                    met.get_coordination_geometry(debug=debug)
                    met.get_coord_sphere_formula(debug=debug)
            elif newmolec.has_IA_IIA:
                if debug >= 0:
                    print(
                        f"GET_MOLECULE: working with {newmolec.formula} with alkali or alkali earth metals"
                    )
                if len(newmolec.ligands) == 0:
                    pass
                else:
                    for lig in newmolec.ligands:
                        lig.get_denticity(debug=debug)
                for met in newmolec.metals:
                    met.get_connected_metals(debug=debug)
                    met.get_coordination_geometry(debug=debug)
                    met.get_coord_sphere_formula(debug=debug)
            elif newmolec.has_post_transition_metal:
                if debug >= 0:
                    print(
                        f"GET_MOLECULE: working with {newmolec.formula} with post-transition metals"
                    )
                if debug >= 0:
                    print(f"GET_MOLECULE: {[met.label for met in newmolec.metals]}")
                if debug >= 0:
                    print(f"GET_MOLECULE: {[lig.formula for lig in newmolec.ligands]}")
                if len(newmolec.ligands) == 0:
                    pass
                else:
                    for lig in newmolec.ligands:
                        lig.get_denticity(debug=debug)
                for met in newmolec.metals:
                    met.get_connected_metals(debug=debug)
                    met.get_coordination_geometry(debug=debug)
                    met.get_coord_sphere_formula(debug=debug)

            newmolec.input_charge = input_charge
            if input_charge is not None:
                newmolec.get_unique_species(debug=debug)
                newmolec.get_selected_cs(debug=debug)
                newmolec.balance_charges_for_molecules(
                    input_charge=input_charge, debug=debug
                )
                if any(
                    [
                        newmolec.error_get_poscharges,
                        newmolec.error_multiple_distrib,
                        newmolec.error_empty_distrib,
                    ]
                ):
                    print("[ERROR] Charge assignment failed.")
                    newmolec.save(molec_fname)
                    return newmolec

                newmolec.assign_charges_for_molecule(debug=debug)
                newmolec.create_bonds(debug=debug)
                if newmolec.error_create_bonds:
                    print("[ERROR] Create Bonds failed.")
                    newmolec.error_case = 8
                else:
                    print_molecule(newmolec)
            newmolec.save(molec_fname)
            return newmolec


# Run the main function
if __name__ == "__main__":
    input = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].strip() != "":
        input_charge = int(sys.argv[2])
    else:
        input_charge = None
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)

    # Example usage, replace with actual arguments
    get_molecule(input_path, name, input_charge, current_dir, debug=1)
