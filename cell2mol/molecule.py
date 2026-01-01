#!/usr/bin/env python

import os
import argparse
import logging
from ase.io import read

from cell2mol.classes import Molecule
from cell2mol.element_utils import labels2formula
from cell2mol.connectivity import split_species
from cell2mol.write_results import print_molecule, setup_logger

logger = logging.getLogger(__name__)

# Constants
VERSION = "2.0"
COV_FACTOR = 1.0
METAL_FACTOR = 1.0


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def process_molecule(input_path, name, input_charge, current_dir):
    molec_fname = os.path.join(current_dir, f"Molecule_{name}.mol")

    logger.info(f"cell2mol version {VERSION}")
    logger.info(f"Initiating molecule object from input path: {input_path}")

    structure = read(input_path)
    labels = structure.get_chemical_symbols()
    coords = structure.get_positions()

    blocklist = split_species(labels, coords, cov_factor=COV_FACTOR)
    logger.info(f"Number of molecules in xyz: {len(blocklist)}")

    if len(blocklist) > 1:
        logger.error("Input file includes more than one molecule. Stopping.")
        for i, block in enumerate(blocklist):
            block_labels = [labels[j] for j in block]
            # block_coords = [coords[j] for j in block]
            logger.info(
                f"Found block {i}: {labels2formula(block_labels)} "
                f"({len(block_labels)} atoms)"
            )
            # if needed, uncomment to generate separate xyz files for each block
            # from cell2mol.write_results import writexyz
            # block_coords = [coords[j] for j in block]
            # writexyz(
            #     os.getcwd(),
            #     f"Block_{name}_{i}.xyz",
            #     block_labels,
            #     block_coords,
            # )
        return None

    if len(blocklist) == 0:
        logger.error("No molecule found in the input file.")
        return None

    newmolec = Molecule.from_positional(labels, coords)

    newmolec.set_adjacency_parameters(
        cov_factor=COV_FACTOR,
        metal_factor=METAL_FACTOR,
    )
    newmolec.set_atoms(create_adjacencies=True)

    if newmolec.iscomplex:
        newmolec.split_complex()
    elif newmolec.has_IA_IIA:
        newmolec.split_IA_IIA()
    elif newmolec.has_post_transition_metal:
        logger.info(f"{newmolec.formula} has post-transition metal")
        newmolec.split_post_transition_metal()
    else:
        newmolec.add_parent(
            newmolec,
            indices=list(range(newmolec.natoms)),
        )

    if newmolec.iscomplex:
        logger.info(f"Working with {newmolec.formula} (transition metal complex)")
        newmolec.get_hapticity()

        if len(newmolec.ligands) == 0:
            logger.info("Metal cluster detected")
        else:
            for lig in newmolec.ligands:
                lig.get_denticity()

        for met in newmolec.metals:
            met.get_connected_metals()
            met.get_coordination_geometry()
            met.get_coord_sphere_formula()

    elif newmolec.has_IA_IIA or newmolec.has_post_transition_metal:
        logger.info(f"Working with {newmolec.formula}")

        for lig in newmolec.ligands:
            lig.get_denticity()

        for met in newmolec.metals:
            met.get_connected_metals()
            met.get_coordination_geometry()
            met.get_coord_sphere_formula()

    # Charge assignment
    newmolec.input_charge = input_charge

    if input_charge is not None:
        logger.info(f"Assigning total charge: {input_charge}")

        newmolec.get_unique_species()
        newmolec.get_selected_cs()
        newmolec.balance_charges_for_molecules(input_charge=input_charge)

        if any(
            [
                newmolec.error_get_poscharges,
                newmolec.error_multiple_distrib,
                newmolec.error_empty_distrib,
            ]
        ):
            logger.error("Charge assignment failed.")
            newmolec.save(molec_fname)
            return newmolec

        newmolec.assign_charges_for_molecule()
        newmolec.create_bonds()

        if newmolec.error_create_bonds:
            logger.error("Bond creation failed.")
            newmolec.error_case = 8
        else:
            print_molecule(newmolec)

    newmolec.save(molec_fname)
    logger.info(f"Molecule saved to {molec_fname}")

    return newmolec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Cell2Mol molecule object from XYZ file"
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="filepath",
        type=str,
        required=True,
        help="Path to the input file (.xyz)",
    )

    parser.add_argument(
        "--charge",
        dest="charge",
        type=int,
        help="Total charge of a molecule in .xyz file",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_file)

    input_path = os.path.normpath(args.input)
    current_dir = os.getcwd()
    _, file = os.path.split(input_path)
    name, _ = os.path.splitext(file)

    process_molecule(
        input_path=input_path,
        name=name,
        input_charge=args.charge,
        current_dir=current_dir,
    )


if __name__ == "__main__":
    main()
