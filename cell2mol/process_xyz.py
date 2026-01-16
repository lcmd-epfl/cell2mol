#!/usr/bin/env python

import os
import logging
import argparse
from ase.io import read
from cell2mol.classes import Molecule
from cell2mol.element_utils import labels2formula
from cell2mol.connectivity import split_species
from cell2mol.write_results import get_molecule_error_message, write_molecule_info
from cell2mol.charge.charge_balancer import balance_molecule_charge
from cell2mol.utils import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-7s %(name)-30s %(funcName)-30s | %(message)s",
    filename="cell2mol.out",
    filemode="w",
)

# Constants
COV_FACTOR = config.COV_FACTOR
METAL_FACTOR = config.METAL_FACTOR


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def interpret_molecule(input_path, name, input_charge, current_dir):
    molec_fname = os.path.join(current_dir, f"Molecule_{name}.mol")

    logger.info("cell2mol version %s", config.VERSION)
    logger.info("Input XYZ: %s", input_path)

    structure = read(input_path)
    labels = structure.get_chemical_symbols()
    coords = structure.get_positions()

    blocklist = split_species(
        labels, coords, cov_factor=COV_FACTOR, metal_factor=METAL_FACTOR
    )
    logger.info("Number of molecules in xyz: %d", len(blocklist))

    # --- sanity checks ---
    if not blocklist:
        logger.error("No molecule found in the input file.")
        return None

    if len(blocklist) > 1:
        logger.error("Input file includes more than one molecule. Stopping.")
        if logger.isEnabledFor(logging.DEBUG):
            for i, block in enumerate(blocklist):
                block_labels = [labels[j] for j in block]
                logger.debug(
                    "Found block %d: %s (%d atoms)",
                    i,
                    labels2formula(block_labels),
                    len(block_labels),
                )
        return None

    # --- build molecule ---
    newmolec = Molecule.from_positional(labels, coords)
    newmolec.set_adjacency_parameters(cov_factor=COV_FACTOR, metal_factor=METAL_FACTOR)
    newmolec.set_atoms(create_adjacencies=True)

    # --- split complexes ---
    if newmolec.iscomplex or newmolec.has_IA_IIA:
        logger.debug("Splitting complex: %s", newmolec.formula)
        newmolec.split_complex()
    elif newmolec.has_post_transition_metal:
        logger.debug("Splitting post-transition metal complex: %s", newmolec.formula)
        newmolec.split_complex(post_tms=True)
    else:
        newmolec.add_parent(newmolec, indices=list(range(newmolec.natoms)))

    # --- classification logging ---
    if newmolec.iscomplex:
        logger.info("Has transition metals: %s", newmolec.formula)
        newmolec.get_hapticity()
        if not newmolec.ligands:
            logger.debug("Metal cluster detected")

    elif newmolec.has_IA_IIA:
        logger.info("Has alkali or alkaline earth metals: %s", newmolec.formula)

    elif newmolec.has_post_transition_metal:
        logger.info("Has post-transition metals: %s", newmolec.formula)
        logger.debug("metals=%s", [m.label for m in newmolec.metals])
        logger.debug("ligands=%s", [l.formula for l in newmolec.ligands])

    else:
        logger.info(
            "Non-complex molecule: %s (non-complex=%s)",
            newmolec.formula,
            newmolec.is_non_complex_molecule,
        )

    # --- common analysis ---
    for lig in newmolec.ligands:
        lig.get_denticity()

    for met in newmolec.metals:
        met.get_connected_metals()
        met.get_coordination_geometry()
        met.get_coord_sphere_formula()

    # --- charge assignment ---
    newmolec.input_charge = input_charge
    if input_charge is None:
        logger.info("No input charge provided.")
        return newmolec

    logger.info("Assigning total charge: %d", input_charge)
    newmolec.get_unique_species()
    newmolec.get_selected_cs()
    newmolec = balance_molecule_charge(newmolec, input_charge=input_charge)
    newmolec.assess_errors()

    if newmolec.error_case != 0:
        logger.error("Charge assignment failed.")
        newmolec.save(molec_fname)
        return newmolec

    newmolec.assign_charges()
    newmolec.create_bonds()
    newmolec.assess_errors()

    if newmolec.error_case != 0:
        logger.error("Bond creation failed.")
        newmolec.save(molec_fname)
        return newmolec

    newmolec.get_spin()
    newmolec.assess_errors()
    newmolec.save(molec_fname)
    logger.info("Molecule saved to %s", molec_fname)

    # Summary of molecule
    summary_molecule_fname = os.path.join(current_dir, "molecule_summary.out")
    with open(summary_molecule_fname, "w") as f:
        print(name, file=f)
        write_molecule_info(newmolec, file=f)
        print(get_molecule_error_message(newmolec.error_case), file=f)

    return newmolec


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cell2mol Molecule object from XYZ file"
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


def _main():
    args = parse_args()

    input_path = os.path.normpath(args.filepath)
    name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(args.filepath)[1].lower()
    if ext != ".xyz":
        raise ValueError("Invalid input file format. Only .xyz files are supported.")

    interpret_molecule(
        input_path=input_path,
        name=name,
        input_charge=args.charge,
        current_dir=os.getcwd(),
    )


if __name__ == "__main__":
    _main()
