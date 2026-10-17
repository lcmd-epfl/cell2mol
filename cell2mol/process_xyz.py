#!/usr/bin/env python

import os
import logging
import argparse
import sys
import gc
from cell2mol.utils.processing import (
    ProcessingError,
    read_structure,
    raise_processing_error,
    save_partial_results,
    run_cli,
)
from cell2mol.utils.limits import set_time_limit
from typing import cast
from ase import Atoms
from ase.io import read
from cell2mol.classes import Molecule, MoleculeSet
from cell2mol.element_utils import symbols2formula
from cell2mol.connectivity import split_species
from cell2mol.operations import extract_from_list
from cell2mol.write_results import (
    get_molecule_error_messages,
    write_molecule_details,
    write_unique_species,
    write_plausible_charges,
)
from cell2mol.charge.balancing import balance_molecule_charge
from cell2mol.utils import config

logger = logging.getLogger(__name__)

# Constants
COV_FACTOR = config.COV_FACTOR
METAL_FACTOR = config.METAL_FACTOR


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def build_molecule(labels, coords):
    """Build one fully prepared Molecule from a block of the xyz file."""
    newmolec = Molecule.from_positional(labels, coords)
    newmolec.set_adjacency_parameters(cov_factor=COV_FACTOR, metal_factor=METAL_FACTOR)
    newmolec.set_atoms(create_adjacencies=True, use_bond_info=False)

    # --- split complexes ---
    if newmolec.is_metal_complex or newmolec.has_s_block_metal:
        logger.debug("Splitting Metal complex: %s", newmolec.formula)
        newmolec.split_complex()
    elif newmolec.is_post_transition_metal_complex:
        logger.debug("Splitting post-transition metal complex: %s", newmolec.formula)
        newmolec.split_complex(post_tms=True)
    else:
        newmolec.add_parent(newmolec, indices=list(range(newmolec.natoms)))

    newmolec.analyze_coordination()
    newmolec.detect_special_moieties()

    return newmolec


def _write_summary(molset, name, path):
    """Write the human-readable summary next to the saved object."""
    with open(path, "w") as f:
        print(name, file=f)
        print(
            f"{len(molset.moleclist)} molecules, totcharge={molset.totcharge}",
            file=f,
        )
        for idx, mol in enumerate(molset.moleclist):
            write_molecule_details(mol, file=f, index=idx)
        write_unique_species(molset, file=f)
        write_plausible_charges(molset, file=f)
        print(
            get_molecule_error_messages(molset.error_cases_all, molset.error_case),
            file=f,
        )


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def interpret_molecule(input_path, name, input_charge, current_dir):
    """Interpret the molecular content of an xyz file.

    Returns a ``MoleculeSet`` holding one ``Molecule`` per connected block of
    the file, a single one included. Input or runtime failures raise
    ProcessingError with any available partial result.
    """
    summary_molecule_fname = os.path.join(current_dir, "molecule_summary.txt")

    logger.info("cell2mol version %s", config.VERSION)
    logger.info("Input XYZ: %s", input_path)

    # An xyz file carries no CIF bond information, so connectivity can only come
    # from interatomic distances.
    previous_bond_info = config.USE_BOND_INFO
    config.USE_BOND_INFO = False

    logger.info("Input total charge: %s", input_charge)
    molset = None
    save_fname = None

    try:
        with set_time_limit(config.TIMEOUT):
            structure = cast(Atoms, read_structure(read, input_path, "xyz"))
            labels = structure.get_chemical_symbols()
            coords = structure.get_positions()

            blocklist = cast(
                "list[list[int]]",
                split_species(
                    labels,
                    coords,
                    cov_factor=COV_FACTOR,
                    metal_factor=METAL_FACTOR,
                    use_bond_info=False,
                ),
            )
            logger.info("Number of molecules in xyz: %d", len(blocklist))

            # --- sanity checks ---
            if not blocklist:
                logger.error("No molecule found in the input file.")
                raise ProcessingError(
                    "No molecule found in the input file", config.ERR_INPUT
                )

            # --- build molecules ---
            moleclist = []
            for idx, block in enumerate(blocklist):
                block_labels = extract_from_list(block, labels, dimension=1)
                block_coords = extract_from_list(block, coords.tolist(), dimension=1)
                logger.info(
                    "Block %d: %s (%d atoms)",
                    idx,
                    symbols2formula(block_labels),
                    len(block_labels),
                )
                moleclist.append(build_molecule(block_labels, block_coords))

            # Even a lone molecule goes into a set: the set owns the unique species,
            # the charge target and the error codes.
            molset = MoleculeSet(name=name, moleclist=moleclist)
            save_fname = os.path.join(current_dir, f"MoleculeSet_{name}")

            molset.collect_unique_species()
            molset.check_hydrogens()
            molset.get_plausible_charges()
            error_case = molset.assess_errors()

            # --- charge assignment ---
            molset.input_charge = input_charge
            if input_charge is None:
                logger.info("No input charge provided.")
                return molset

            logger.info("Assigning total charge: %d", input_charge)
            if error_case != 0:
                logger.error(
                    "Fails checking hydrogens and plausible charges: %s",
                    get_molecule_error_messages(molset.error_cases_all, error_case),
                )
                return molset

            molset = balance_molecule_charge(molset, input_charge=input_charge)
            error_case = molset.assess_errors()
            if error_case != 0:
                logger.error(
                    "Fails balancing charges: %s",
                    get_molecule_error_messages(molset.error_cases_all, error_case),
                )
                return molset

            molset.assign_charges()
            error_case = molset.assess_errors()
            if error_case != 0:
                logger.error(
                    "Fails assigning charges: %s",
                    get_molecule_error_messages(molset.error_cases_all, error_case),
                )
                return molset

            molset.assign_spin()
            molset.assess_errors()

            return molset

    except Exception as exc:
        if isinstance(exc, MemoryError):
            structure = None
            gc.collect()
        logger.error("Molecule processing failed: %s", exc)
        raise_processing_error(exc, molset)

    finally:
        config.USE_BOND_INFO = previous_bond_info

        def save_outputs():
            if molset is not None and save_fname is not None:
                molset.save(save_fname + ".mol", format="pickle")
                molset.save(save_fname + ".json", format="json")
                _write_summary(molset, name, summary_molecule_fname)

        save_partial_results(save_outputs, molset)


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
        help="Total charge of all molecules in the .xyz file",
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
        sys.exit(config.ERR_INPUT)

    sys.exit(
        run_cli(
            interpret_molecule,
            input_path=input_path,
            name=name,
            input_charge=args.charge,
            current_dir=os.getcwd(),
        )
    )


if __name__ == "__main__":
    _main()
