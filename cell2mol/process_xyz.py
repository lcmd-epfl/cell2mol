#!/usr/bin/env python

import os
import logging
import argparse
from typing import cast
from ase import Atoms
from ase.io import read
from cell2mol.classes import Molecule, MoleculeSet
from cell2mol.element_utils import labels2formula
from cell2mol.connectivity import split_species
from cell2mol.operations import extract_from_list
from cell2mol.write_results import (
    get_molecule_error_message_all,
    write_molecule_info,
    write_unique_species,
    write_plausible_charges,
)
from cell2mol.charge.charge_balancer import balance_molecule_charge
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
    if newmolec.iscomplex or newmolec.has_ia_iia:
        logger.debug("Splitting complex: %s", newmolec.formula)
        newmolec.split_complex()
    elif newmolec.has_post_transition_metal:
        logger.debug("Splitting post-transition metal complex: %s", newmolec.formula)
        newmolec.split_complex(post_tms=True)
    else:
        newmolec.add_parent(newmolec, indices=list(range(newmolec.natoms)))

    newmolec.analyze_coordination()
    newmolec.detect_special_moieties()

    return newmolec


def _write_summary(target, name, path):
    """Write the human-readable summary next to the saved object."""
    with open(path, "w") as f:
        print(name, file=f)
        if isinstance(target, MoleculeSet):
            print(
                f"{len(target.moleclist)} molecules, totcharge={target.totcharge}",
                file=f,
            )
            for idx, mol in enumerate(target.moleclist):
                write_molecule_info(mol, file=f, index=idx)
        else:
            write_molecule_info(target, file=f)
        write_unique_species(target, file=f)
        write_plausible_charges(target, file=f)
        print(
            get_molecule_error_message_all(target.error_cases_all, target.error_case),
            file=f,
        )


# -----------------------------------------------------------------------------
# Core function
# -----------------------------------------------------------------------------
def interpret_molecule(input_path, name, input_charge, current_dir):
    """Interpret the molecular content of an xyz file.

    Returns a ``Molecule`` when the file holds a single molecule, and a
    ``MoleculeSet`` when it holds several molecules.
    """
    summary_molecule_fname = os.path.join(current_dir, "molecule_summary.txt")

    logger.info("cell2mol version %s", config.VERSION)
    logger.info("Input XYZ: %s", input_path)

    # An xyz file carries no CIF bond information, so connectivity can only come
    # from interatomic distances.
    config.USE_BOND_INFO = False

    logger.info("Input total charge: %s", input_charge)
    target = None
    save_fname = None

    try:
        try:
            # read(...) is typed Atoms | list[Atoms]; a single-frame xyz yields one Atoms.
            structure = cast(Atoms, read(input_path, format="xyz"))
        except (AssertionError, Exception) as e:
            logger.error(f"ASE failed to parse {input_path}: {e}")
            raise
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
            return None

        # --- build molecules ---
        moleclist = []
        for idx, block in enumerate(blocklist):
            block_labels = extract_from_list(block, labels, dimension=1)
            block_coords = extract_from_list(block, coords.tolist(), dimension=1)
            logger.info(
                "Block %d: %s (%d atoms)",
                idx,
                labels2formula(block_labels),
                len(block_labels),
            )
            moleclist.append(build_molecule(block_labels, block_coords))

        # A lone molecule stays a Molecule; only several of them need a set that
        # shares one pool of unique species and one charge target.
        if len(moleclist) == 1:
            target = moleclist[0]
            save_fname = os.path.join(current_dir, f"Molecule_{name}")
        else:
            target = MoleculeSet(name=name, moleclist=moleclist)
            save_fname = os.path.join(current_dir, f"MoleculeSet_{name}")

        # --- charge assignment ---
        target.input_charge = input_charge
        if input_charge is None:
            logger.info("No input charge provided.")
            return target

        logger.info("Assigning total charge: %d", input_charge)
        target.get_unique_species()

        # Hydrogens first, as in the reference pipeline
        target.check_hydrogens()
        target.get_plausible_charges()

        target.assess_errors()
        if target.error_case != 0:
            logger.error(
                "Fails checking hydrogens and plausible charges: %s",
                get_molecule_error_message_all(
                    target.error_cases_all, target.error_case
                ),
            )
            return target

        target = balance_molecule_charge(target, input_charge=input_charge)
        target.assess_errors()
        if target.error_case != 0:
            logger.error(
                "Fails balancing charges: %s",
                get_molecule_error_message_all(
                    target.error_cases_all, target.error_case
                ),
            )
            return target

        target.assign_charges()
        target.assess_errors()
        if target.error_case != 0:
            logger.error(
                "Fails assigning charges: %s",
                get_molecule_error_message_all(
                    target.error_cases_all, target.error_case
                ),
            )
            return target

        target.get_spin()
        target.assess_errors()

        return target

    except Exception as err:
        logger.error("interpret_molecule failed: %s", err)
        logger.debug("Exception details", exc_info=True)
        return target

    finally:
        # --- always save what was built ---
        if target is not None and save_fname is not None:
            target.save(save_fname + ".mol", format="pickle")
            logger.info("Saved to %s as pickle", save_fname + ".mol")
            target.save(save_fname + ".json", format="json")
            logger.info("Saved to %s as JSON", save_fname + ".json")
            _write_summary(target, name, summary_molecule_fname)


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
        raise ValueError("Invalid input file format. Only .xyz files are supported.")

    interpret_molecule(
        input_path=input_path,
        name=name,
        input_charge=args.charge,
        current_dir=os.getcwd(),
    )


if __name__ == "__main__":
    _main()
