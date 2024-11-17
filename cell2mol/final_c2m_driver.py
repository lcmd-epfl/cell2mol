import os
import sys
from cell2mol.helper import parsing_arguments
from cell2mol.refcell import process_refcell
from cell2mol.unitcell import process_unitcell
from cell2mol.xyz_molecule import get_molecule
from cell2mol.read_write import prefilter_cif, exit_with_error

def main():
    input, system_type, cell_para, debug_mode = parsing_arguments()
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)

    print(input, input_path, system_type, cell_para, debug_mode, name, extension)
    if not os.path.exists(input_path):
        exit_with_error(f"Input file not found: {input_path}")
    if extension == ".cif":
        if prefilter_cif(input_path):
            handle_cif_file(input_path, system_type, name, current_dir, debug_mode)
        else:
            exit_with_error("CIF file is not suitable for processing")
    elif extension == ".xyz":
        handle_xyz_file(input_path, system_type, name, cell_para, current_dir, debug_mode)
    else:
        exit_with_error("Invalid file extension")

def handle_cif_file(input_path, system_type, name, current_dir, debug_mode):
    if system_type == "reference":
        print("Processing reference (Wyckoff sites) from .cif file")
        process_refcell(input_path, name, current_dir, debug_mode)
    elif system_type == "unitcell":
        print("Processing unit cell from .cif file")
        process_unitcell(input_path, name, current_dir, debug_mode)
    else:
        exit_with_error("Invalid system type for .cif file", {"system_type": system_type})


def handle_xyz_file(input_path, system_type, name, cell_para, current_dir, debug_mode):
    if system_type == "unitcell":
        if cell_para is None:
            exit_with_error("Cell parameters must be provided for .xyz file of a unit cell")
        else:
            print("Processing unit cell from .xyz file")
            # users should provide chemical formula (Fe-O2-H3) of refmoleculist
            # if users provide smiles, we check compare_species in connectivity module
    elif system_type == "molecule":
        print("Processing molecule from .xyz file")
        get_molecule(input_path, name, current_dir, debug_mode)
    else:
        exit_with_error("Invalid system type for .xyz file", {"system_type": system_type})


if __name__ == "__main__" or __name__ == "cell2mol.final_c2m_driver":
    main()