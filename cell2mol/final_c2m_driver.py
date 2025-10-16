import os
from cell2mol.helper import parsing_arguments
from cell2mol.refcell import process_refcell
from cell2mol.unitcell import process_unitcell
from cell2mol.xyz_molecule import get_molecule
from cell2mol.read_write import (
    screening_cif,
    prefilter_cif,
    exit_with_error_input,
    exit_with_error_exception,
)


def main():
    input, system_type, cif_bond_info, cell_para, input_charge, debug_mode = (
        parsing_arguments()
    )
    current_dir = os.getcwd()
    input_path = os.path.normpath(input)
    dir, file = os.path.split(input_path)
    name, extension = os.path.splitext(file)

    # print(input, input_path, system_type, cell_para, debug_mode, name, extension)

    if not os.path.exists(input_path):
        exit_with_error_input(f"Input file not found: {input_path}")
    if extension == ".cif":
        # Check for radical, disorder, 3D fractional coordinates, and polymeric structure
        radical, disorder, notfound_atom, polymeric = screening_cif(input_path)
        cif_okay, error_message = prefilter_cif(input_path)
        if cif_okay:
            # if not any([notfound_atom]):
            handle_cif_file(
                input_path, system_type, name, current_dir, cif_bond_info, debug_mode
            )
        else:
            exit_with_error_input(
                f"CIF file is not suitable for processing {error_message}"
            )
    elif extension == ".xyz":
        handle_xyz_file(
            input_path,
            system_type,
            name,
            cell_para,
            input_charge,
            current_dir,
            debug_mode,
        )
    else:
        exit_with_error_input(f"Invalid file extension: {input_path}")


def handle_cif_file(
    input_path, system_type, name, current_dir, cif_bond_info, debug_mode
):
    if system_type == "reference":
        print("Processing reference (Wyckoff sites) from .cif file")
        try:
            process_refcell(input_path, name, current_dir, cif_bond_info, debug_mode)
        except Exception as e:
            exit_with_error_exception(e)
    elif system_type == "unitcell":
        print("Processing unit cell from .cif file")
        try:
            process_unitcell(input_path, name, current_dir, cif_bond_info, debug_mode)
        except Exception as e:
            exit_with_error_exception(e)
    else:
        exit_with_error_input(
            "Invalid system type for .cif file", {"system_type": system_type}
        )


def handle_xyz_file(
    input_path, system_type, name, cell_para, input_charge, current_dir, debug_mode
):
    if system_type == "unitcell":
        if cell_para is None:
            exit_with_error_input(
                "Cell parameters must be provided for .xyz file of a unit cell"
            )
        else:
            print("Processing unit cell from .xyz file")
            # users should provide chemical formula (Fe-O2-H3) of refmoleculist
            # if users provide smiles, we check compare_species in connectivity module
    elif system_type == "molecule":
        print("Processing molecule from .xyz file")
        get_molecule(input_path, name, input_charge, current_dir, debug_mode)
    else:
        exit_with_error_input(
            "Invalid system type for .xyz file", {"system_type": system_type}
        )


if __name__ == "__main__" or __name__ == "cell2mol.final_c2m_driver":
    main()
