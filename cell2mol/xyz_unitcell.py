from ase.io import read


def process_unitcell_from_xyz(input_path, name, cell_para, current_dir, debug_mode):
    labels, pos, ref_labels, ref_fracs, cellvec, cellparam = readinfo(infopath)
    atoms = read(input_path)
    newcell = cell.from_positional(name, labels, pos, cellvec, cellparam)

    ## Get the fragments, which is the moleclist of a fragmented cell
    fragments = newcell.get_moleclist(
        cov_factor=cov_factor, metal_factor=metal_factor, debug=debug
    )
