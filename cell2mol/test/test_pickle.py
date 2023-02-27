#!/usr/bin/env python

import os
import pytest
import numpy as np
import pickle

from cell2mol.c2m_module import cell2mol
from cell2mol.c2m_module import load_cell_reset_charges


def run_cell2mol(infofile):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    root = infofile.split(".")
    refcode = root[0]
    outfile = f"Cell_{refcode}.gmol"

    infopath = os.path.join(dir_path, "infodata", infofile)
    outpath = os.path.join(dir_path, "infodata", outfile)

    cell = cell2mol(infopath, refcode, outpath)

    return cell


@pytest.mark.parametrize("refcode",["DAPGAF", "EGITOF", "ISIPIJ", "KANYUT", "YOXKUS", "ROKQAM", "LOKXUG"])
def test_check_info_vs_pickle(refcode):
    infofile = f"{refcode}.info"
    cell = run_cell2mol(infofile)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cellpath = os.path.join(dir_path, "infodata", f"Cell_{refcode}.gmol")

    file = open(cellpath, "rb")
    result = pickle.load(file)
    print("=======Result======")
    print(result.refcode)
    print(result.cellvec)
    print(result.cellparam)
    print(result.labels)
    print(result.atom_coord)
    print(result.warning_list)

    assert cell.refcode == result.refcode
    assert cell.cellvec == result.cellvec
    assert cell.cellparam == result.cellparam
    assert cell.labels == result.labels
    assert np.allclose(cell.atom_coord, result.atom_coord)
    assert cell.warning_list == result.warning_list
    assert cell.warning_after_reconstruction == result.warning_after_reconstruction


@pytest.mark.parametrize("refcode",["DAPGAF", "EGITOF", "ISIPIJ", "KANYUT", "YOXKUS", "ROKQAM", "LOKXUG"])
def test_load_cell_reset_charges (refcode):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    cellpath = os.path.join(dir_path, "infodata", f"Cell_{refcode}.gmol")

    file = open(cellpath, "rb")
    cell = pickle.load(file)

    temp = load_cell_reset_charges(cellpath)

    assert temp.warning_list == cell.warning_after_reconstruction
    assert temp.warning_after_reconstruction == cell.warning_after_reconstruction
