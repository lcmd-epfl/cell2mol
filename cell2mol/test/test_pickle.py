#!/usr/bin/env python

from array import array
import os
import pytest
import numpy as np
import pickle

from cell2mol.tmcharge_common import Cell
from cell2mol.c2m_module import cell2mol


def run_cell2mol(infofile):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    root = infofile.split(".")
    refcode = root[0]
    outfile = f"Cell_{refcode}.gmol"

    infopath = os.path.join(dir_path, "infodata", infofile)
    outpath = os.path.join(dir_path, "infodata", outfile)

    cell = cell2mol(infopath, refcode, outpath)

    return cell


@pytest.mark.parametrize(
    "refcode",
    ["DAPGAF", "EGITOF", "HACXOY", "ISIPIJ", "KANYUT", "YOXKUS", "ROKQAM", "LOKXUG"],
)
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
    print(result.pos)
    print(result.warning_list)

    assert cell.refcode == result.refcode
    assert cell.cellvec == result.cellvec
    assert cell.cellparam == result.cellparam
    assert cell.labels == result.labels
    assert np.allclose(cell.pos, result.pos)
    assert cell.warning_list == result.warning_list
