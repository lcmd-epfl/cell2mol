#!/usr/bin/env python

from array import array
import os
import numpy as np
import pickle

from cell2mol.tmcharge_common import Cell
from cell2mol.c2m_module import cell2mol, split_infofile


def run_cell2mol(infofile):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    infopath = dir_path + "/infodata/" + infofile
    refcode = split_infofile(infofile)
    outfile = f"Cell_{refcode}.gmol"
    outpath = dir_path + "/infodata/" + outfile
    cell = cell2mol(infopath, refcode, outpath)
    return cell


def test_check_cell_vs_pickle():
    infofile = "YOXKUS.info"
    cell = run_cell2mol(infofile)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    cellpath = dir_path + "/infodata/"
    file = open(f"{cellpath}Cell_{cell.refcode}.gmol", "rb")
    result = pickle.load(file)
    print("=======Result======")
    print(result.version)
    print(result.refcode)
    print(result.cellvec)
    print(result.cellparam)
    print(result.labels)
    print(result.pos)
    print(result.warning_list)

    assert cell.version == result.version
    assert cell.refcode == result.refcode
    assert cell.cellvec == result.cellvec
    assert cell.cellparam == result.cellparam
    assert cell.labels == result.labels
    assert np.allclose(cell.pos, result.pos)
    assert cell.warning_list == result.warning_list


if __name__ == "__main__":
    test_check_cell_vs_pickle()
