#!/usr/bin/env python

from array import array
import os
import numpy as np
import pickle

from cell2mol.cell2mol import cell2mol, split_infofile

def test_cell2mol():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    infofile = "YOXKUS.info"
    infopath = dir_path + "/infodata/" + infofile
    refcode = split_infofile(infofile)
    cell = cell2mol(infopath, refcode)
    
    return cell

def test_check_cell():
    
    cell = test_cell2mol()
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cellpath = dir_path + "/infodata" 
    print(cellpath)
    file = open(f"{cellpath}/Cell_{cell.refcode}.gmol",'rb')
    result = pickle.load(file)
    
    print(result.version)
    print(result.refcode)
    print(result.cellvec)
    print(result.cellparam)
    print(result.labels)
    print(result.pos)
    pass

    # assert cell.version == result.version
    # assert cell.refcode == result.refcode
    # assert cell.cellvec == result.cellvec
    # assert cell.cellparam == result.cellparam
    # assert cell.labels == result.labels
    # assert np.allclose(cell.pos, result.pos)
    # assert cell.warning_list == result.warning_list
