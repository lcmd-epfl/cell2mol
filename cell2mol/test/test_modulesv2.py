#!/usr/bin/env python

import os 
# from cell2mol.module1 import test_addone, test_subtwo
from cell2mol.module1 import addone, subtwo
from cell2mol.cell2mol import cell2mol, split_infofile

def test_modules1():
  assert addone(10) == 11
  assert subtwo(8) == 6 

def test_cell2mol():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  infofile = dir_path+"/infodata/YOXKUS.info"
  refcode = split_infofile(infofile)
  cell = cell2mol(infofile, refcode)

# if __name__== "__main__":
#   test_modules1() 
