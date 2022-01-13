#!/usr/bin/env python

# from cell2mol.module1 import test_addone, test_subtwo
from cell2mol.module1 import addone, subtwo

def test_modules1():
  assert addone(10) == 11
  assert subtwo(8) == 6 

# def test_cell2mol ():
  
#   cell2mol()

# if __name__== "__main__":
#   test_modules1() 
