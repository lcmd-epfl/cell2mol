import pickle
import sys
import os

from cell2mol.tmcharge_common import Cell, atom, molecule, ligand, metal
from cell2mol.readwrite import writexyz, print_molecule

pwd = os.getcwd()
pwd = pwd.replace("\\", "/")

cellfile = sys.argv[1]

if cellfile.endswith(".gmol"):
    with open(pwd + "/" + cellfile, "rb") as pickle_file:
        cell = pickle.load(pickle_file)
        print("READING", cell.refcode)
        print("")

        if not hasattr(cell, "cellvec"):
            print("Please load a cell object. Stopping")
            exit()
        
        print("Cell Warning List", cell.warning_list)
        print("")
        print("#### PRINTING MOLS ####")
        for mol in cell.moleclist:
            if hasattr(mol, "totcharge"):     print("MOL:", mol.type, mol.formula, mol.totcharge, mol.totmconnec) 
            if not hasattr(mol, "totcharge"): print("MOL:", mol.type, mol.formula, mol.totmconnec) 

            if mol.type == "Complex":
                for lig in mol.ligandlist: 
                    if hasattr(lig, "totcharge"):     print("    LIG:", lig.formula, lig.totcharge, lig.totmconnec, lig.smiles)
                    if not hasattr(lig, "totcharge"): print("    LIG:", lig.formula, lig.totmconnec)
                for met in mol.metalist: 
                    if hasattr(met, "totcharge"):     print("    MET:", met.atlist, met.label, met.totcharge, met.totmconnec)
                    if not hasattr(met, "totcharge"): print("    MET:", met.atlist, met.label, met.totmconnec)

        print("")
        print("#### PRINTING SPECS ####")
        for mol in cell.speclist:
            if mol.type != "Metal":
                if hasattr(mol, "totcharge"):     print("MOL:", mol.type, mol.formula, mol.totcharge, mol.totmconnec, mol.smiles) 
                if not hasattr(mol, "totcharge"): print("MOL:", mol.type, mol.formula, mol.totmconnec) 
            else:
                if hasattr(mol, "totcharge"):     print("MOL:", mol.type, mol.label, mol.totcharge, mol.totmconnec) 
                if not hasattr(mol, "totcharge"): print("MOL:", mol.type, mol.label, mol.totmconnec) 

            if mol.type == "Complex":
                for lig in mol.ligandlist: 
                    if hasattr(lig, "totcharge"):     print("    LIG:", lig.formula, lig.totcharge, lig.totmconnec, lig.smiles)
                    if not hasattr(lig, "totcharge"): print("    LIG:", lig.formula, lig.totmconnec)
                for met in mol.metalist: 
                    if hasattr(met, "totcharge"):     print("    MET:", met.atlist, met.label, met.totcharge, met.totmconnec)
                    if not hasattr(met, "totcharge"): print("    MET:", met.atlist, met.label, met.totmconnec)
        
    
else:
    print("File does not have .gmol extension")
    exit()

