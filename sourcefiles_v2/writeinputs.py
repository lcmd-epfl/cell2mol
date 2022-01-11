#!/usr/bin/env python3
import sys
import os
import numpy as np
import pickle
from collections import Counter

import tmcharge_common
from tmcharge_common import atom
from tmcharge_common import molecule
from tmcharge_common import metal
from tmcharge_common import ligand

#######################
### Loads Functions ###
#######################

utilspath = 'C:/Users/sergi/Documents/PostDoc/Marvel_TM_Database/Get_TMCharge/sourcefiles'
sys.path.append(utilspath)

#########################
# Loads Element Library #
#########################

elempath = 'C:/Users/sergi/Documents/PostDoc/Marvel_TM_Database/Get_TMCharge/sourcefiles'
elempath = elempath.replace("\\","/")

sys.path.append(elempath)
from elementdata import ElementData
elemdatabase = ElementData()

#######################
def read_gmols(folder):

    moleclist = []
    namelist = []
    for file in os.listdir(folder):
        if file.endswith(".gmol"):
            with open(folder+file, 'rb') as gmol:
                loadedmol = pickle.load(gmol)
                splitname = file.split(".")
                corename = splitname[0]
                moleclist.append(loadedmol)
                namelist.append(corename)
            
    return moleclist, namelist  

#######################
def ORCA_input(mol, folder, name, extension, nprocs=4, functional="B3LYP", basis="def2-TZVP", printdensity=False):

    with open(folder+name+extension, 'w') as inp:
        print("%pal", file=inp)
        print(" nprocs=", nprocs, file=inp)
        print("end", file=inp)
        print("", file=inp)
        print("! UKS", functional, basis, "def2/JK RIJK PrintBasis", file=inp)
        if printdensity:
            print("%output", file=inp)
            print("Print[P_Density] 1", file=inp)
        print("end", file=inp)
        print("", file=inp)

        if hasattr(mol, 'spin'):
            posspin = mol.spin
        else:
            if ((mol.eleccount + mol.totcharge) % 2 == 0):
                posspin = 1
            else:
                posspin = 2

        print("*xyz", mol.totcharge, posspin, file=inp)
        for a in mol.atoms:
            print("%s  %.6f  %.6f  %.6f" % (a.label, a.coord[0], a.coord[1], a.coord[2]), file=inp)
        print("*", file=inp)

#######################
def get_pp(ppfolder, elem):
    
    Found = False
    while not Found:
        for file in os.listdir(ppfolder):
            with open(ppfolder+file, 'rb') as pp:
                splitpp = file.split(".")
                if (len(splitpp) == 3):
                    element=splitpp[0]
                    functional=splitpp[1]
                    typ=splitpp[2]
                elif (len(splitpp) == 4):
                    element=splitpp[0]
                    functional=splitpp[1]
                    typ1=splitpp[2]
                    typ2=splitpp[3]
                else:
                    print("Can't interpret pp name for", file)
                    
                if (element == elem):
                    Found = True
                    sendpp = file
                    break
        
        if file == os.listdir(pppath)[-1] and not Found:
            sendpp = "Not found"
            break
                    
    return sendpp

#######################
def QE_input(mol, folder, name, extension, PP_Library, cubeside, typ="scf", isHubbard=False, cutoff=70, isGrimme=True):
 
    #IDENTIFIES METALS
    elems = []
    thereismetal = []
    for idx, l in enumerate((list(set(mol.labels)))):
        elems.append(l)
        if (elemdatabase.elementblock[l] == 'd' or elemdatabase.elementblock[l] == 'f'):
            thereismetal.append(True)
        else:
            thereismetal.append(False)
    if any(thereismetal):
        nummetal = sum(thereismetal)
        trues = [i for i, x in enumerate(thereismetal) if x]
        
    # IS THERE SPIN?
    if hasattr(mol, 'spin'):
        posspin = mol.spin
    else:
        if ((mol.eleccount + mol.totcharge) % 2 == 0):
            posspin = 1         #Singlet
            magnetization = 0   #Singlet 
        else:
            posspin = 2         #Doublet
            magnetization = 1   #Doublet
    
    with open(folder+name+extension, 'w') as inp:
        print(" &control", file=inp)
        print("    calculation=", typ, file=inp)
        print("    restart_mode='from_scratch'", file=inp)
        print("    pseudo_dir =", PP_Library, file=inp)
        print("    disk_io='low'", file=inp)
        print("    outdir='/scratch/velallau/QE_WFC/'", file=inp)
        print("    prefix=", name, file=inp)
        print("/", file=inp)
        print(" &system", file=inp)
        print("    ibrav=1, celldm(1)=", cubeside, file=inp)
        print("    nat=", mol.natoms, "ntyp=", len(list(set(mol.labels))), ", ecutwfc =", cutoff, ", ecutrho =", cutoff*8, file=inp)
        print("    nspin=2,", file=inp)
        
        if hasattr(mol, 'totcharge'):
            print("    tot_charge=", mol.totcharge, file=inp)
        else:
            print("    tot_charge=", 0, file=inp)
        
        if any(thereismetal): 
            for idx, elem in enumerate(elems):
                totalmagnetization = 0
                if idx in enumerate(trues):
                    print("    starting_magnetization(",idx,")=",magnetization, file=inp)
                    totalmagnetization += magnetization
                else:
                    pass
                print("    tot_magnetization=", totalmagnetization, file=inp)
        
#         if isHubbard:
#             print("    lda_plus_u=.true.,", file=inp)
#             print("    Hubbard_U(1)=2.35", file=inp)
#             print("    Hubbard_U(2)=2.35", file=inp)
        if isGrimme:
            print("    vdw_corr='grimme-d3'", file=inp)
            print("    dftd3_version=4", file=inp)
            
        print("    assume_isolated='mp'", file=inp)
        print("/", file=inp)
        
        print(" &electrons", file=inp)
        print("    diagonalization='david'", file=inp)
        print("    electron_maxstep=150", file=inp)
        print("    conv_thr = 1.0e-5", file=inp)
        print("    mixing_beta = 0.20", file=inp)
        print("/", file=inp)
        
        print("ATOMIC_SPECIES", file=inp)
        for idx, elem in enumerate(elems):
            label = elem
            weight = elemdatabase.elementweight[elem]
            pp = get_pp(PP_Library, elem)
            print(label, weight, pp, file=inp)
        
        print("ATOMIC_POSITIONS angstrom", file=inp)
        for a in mol.atoms:
            print("%s  %.6f  %.6f  %.6f" % (a.label, a.coord[0], a.coord[1], a.coord[2]), file=inp)
        print("K_POINTS gamma", file=inp)


