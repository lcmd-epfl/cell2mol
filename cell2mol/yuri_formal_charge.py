#!/usr/bin/env python

import numpy as np  
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()

#############################
### Loads Rdkit & xyz2mol ###
#############################

from rdkit import Chem
from rdkit.Chem.Draw.MolDrawing import DrawingOptions  # Only needed if modifying defaults
DrawingOptions.bondLineWidth = 2.2

# IPythonConsole.ipython_useSVG = False
from rdkit import rdBase
if "ipykernel" in sys.modules:
    try:
        from rdkit.Chem.Draw import IPythonConsole
    except ModuleNotFoundError:
        pass
# print("RDKIT Version:", rdBase.rdkitVersion)
rdBase.DisableLog("rdApp.*")

def classify_mols(cell: object, debug: int=0):
    """ Identify unique species in the cell
    """
    unique_spices = []
    metal_coord_sphere_types = []
    unique_indices = []

    for mol in cell.moleclist:
        if mol.iscomplex: 
            for metal in mol.metals:
                done = False
                for index, coord_atoms_indices in enumerate(metal_coord_sphere_types):
                    if coord_atoms_indices == [ atom.index for atom in metal.get_coord_sphere() ] and not done:
                        done = True
                        # index = 

                if not done:
                    index = len(unique_spices)
                    coord_atoms_indices = [ atom.index for atom in metal.get_coord_sphere() ]
                    metal_coord_sphere_types.append(coord_atoms_indices)
                    unique_spices.append(list[metal.subtype, metal, mol])
                unique_indices.append(index)


                # compare_atoms(metal, unique_spicies, debug)                


            for ligand in mol.ligands:
                occurrence = cell.get_occurrences(ligand, debug)
        occurrence = cell.get_occurrences(mol, debug)
