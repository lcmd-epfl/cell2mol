# cell2mol


Program that interprets .cif files of molecular crystals and retrieves structural, connectivity and charge information of all molecules present in the unit cell. This includes (but is not limited to):

- Atomic coordinates, labels
- Total molecular charge, and formal atomic charges, including metal oxidation states
- Connectivity network as defined by either the adjacency matrix, or the bond-order matrix. 

The program generates a so-called "cell" object, with hierarchical information on the unit cell:
"Cells" have "Molecules". "Molecules that hold a transition metal are considered a "Complex". Complexes are made of "Ligands" and "Metals". "Ligands" are made of "Groups" of connected "Atoms". 

"Complexes", "Ligands", "Metals" and "Groups" host the information of their constituent "Atoms". That way, cell2mol provides a very in-depth interpretation of unit cells, which can be particularly useful to generate controlled databases for Quantum Chemistry and Machine Learning applications 
