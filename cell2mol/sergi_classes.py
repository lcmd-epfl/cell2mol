import numpy as np
from Scope.Adapted_from_cell2mol import * 
from Scope.Other import get_metal_idxs
from Scope.Unit_cell_tools import * 
#from Scope.Reconstruct import * 

################################
####  BASIS FOR CELL2MOL 2  ####
################################
class specie(object):
    def __init__(self, labels: list, coord: list, indices: list=None, radii: list=None, parent: object=None) -> None:

       # Sanity Checks
        assert len(labels) == len(coord)
        if indices is not None: assert len(labels) == len(indices)
        if radii   is not None: assert len(labels) == len(radii)

        # Optional Information
        if radii   is not None: self.radii   = radii
        else:                   self.radii   = get_radii(labels)
        if parent  is not None: self.parent = parent
#        if parent  is not None: self.occurence = parent.get_occurrence(self)

        self.type      = "specie"
        self.version   = "0.1"
        self.labels    = labels
        self.coord     = coord
        self.formula   = labels2formula(labels)
        self.eleccount = labels2electrons(labels)
        self.natoms    = len(labels)
        self.iscomplex = any((elemdatabase.elementblock[l] == "d") or (elemdatabase.elementblock[l] == "f") for l in self.labels)

        if indices is not None: self.indices = indices
        else:                   self.indices = [*range(0,self.natoms,1)]

        self.cov_factor   = 1.3
        self.metal_factor = 1.0

    def get_centroid(self):
        self.centroid = compute_centroid(self.coord)
        if hasattr(self,"frac_coord"): self.frac_centroid = compute_centroid(self.frac_coord)
        return self.centroid

    def set_fractional_coord(self, frac_coord: list) -> None:
        self.frac_coord = frac_coord 

    def get_fractional_coord(self, cell_vector=None) -> None:
        from Scope.Reconstruct import cart2frac
        if cell_vector is None:
            if hasattr(self,"parent"):
                if hasattr(self.parent,"cellvec"): cell_vector = self.parent.cellvec.copy()
            else:     print("CLASS_SPECIE: get_fractional coordinates. Missing cell vector. Please provide it"); return None
        else:         
            self.frac_coord = cart2frac(self.coord, cell_vector)
        return self.frac_coord

    def get_atomic_numbers(self):
        if not hasattr(self,"atoms"): self.set_atoms()
        self.atnums = []
        for at in self.atoms:
            self.atnums.append(at.atnum)
        return self.atnums

    def set_atoms(self, atomlist: object=None):
        if atomlist is None:
            self.atoms = []
            for idx, l in enumerate(self.labels):
                newatom = atom(l, self.coord[idx], parent=self, index=idx, radii=self.radii[idx])
                self.atoms.append(newatom)
        else:
            self.atoms = atomlist

    def set_element_count(self, heavy_only: bool=False):
        self.element_count = get_element_count(self.labels, heavy_only=heavy_only)
        return self.element_count

    def set_adj_types(self):
        if not hasattr(self,"adjmat"): self.get_adjmatrix()
        self.adj_types = get_adjacency_types(self.labels, self.adjmat)
        return self.adj_types

    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        # Stores the covalentradii factor and metal factor that were used to generate the molecule
        self.cov_factor   = cov_factor
        self.metal_factor = metal_factor

    def set_charges(self, totcharge: int=None, atomic_charges: list=None) -> None:
        ## Sets total charge  
        if totcharge is not None:                              self.totcharge = totcharge
        elif totcharge is None and atomic_charges is not None: self.totcharge = np.sum(atomic_charges)
        elif totcharge is None and atomic_charges is None:     self.totcharge = "Unknown" 
        ## Sets atomic charges
        if atomic_charges is not None:
            self.atomic_charges = atomic_charges
            if not hasattr(self,"atoms"): self.set_atoms()
            for idx, a in enumerate(self.atoms):
                a.set_charge(self.atomic_charges[idx])

    def get_adjmatrix(self):
        isgood, adjmat, adjnum = get_adjmatrix(self.labels, self.coord, self.cov_factor, self.radii)
        if isgood:
            self.adjmat = adjmat
            self.adjnum = adjnum
        else:
            self.adjmat = None
            self.adjnum = None
        return self.adjmat, self.adjnum

    def get_occurrence(self, substructure: object) -> int:
        occurrence = 0
        ## Ligands in Complexes or Groups in Ligands
        done = False
        if hasattr(substructure,"subtype") and hasattr(self,"subtype"):
            if substructure.subtype == 'ligand' and self.subtype == 'molecule':
                if not hasattr(self,"ligands"): self.split_complex()
                if self.ligands is not None:
                    for l in self.ligands:
                        issame = compare_species(substructure, l, debug=1)
                        if issame: occurrence += 1
                    done = True 
            elif substructure.subtype == 'group' and self.subtype == 'ligand':
                if not hasattr(self,"ligands"): self.split_complex()
                if self.ligands is not None:
                    for l in self.ligands:
                        if not hasattr(l,"groups"): self.split_ligand()
                        for g in l.groups:
                            issame = compare_species(substructure, g, debug=1)
                            if issame: occurrence += 1
                done = True 
        ## Atoms in Species
        if not done:
            if substructure.type == 'atom' and self.type == 'specie':
                if not hasattr(self,"atoms"): self.set_atoms()
                for at in self.atoms:
                    issame = compare_atoms(substructure, at)
                    if issame: occurrence += 1
        return occurrence

    def magnetism(self, spin: int) -> None:
        self.spin = spin

    def print_xyz(self):
        print(self.natoms)
        print("")
        for idx, l in enumerate(self.labels):
            print("%s  %.6f  %.6f  %.6f" % (l, self.coord[idx][0], self.coord[idx][1], self.coord[idx][2]))

    ## To be implemented
    def __add__(self, other):
        if not isinstance(other, type(self)): return self
        return self

    def __repr__(self):
        to_print  = f'---------------------------------------------------\n'
        to_print +=  '   >>> SPECIE >>>                                  \n'
        to_print += f'---------------------------------------------------\n'
        to_print += f' Version               = {self.version}\n'
        to_print += f' Type                  = {self.type}\n'
        if hasattr(self,'subtype'): to_print += f' Sub-Type              = {self.subtype}\n'
        to_print += f' Number of Atoms       = {self.natoms}\n'
        to_print += f' Formula               = {self.formula}\n'
        if hasattr(self,"adjmat"):     to_print += f' Has Adjacency Matrix  = YES\n'
        else:                          to_print += f' Has Adjacency Matrix  = NO \n'
        if hasattr(self,"parent"):    
            if self.parent is not None:    to_print += f' Has Parent            = YES\n'
            else:                          to_print += f' Has Parent            = NO \n'
        if hasattr(self,"occurrence"): to_print += f' Occurrence in Parent  = {self.occurrence}\n'
        if hasattr(self,"totcharge"):  to_print += f' Total Charge          = {self.totcharge}\n'
        if hasattr(self,"spin"):       to_print += f' Spin                  = {self.spin}\n'
        if hasattr(self,"smiles"):     to_print += f' SMILES                = {self.smiles}\n'
        to_print += '---------------------------------------------------\n'
        return to_print

###############
### MOLECULE ##
###############
class molecule(specie):
    def __init__(self, labels: list, coord: list, indices: list=None, radii: list=None, parent: object=None) -> None:
        self.subtype = "molecule"
        specie.__init__(self, labels, coord, indices, radii, parent)

############
    def split_complex(self, debug: int=0):
        if not self.iscomplex: 
            self.ligands = None
            self.metals  = None
            #print("MOLECULE.SPLIT_COMPLEX: This molecule is not a transition metal complex");
    
        else: 
            self.ligands = []
            self.metals  = []
            # Identify Metals and the rest
            metal_idx = get_metal_idxs(self.labels, debug=debug)
            rest_idx  = list(idx for idx in range(0,len(self.labels)) if idx not in metal_idx) 
            if debug > 0: 
                print(f"SPLIT COMPLEX: found metals in indices {metal_idx}")
                print(f"SPLIT COMPLEX: rest of indices:        {rest_idx}")
            #rest_idx  = list(idx for idx in self.indices if idx not in metal_idx)
            # Split the "rest" to obtain the ligands
            rest_labels  = extract_from_list(rest_idx, self.labels, dimension=1)
            rest_coords  = extract_from_list(rest_idx, self.coord, dimension=1)
            rest_indices = None
            #rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)
            rest_radii   = extract_from_list(rest_idx, self.radii, dimension=1)
            if hasattr(self,"frac_coord"): rest_frac    = extract_from_list(rest_idx, self.frac_coord, dimension=1)
            if debug > 0: print(f"SPLIT COMPLEX: rest labels: {rest_labels}")
            if debug > 0: print(f"SPLIT COMPLEX: splitting species with {len(rest_labels)} atoms in block")
            if hasattr(self,"cov_factor"): blocklist = split_species(rest_labels, rest_coords, radii=rest_radii, indices=rest_indices, cov_factor=self.cov_factor, debug=debug)
            else:                          blocklist = split_species(rest_labels, rest_coords, radii=rest_radii, indices=rest_indices, debug=debug)
            ## Arranges Ligands
            for b in blocklist:
                if debug > 0: print(f"PREPARING BLOCK: {b}")
                lig_labels  = extract_from_list(b, rest_labels, dimension=1) 
                lig_coord   = extract_from_list(b, rest_coords, dimension=1) 
                #lig_indices = extract_from_list(b, rest_indices, dimension=1) 
                lig_radii   = extract_from_list(b, rest_radii, dimension=1) 
                newligand   = ligand(lig_labels, lig_coord, indices=None, radii=lig_radii, parent=self)
                #newligand   = ligand(lig_labels, lig_coord, indices=lig_indices, radii=lig_radii, parent=self)
                # If fractional coordinates are available...
                if hasattr(self,"frac_coord"): 
                    lig_frac_coords  = extract_from_list(b, rest_frac_coord, dimension=1)
                    newligand.set_fractional_coord(lig_frac_coords)
                self.ligands.append(newligand)
            ## Arranges Metals
            for m in metal_idx:
                newmetal    = metal(self.labels[m], self.coord[m], self.indices[m], self.radii[m], parent=self)
                self.metals.append(newmetal)
        return self.ligands, self.metals
        
    #def get_metal_adjmatrix(self):
    #    if not hasattr(self,"adjmat"): self.get_adjmatrix() 
    #    if self.adjmat is None: return None, None
    #    madjmat = np.zeros((self.natoms,self.natoms))
    #    madjnum = np.zeros((self.natoms)) 
    #    metal_idx = get_connected_idx(self, debug=debug)
    #    for idx, row in enumerate(self.adjmat):
    #        for jdx, col in enumerate(row):
    #            if not idx in metal_idx: madjmat[idx,jdx] = int(0) 
    #            else:                    madjmat[idx,jdx] = adjmat[idx,jdx]
    #    for i in range(0, len(adjmat)):
    #        madjnum[i] = np.sum(madjmat[i, :])
    #    return self.madjmat, self.madjnum

############

###############
### LIGAND ####
###############
class ligand(specie):
    def __init__(self, labels: list, coord: list, indices: list=None, radii: list=None, parent: object=None) -> None:
        self.subtype  = "ligand"
        specie.__init__(self, labels, coord, indices, radii, parent)

    def set_hapticity(self, hapttype):
        self.hapticity = True 
        self.hapttype  = hapttype 

    def get_connected_idx(self, debug: int=0):
       self.connected_idx = [] 
       if not hasattr(self.parent,"adjmat"): madjmat, madjnum = self.parent.get_metal_adjmatrix()
       self.madjmat = extract_from_list(self.indices, madjmat, dimension=2)
       self.madjnum = extract_from_list(self.indices, madjnum, dimension=1)
       for idx, con in enumerate(self.madjnum):
           if con > 0: self.connected_idx.append(idx) 
       return self.connected_idx 
        
    def split_ligand(self, debug: int=0):
        self.groups = []
        # Identify Connected and Unconnected atoms (to the metal)
        if not hasattr(self,"connected_idx"): self.get_connected_idx()
        conn_idx = self.connected_idx
        rest_idx = list(idx for idx in self.indices if idx not in conn_idx)

        # Split the "ligand to obtain the groups
        conn_labels  = extract_from_list(conn_idx, self.labels, dimension=1)
        conn_coords  = extract_from_list(conn_idx, self.coord, dimension=1)
        conn_indices = extract_from_list(conn_idx, self.indices, dimension=1)

        if hasattr(self,"cov_factor"): blocklist = split_species(conn_labels, conn_coords, indices=conn_indices, cov_factor=self.cov_factor)
        else:                          blocklist = split_species(conn_labels, conn_coords, indices=conn_indices)
        ## Arranges Groups 
        for b in blocklist:
            gr_labels  = extract_from_list(b, self.labels, dimension=1)
            gr_coord   = extract_from_list(b, self.coord, dimension=1)
            gr_indices = extract_from_list(b, self.indices, dimension=1)
            gr_radii   = extract_from_list(b, self.radii, dimension=1)
            newgroup   = group(gr_labels, gr_coord, gr_indices, gr_radii, parent=self)
            self.groups.append(newgroup)
            
        return self.groups

###############
#### GROUP ####
###############
class group(specie):
    def __init__(self, labels: list, coord: list, indices: list=None, radii: list=None, parent: object=None) -> None:
        self.subtype = "group"
        specie.__init__(self, labels, coord, indices, radii, parent)

###############
### ATOM ######
###############
class atom(object):
    def __init__(self, label: str, coord: list, index: int=None, radii: float=None, parent: object=None, frac_coord: list=None) -> None:
        self.type    = "atom"
        self.version = "0.1"
        self.label = label
        self.coord = coord
        self.atnum = elemdatabase.elementnr[label]
        self.block = elemdatabase.elementblock[label]

        if index is not None:      self.index = index
        if radii is None:          self.radii = getradii(label)
        else:                      self.radii = radii
        if parent is not None:     self.parent  = parent
        if parent is not None:     self.occurence = parent.get_occurrence(self)
        if frac_coord is not None: self.frac_coord = frac_coord

    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        self.cov_factor   = cov_factor
        self.metal_factor = metal_factor

    def set_charge(self, charge: int) -> None:
        self.charge = charge 

    def information(self, cov_factor: float, metal_factor: float) -> None:
        self.cov_factor   = cov_factor
        self.metal_factor = metal_factor

    def __repr__(self):
        to_print  = f'---------------------------------------------------\n'
        to_print +=  '   >>> ATOM >>>                                    \n'
        to_print += f'---------------------------------------------------\n'
        to_print += f' Version               = {self.version}\n'
        to_print += f' Type                  = {self.type}\n'
        if hasattr(self,'subtype'): to_print += f' Sub-Type              = {self.subtype}\n'
        to_print += f' Label                 = {self.label}\n'
        to_print += f' Atomic Number         = {self.atnum}\n'
        to_print += f' Index                 = {self.index}\n'
        if hasattr(self,"occurrence"): to_print += f' Occurrence in Parent  = {self.occurrence}\n'
        if hasattr(self,"charge"):     to_print += f' Atom Charge           = {self.charge}\n'
        to_print += '----------------------------------------------------\n'
        return to_print

###############
#### METAL ####
###############
class metal(atom):
    def __init__(self, label: str, coord: list, index: int=None, radii: float=None, parent: object=None, frac_coord: list=None) -> None:
        self.subtype      = "metal"
        atom.__init__(self, label, coord, index=index, radii=radii, parent=parent, frac_coord=frac_coord)

    def get_coord_sphere(self):
        if not hasattr(self,"parent"): return None
        if not hasattr(self.parent,"adjmat"): self.parent.get_adjmatrix()
        adjmat = self.parent.adjmat.copy()
        connec_atoms_labels = []  
        for idx, at in enumerate(adjmat[self.index]):
            if at >= 1:
                connec_atoms_labels.append(str(self.parent.labels[idx]))
        self.coord_sphere    = labels2formula(connec_atoms_labels)
        return self.coord_sphere

##############
#### CELL ####
##############
class cell(object):
    def __init__(self, refcode: str, labels: list, pos: list, cellvec: list, cellparam: list) -> None:
        self.version    = "0.1"
        self.type       = "cell"
        self.refcode    = refcode
        self.labels     = labels 
        self.coord      = pos
        self.pos        = pos
        self.cellvec    = cellvec
        self.cellparam  = cellparam
        self.natoms     = len(labels)
 
    def set_fractional_coord(self, frac_coord: list) -> None:
        self.frac_coord = frac_coord 

    def set_moleclist(self, moleclist: list) -> None:
        self.moleclist = moleclist

    def get_moleclist(self, blocklist=None):
        if not hasattr(self,"labels") or not hasattr(self,"pos"): return None
        if len(self.labels) == 0 or len(self.pos) == 0: return None
        cov_factor = 1.3

        if blocklist is None: blocklist = split_species(self.labels, self.pos, cov_factor=cov_factor)
        self.moleclist = []
        for b in blocklist:
            mol_labels  = extract_from_list(b, self.labels, dimension=1)
            mol_coords  = extract_from_list(b, self.coord, dimension=1)
            newmolec    = molecule(mol_labels, mol_coords)
            # If fractional coordinates are available...
            if hasattr(self,"frac_coord"): 
                mol_frac_coords  = extract_from_list(b, self.frac_coord, dimension=1)
                newmolec.set_fractional_coord(mol_frac_coords)
            # This must be below the frac_coord, so they are carried on to the ligands
            if newmolec.iscomplex: 
                newmolec.split_complex()
            self.moleclist.append(newmolec)
        return self.moleclist
   
    def arrange_cell_coord(self): 
        ## Updates the cell coordinates preserving the original atom ordering
        ## Do do so, it uses the variable atlist stored in each molecule
        self.coord = np.zeros((self.natoms,3))
        for mol in self.moleclist:
            for z in zip(mol.indices, mol.coord):
                for i in range(0,3):
                    self.coord[z[0]][i] = z[1][i]
        self.coord = np.ndarray.tolist(self.coord)

    def get_occurrence(self, substructure: object) -> int:
        occurrence = 0
        ## Molecules in Cell
        if hasattr(substructure,"subtype") and hasattr(self,"moleclist"): 
            if substructure.subtype == 'molecule':
                for m in self.moleclist:
                    issame = compare_species(substructure, m)
                    if issame: occurrence += 1
        return occurrence

#    def reconstruct(self, debug: int=0):
#        if not hasattr(self,"refmoleclist"): print("CLASS_CELL missing list of reference molecules"); return None
#        from Scope.Other import HiddenPrints
#        import itertools
#        print("CLASS_CELL: reconstructing cell")
#        with HiddenPrints():
#            if not hasattr(self,"moleclist"): self.get_moleclist()
#            blocklist = self.moleclist.copy() # In principle, in moleclist now there are both fragments and molecules
#            refmoleclist = self.refmoleclist.copy()
#            cov_factor = refmoleclist[0].cov_factor
#            metal_factor = refmoleclist[0].metal_factor
#            ## Prepares Blocks
#            for b in blocklist:
#                if not hasattr(b,"frac_coord"):       b.get_fractional_coord(cellvec)
#                if not hasattr(b,"centroid"):         b.get_centroid()
#                if not hasattr(b,"element_count"):    b.set_element_count()
#                if not hasattr(b,"numH"):             b.numH = b.set_element_count()[4]
#            ## Prepares Reference Molecules
#            for ref in refmoleclist:
#                if not hasattr(ref,"element_count"):  ref.set_element_count()
#                if not hasattr(ref,"numH"):           ref.numH = ref.set_element_count()[4]
#            ## Classifies fragments
#            moleclist, fraglist, Hlist = classify_fragments(blocklist, moleclist, refmoleclist, self.cellvec)
#            ## Determines if Reconstruction is necessary
#            if len(fraglist) > 0 or len(Hlist) > 0: isfragmented = True
#            else:                                   isfragmented = False
#
#            if not isfragmented: return self.moleclist 
#            moleclist, finalmols, Warning = fragments_reconstruct(moleclist,fraglist,Hlist,refmoleclist,self.cellvec,cov_factor,metal_factor)
#            moleclist.extend(finalmols)
#            self.moleclist = moleclist
#            self.set_geometry_from_moleclist()
#        return self.moleclist

    def __repr__(self):
        to_print  = f'---------------------------------------------------\n'
        to_print +=  '   >>> CELL >>>                                    \n'
        to_print += f'---------------------------------------------------\n'
        to_print += f' Version               = {self.version}\n'
        to_print += f' Type                  = {self.type}\n'
        to_print += f' Refcode               = {self.refcode}\n'
        to_print += f' Num Atoms             = {self.natoms}\n'
        to_print += f' Cell Parameters a:c   = {self.cellparam[0:3]}\n'
        to_print += f' Cell Parameters al:ga = {self.cellparam[3:6]}\n'
        if hasattr(self,"moleclist"):  
            to_print += f' # Molecules:          = {len(self.moleclist)}\n'
            to_print += f' With Formulae:                               \n'
            for idx, m in enumerate(self.moleclist):
                to_print += f'    {idx}: {m.formula} \n'
        to_print += '---------------------------------------------------\n'
        if hasattr(self,"refmoleclist"):
            to_print += f' # of Ref Molecules:   = {len(self.refmoleclist)}\n'
            to_print += f' With Formulae:                                  \n'
            for idx, ref in enumerate(self.refmoleclist):
                to_print += f'    {idx}: {ref.formula} \n'
        return to_print


#####################
### SPLIT SPECIES ### 
#####################
#def split_species_from_object(obj: object, debug: int=0):
#
#    if not hasattr(obj,"adjmat"): obj.get_adjmatrix()
#    if obj.adjmat is None: return None
#
#    degree = np.diag(self.adjnum)  # creates a matrix with adjnum as diagonal values. Needed for the laplacian
#    lap = obj.adjmat - degree     # computes laplacian
#
#    # creates block matrix
#    graph = csr_matrix(lap)
#    perm = reverse_cuthill_mckee(graph)
#    gp1 = graph[perm, :]
#    gp2 = gp1[:, perm]
#    dense = gp2.toarray()
#
#    # detects blocks in the block diagonal matrix called "dense"
#    startlist, endlist = get_blocks(dense)
#
#    nblocks = len(startlist)
#    # keeps track of the atom movement within the matrix. Needed later
#    atomlist = np.zeros((len(dense)))
#    for b in range(0, nblocks):
#        for i in range(0, len(dense)):
#            if (i >= startlist[b]) and (i <= endlist[b]):
#                atomlist[i] = b + 1
#    invperm = inv(perm)
#    atomlistperm = [int(atomlist[i]) for i in invperm]
#
#    # assigns atoms to molecules
#    blocklist = []
#    for b in range(0, nblocks):
#        atlist = []    # atom indices in the original ordering
#        for i in range(0, len(atomlistperm)):
#            if atomlistperm[i] == b + 1:
#                atlist.append(indices[i])
#        blocklist.append(atlist)
#    return blocklist

######################
####    IMPORT    ####
######################
def import_cell(old_cell: object, debug: int=0) -> object:
    assert hasattr(old_cell,"labels") and (hasattr(old_cell,"coord") or hasattr(old_cell,"pos"))
    assert hasattr(old_cell,"cellvec")
    assert hasattr(old_cell,"refcode")


    labels     = old_cell.labels
    refcode    = old_cell.refcode
    if   hasattr(old_cell,"coord"):       coord      = old_cell.coord
    elif hasattr(old_cell,"pos"):         coord      = old_cell.pos

    cellvec    = old_cell.cellvec
    if   hasattr(old_cell,"cellparam"):   cellparam  = old_cell.cellparam
    else:                                 cellparam  = cellvec_2_cellparam(old_cell.cellvec)

    if   hasattr(old_cell,"frac_coord"):  frac_coord = old_cell.frac_coord
    else:                                 frac_coord = cart2frac(coord, old_cell.cellvec)

    new_cell = cell(refcode, labels, coord, cellvec, cellparam)
    new_cell.set_fractional_coord(frac_coord)
    if debug > 0: print(f"IMPORT CELL: created new_cell {new_cell}")

    ## Moleclist
    moleclist = []
    for mol in old_cell.moleclist: 
        if debug > 0: print(f"IMPORT CELL: importing molecule {mol.formula} with charge {mol.totcharge}")
        new_mol = import_molecule(mol, parent=new_cell, debug=debug)
        moleclist.append(new_mol)
    new_cell.set_moleclist(moleclist)

    ## Refmoleclist
    new_cell.refmoleclist = []
    if hasattr(old_cell,"refmoleclist"):
        for rmol in old_cell.refmoleclist:
            new_cell.refmoleclist.append(import_molecule(rmol))
    elif hasattr(new_cell,"moleclist"):
        for mol in new_cell.moleclist:
            found = False
            for rmol in new_cell.refmoleclist:
                issame = compare_species(rmol, mol)
                if issame: found = True 
            if not found: new_cell.refmoleclist.append(mol)

    ## Temporary things that I'd like to remove from the import once sorted 
    if hasattr(old_cell,"warning_list"): new_cell.warning_list = old_cell.warning_list 

    return new_cell

################################
def import_molecule(mol: object, parent: object=None, debug: int=0) -> object:
    assert hasattr(mol,"labels") and (hasattr(mol,"coord") or hasattr(mol,"pos"))
    labels     = mol.labels
    if   hasattr(mol,"coord"):       coord      = mol.coord
    elif hasattr(mol,"pos"):         coord      = mol.pos

    if   hasattr(mol,"indices"):     indices    = mol.indices
    elif hasattr(mol,"atlist"):      indices    = mol.atlist
    else:                            indices    = None

    if   hasattr(mol,"radii"):       radii      = mol.radii
    else:                            radii      = None          
 
    if parent is None: print(f"IMPORT MOLECULE {mol.formula}: parent is NONE")

    new_molec = molecule(labels, coord, indices, radii, parent)
    if debug > 0: print(f"IMPORT MOLEC: created new_molec with {new_molec.formula}")

    ## Charges
    if hasattr(mol,"totcharge") and hasattr(mol,"atcharge"):
        if debug > 0: print(f"IMPORT MOLEC: old molecule has total charge and atomic charges")
        new_molec.set_charges(mol.totcharge, mol.atcharge)
    elif hasattr(mol,"totcharge") and not hasattr(mol,"atcharge"):
        if debug > 0: print(f"IMPORT MOLEC: old molecule has total charge but no atomic charges")
        new_molec.set_charges(mol.totcharge)
    else:
        if debug > 0: print(f"IMPORT MOLEC: old molecule has no total charge nor atomic charges")

    ## Smiles
    if   hasattr(mol,"Smiles"): new_molec.smiles = mol.Smiles
    elif hasattr(mol,"smiles"): new_molec.smiles = mol.smiles        

    ## Substructures
    if not hasattr(mol,"ligandlist") or not hasattr(mol,"metalist"):  new_molec.split_complex()
    if hasattr(mol,"ligandlist"): 
        ligands = []
        for lig in mol.ligandlist: 
            if debug > 0: print(f"IMPORT MOLEC: old molecule has ligand {lig.formula}")
            new_lig = import_ligand(lig, parent=new_molec, debug=debug)
            ligands.append(new_lig)
        new_molec.ligands = ligands
    if hasattr(mol,"metalist"): 
        metals = []
        for met in mol.metalist: 
            if debug > 0: print(f"IMPORT MOLEC: old molecule has metal {met.label}")
            new_atom = import_atom(met, parent=new_molec, debug=debug)
            metals.append(new_atom)
        new_molec.metals = metals

    ## Atoms
    if not hasattr(mol,"atoms"):  
        if debug > 0: print(f"IMPORT MOLEC: old molecule has no atoms")
        new_molec.set_atoms()
    else: 
        atoms = []
        for at in mol.atoms: 
            if debug > 0: print(f"IMPORT MOLEC: old molecule has atom {at.label}")
            new_atom = import_atom(at, parent=new_molec, debug=debug)
            atoms.append(new_atom)
        new_molec.set_atoms(atoms)

    return new_molec

################################
def import_ligand(lig: object, parent: object=None, debug: int=0) -> object:
    assert hasattr(lig,"labels") and (hasattr(lig,"coord") or hasattr(lig,"pos"))
    labels     = lig.labels
    if   hasattr(lig,"coord"):       coord      = lig.coord
    elif hasattr(lig,"pos"):         coord      = lig.pos

    if   hasattr(lig,"indices"):     indices    = lig.indices
    elif hasattr(lig,"atlist"):      indices    = lig.atlist
    else:                            indices    = None

    if   hasattr(lig,"radii"):       radii      = lig.radii
    else:                            radii      = None          

    if debug > 0 and parent is None: print("IMPORT LIGAND: parent is NONE")

    new_ligand = ligand(labels, coord, indices, radii, parent)
    if debug > 0: print(f"IMPORT LIGAND: created new_ligand with {new_ligand.formula}")
    
    ## Charges
    if hasattr(lig,"totcharge") and hasattr(lig,"atcharge"):
        new_ligand.set_charges(lig.totcharge, lig.atcharge)
    elif hasattr(lig,"totcharge") and not hasattr(lig,"atcharge"):
        new_ligand.set_charges(lig.totcharge)

    ## Smiles
    if   hasattr(lig,"Smiles"): new_ligand.smiles = lig.Smiles
    elif hasattr(lig,"smiles"): new_ligand.smiles = lig.smiles     

    ## Rdkit Object
    if   hasattr(lig,"object"): new_ligand.rdkit_obj = lig.object
    
    ## Substructures
    if not hasattr(lig,"grouplist"):  new_ligand.split_ligand()
    else: 
        groups = []
        for gr in lig.grouplist: 
            new_group = import_group(gr, parent=new_ligand, debug=debug)
            groups.append(new_group)
    new_ligand.groups = groups
    
    ## Atoms
    if not hasattr(lig,"atoms"):  new_ligand.set_atoms()
    else: 
        atoms = []
        for at in lig.atoms: 
            new_atom = import_atom(at, parent=new_ligand, debug=debug)
            atoms.append(new_atom)
        new_ligand.set_atoms(atoms)
            
    return new_ligand

################################
def import_group(old_group: object, parent: object=None, debug: int=0) -> object:
    assert hasattr(old_group,"atlist") or hasattr(old_group,"indices")
    
    if   hasattr(old_group,"labels"):      labels     = old_group.labels
    elif parent is not None:
        if hasattr(parent,"labels"):       labels     = extract_from_list(old_group.atlist, parent.labels, dimension=1)
        
    if   hasattr(old_group,"coord"):       coord      = old_group.coord
    elif hasattr(old_group,"pos"):         coord      = old_group.pos
    elif parent is not None:
        if hasattr(parent,"coord"):        coord     = extract_from_list(old_group.atlist, parent.coord, dimension=1)
        elif hasattr(parent,"coord"):      coord     = extract_from_list(old_group.atlist, parent.pos, dimension=1)

    if   hasattr(old_group,"indices"):     indices    = old_group.indices
    elif hasattr(old_group,"atlist"):      indices    = old_group.atlist
    else:                                  indices    = None

    if   hasattr(old_group,"radii"):       radii      = old_group.radii
    else:                                  radii      = None          

    if parent is None: print("IMPORT GROUP: parent is NONE")

    new_group = group(labels, coord, indices, radii, parent)
    if debug > 0: print(f"IMPORT GROUP: created new_group with {new_group.formula}")
    
    ## Charges
    if hasattr(old_group,"totcharge") and hasattr(old_group,"atcharge"):
        new_group.set_charges(old_group.totcharge, old_group.atcharge)
    elif hasattr(old_group,"totcharge") and not hasattr(old_group,"atcharge"):
        new_group.set_charges(old_group.totcharge)

    ## Smiles
    if   hasattr(old_group,"Smiles"): new_group.smiles = old_group.Smiles
    elif hasattr(old_group,"smiles"): new_group.smiles = old_group.smiles     
    
    ## Atoms
    if not hasattr(old_group,"atoms"):  new_group.set_atoms()
    else: 
        atoms = []
        for at in old_group.atoms: 
            new_atom = import_atom(at, parent=new_group, debug=debug)
            atoms.append(new_atom)
        new_group.set_atoms(atoms)
            
    return new_group

################################
def import_atom(old_atom: object, parent: object=None, debug: int=0) -> object:
    assert hasattr(old_atom,"label") and (hasattr(old_atom,"coord") or hasattr(old_atom,"pos"))
    label     = old_atom.label
    
    if   hasattr(old_atom,"coord"):      coord      = old_atom.coord
    elif hasattr(old_atom,"pos"):        coord      = old_atom.pos

    if   hasattr(old_atom,"index"):      index      = old_atom.index
    elif hasattr(old_atom,"atlist"):     index      = old_atom.atlist
    else:                                index      = None

    if   hasattr(old_atom,"radii"):      radii      = old_atom.radii
    else:                                radii      = None          

    if   hasattr(old_atom,"block"):      block      = old_atom.block
    else:                                block      = elemdatabase.elementblock[label]    

    if parent is None: print("IMPORT ATOM: parent is NONE")
    
    if block == 'd' or block == 'f': 
        new_atom = metal(label, coord, index=index, radii=radii, parent=parent)
        if debug > 0: print(f"IMPORT ATOM: created new_metal {new_atom.label}")
    else:                            
        new_atom = atom(label, coord, index=index, radii=radii, parent=parent)
        if debug > 0: print(f"IMPORT ATOM: created new_atom {new_atom.label}")
    
    ## Charge. For some weird reason, charge is "charge" in metals, and "totcharge" in regular atoms
    if hasattr(old_atom,"charge"):       
        if type(old_atom.charge) == (int):      new_atom.set_charge(old_atom.charge)
        elif hasattr(old_atom,"totcharge"):     
            if type(old_atom.totcharge) == int: new_atom.set_charge(old_atom.totcharge)
            
    return new_atom
