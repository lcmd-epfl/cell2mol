import numpy as np
import sys
from cell2mol.connectivity import * 
from cell2mol.cell_reconstruct import *
from cell2mol.cell_operations import *
from cell.yuri_formal_charge import *
import pickle

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
#        if parent  is not None: self.occurence = parent.get_occurrence(self)  ## Commented because it wasn't working well sometimes.

        self.type      = "specie"
        self.version   = "0.1"
        self.labels    = labels
        self.coord     = coord
        self.formula   = labels2formula(labels)
        self.eleccount = labels2electrons(labels)   ### Assuming neutral specie (so basically this is the sum of atomic numbers)
        self.natoms    = len(labels)
        self.iscomplex = any((elemdatabase.elementblock[l] == "d") or (elemdatabase.elementblock[l] == "f") for l in self.labels)

        if indices is not None: self.indices = indices
        else:                   self.indices = [*range(0,self.natoms,1)]

        self.cov_factor   = 1.3
        self.metal_factor = 1.0

    def get_centroid(self):
        from cell2mol.other import compute_centroid 
        self.centroid = compute_centroid(self.coord)
        if hasattr(self,"frac_coord"): self.frac_centroid = compute_centroid(self.frac_coord) # If fractional coordinates exists, then also computes their centroid
        return self.centroid
    
    def get_fractional_coord(self, cell_vector=None) -> None:
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

    def get_metal_adjmatrix(self):
        isgood, madjmat, madjnum = get_adjmatrix(self.labels, self.coord, self.cov_factor, self.radii, metal_only=True)
        if isgood:
            self.madjmat = madjmat
            self.madjnum = madjnum
        else:
            self.madjmat = None
            self.madjnum = None
        return self.madjmat, self.madjnum

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
        to_print +=  '   >>> Cell2mol SPECIE Object >>>                  \n'
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
                    #newligand = check_metal_coordinating_atoms (newligand, debug=debug)  
                    # newligand.check_metal_coordinating_atoms(debug=debug) # TODO: Implement this function to ligand class
                self.ligands.append(newligand)

            ## Arranges Metals
            for m in metal_idx:
                newmetal    = metal(self.labels[m], self.coord[m], self.indices[m], self.radii[m], parent=self)
                self.metals.append(newmetal)                            

        return self.ligands, self.metals
        
###############
### LIGAND ####
###############
class ligand(specie):
    def __init__(self, labels: list, coord: list, indices: list=None, radii: list=None, parent: object=None) -> None:
        self.subtype  = "ligand"
        specie.__init__(self, labels, coord, indices, radii, parent)

    # def check_coordination (self, debug: int=0) -> list : 
    #     if not hasattr(self,"groups"): self.split_ligand(debug=debug)
    #     remove = []
    #     for group in self.groups :
    #         remove = group.check_coordination(debug=debug)        
    #     self.reset_adjacencies_lig_metalist_v2 (remove, debug=debug)    

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
            newgroup.get_closest_metal()
            self.groups.append(newgroup)
            
        return self.groups

###############
#### GROUP ####
###############
class group(specie):
    def __init__(self, labels: list, coord: list, indices: list=None, radii: list=None, parent: object=None) -> None:
        self.subtype = "group"
        specie.__init__(self, labels, coord, indices, radii, parent)
    
    def set_hapticity(self, hapttype):
        self.hapticity = True 
        self.hapttype  = hapttype 

    def get_closest_metal(self, debug: int=0):
        apos = compute_centroid(np.array(self.coord))
        dist = []
        for met in self.parent.metalist:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        # finds the closest Metal Atom (tgt)
        self.closest_metal = self.parent.metalist[np.argmin(dist)]
        return self.closest_metal


###############
### ATOM ######
###############
class atom(object):
    def __init__(self, label: str, coord: list, index: int=None, radii: float=None, parent: object=None, frac_coord: list=None) -> None:
        self.type          = "atom"
        self.version       = "0.1"
        self.label         = label
        self.coord         = coord
        self.atnum         = elemdatabase.elementnr[label]
        self.block         = elemdatabase.elementblock[label]

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
        to_print +=  '   >>> Cell2mol ATOM Object >>>                    \n'
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

    def count_valence_elec (self, m_ox: int):
        """ Count valence electrons for a given transition metal and metal oxidation state """
        v_elec = elemdatabase.valenceelectrons[self.label] - m_ox      
        if v_elec >= 0 :
            self.valence_elec = v_elec
        else :
            self.valence_elec = elemdatabase.elementgroup[self.label] - m_ox
        return self.valence_elec

    def get_coord_sphere(self):
        if not hasattr(self,"parent"): return None
        if not hasattr(self.parent,"adjmat"): self.parent.get_adjmatrix()
        adjmat = self.parent.adjmat.copy()
        
        ## Cordination sphere defined as a collection of atoms
        self.coord_sphere = []
        for idx, at in enumerate(adjmat[self.index]):
            if at >= 1:
                self.coord_sphere.append(self.parent.atoms[idx])
        return self.coord_sphere

    def get_connected_groups(self, debug: int=0):
        if not hasattr(self,"parent"): return None
        self.groups = []
        for group in self.parent.ligand.groups:
            if not hasattr(group,"closest_metal"): group.get_closest_metal(debug=debug)
            issame = compare_atoms(group.closest_metal, self)
            if issame: self.groups.append(group)
        return self.groups
        # metal.groups will be used for the calculation of the relative metal radius 
        # and define the coordination geometry of the metal /hapicitiy/ hapttype    

    def get_relative_metal_radius(self, debug: int=0):
        if not hasattr(self,"groups"): self.get_connected_groups(debug=debug)
        
        diff_list = []
        for group in self.groups:
            if group.hapticity == False :
                for atom in group.atoms:
                    diff = round(get_dist(self.coord, atom.coord) - elemdatabase.CovalentRadius3[atom.label], 3)
                    diff_list.append(diff)
            else :
                haptic_center_label = "C"
                haptic_center_coord = compute_centroid([atom.coord for atom in group.atoms]) 
                diff = round(get_dist(self.coord, haptic_center_coord) - elemdatabase.CovalentRadius3[haptic_center_label], 3)
                diff_list.append(diff)     
        average = round(np.average(diff_list), 3)    
        self.rel_metal_radius = round(average/elemdatabase.CovalentRadius3[self.label], 3)
        
        return self.rel_metal_radius

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

    def get_fractional_coord(self):
        self.frac_coord = cart2frac(self.coord, self.cellv)
        return self.frac_coord

    def check_missing_H(self, debug: int=0):
        from cell2mol.missingH import check_missingH
        Warning, ismissingH, Missing_H_in_C, Missing_H_in_CoordWater = check_missingH(self.refmoleclist, debug=debug)
        if ismissingH or Missing_H_in_C or Missing_H_in_CoordWater: self.has_missing_H = True
        else:                                                       self.has_missing_H = False
        return self.has_missing_H 

    def get_references_molecules(self, ref_labels: list, ref_fracs: list, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
        # In the info file, the reference molecules only have fractional coordinates. We convert them to cartesian
        ref_pos = frac2cart_fromparam(ref_fracs, self.cellparam)

        # Get reference molecules
        blocklist = split_species(ref_labels, ref_pos, cov_factor=cov_factor)
        self.refmoleclist = []
        for b in blocklist:
            mol_labels       = extract_from_list(b, ref_labels, dimension=1)
            mol_coords       = extract_from_list(b, ref_pos, dimension=1)
            mol_frac_coords  = extract_from_list(b, self.frac_coord, dimension=1)
            newmolec         = molecule(mol_labels, mol_coords)
            newmolec.set_fractional_coord(mol_frac_coords)
            # This must be below the frac_coord, so they are carried on to the ligands
            if newmolec.iscomplex: 
                newmolec.split_complex()
            self.refmoleclist.append(newmolec)
            
        # Checks for isolated atoms, and retrieves warning if there is any. Except if it is H, halogen (group 17) or alkalyne (group 2)
        isgood = True 
        for ref in self.refmoleclist:
            if ref.natoms == 1:
                label = ref.atoms[0].label
                group = elemdatabase.elementgroup[label]
                if (group == 2 or group == 17) and label != "H": pass 
                else:
                    isgood = False
                    if debug >= 2: print(f"GETREFS: found ref molecule with only one atom {ref.labels}")

        # If all good, then works with the reference molecules
        if isgood:
            for ref in self.refmoleclist:
                if ref.iscomplex: 
                    ref.check_hapticity(debug=debug)                          ### Former "get_hapticity(ref)" function
                    ref.get_coordination_geometry(debug=debug)                ### Former "get_coordination_Geometry(ref)" function 
                    for lig in ref.ligandlist:
                        verify_connectivity(lig, ref, debug=0)
                        if debug >= 2: print(f"Verifying Connectivity for Ligand {lig} in Reference {ref}") 

        if isgood: self.has_isolated_H = False
        else:      self.has_isolated_H = True
        return self.refmoleclist

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
    
    def data_for_postproc(self, molecules: list, indices: list, options: list):
        self.pp_molecules = molecules
        self.pp_indices = indices
        self.pp_options = options

    def reconstruct(self, debug: int=0):
        if not hasattr(self,"refmoleclist"): print("CLASS_CELL missing list of reference molecules"); return None
        if not hasattr(self,"moleclist"): self.get_moleclist()
        blocklist    = self.moleclist.copy() # In principle, in moleclist now there are both fragments and molecules
        cov_factor   = self.refmoleclist[0].cov_factor
        metal_factor = self.refmoleclist[0].metal_factor
        ## Prepares Blocks
        for b in blocklist:
            if not hasattr(b,"frac_coord"):       b.get_fractional_coord(cellvec)
            if not hasattr(b,"centroid"):         b.get_centroid()
            if not hasattr(b,"element_count"):    b.set_element_count()
            if not hasattr(b,"numH"):             b.numH = b.set_element_count()[4]
        ## Prepares Reference Molecules
        for ref in self.refmoleclist:
            if not hasattr(ref,"element_count"):  ref.set_element_count()
            if not hasattr(ref,"numH"):           ref.numH = ref.set_element_count()[4]
        ## Classifies fragments
        moleclist, fraglist, Hlist = classify_fragments(blocklist, moleclist, self.refmoleclist, self.cellvec)
        ## Determines if Reconstruction is necessary
        if len(fraglist) > 0 or len(Hlist) > 0: self.is_fragmented = True
        else:                                   self.is_fragmented = False

        if not self.is_fragmented: return self.moleclist 
        moleclist, finalmols, Warning = fragments_reconstruct(moleclist,fraglist,Hlist,refmoleclist,self.cellvec,cov_factor,metal_factor)
        moleclist.extend(finalmols)
        self.moleclist = moleclist
        return self.moleclist
    
    def determine_charge(self, debug: int=0) -> object:
        
        # self.is_fragmented:         "Warning 5! Errors in cell reconstruction"
        # self.is_empty_poscharges:   "Warning 6! Empty list of possible charges received for molecule or ligand"
        # self.is_multiple_distrib:   "Warning 7! More than one valid possible charge distribution found"
        # self.is_empty_distrib:      "Warning 8! No valid possible charge distribution found"
        # self.is_preparemol:         "Warning 9! Error while preparing molecules"

        if not hasattr(self,"is_fragmented"): self.reconstruct(debug=debug)  
        if self.is_fragmented: return None # Stopping. self.isfragmented must be false to determine the charges of the cell

        # Indentify unique chemical species
        molec_indices, ligand_indices, unique_indices, unique_species = classify_mols(self.moleclist, debug=debug)

        if len(unique_species) == 0:
            if debug >= 1: print("Empty list of species found. Stopping")
            return None
            # sys.exit()
        else :
            if debug >= 1: print(f"{len(unique_species)} Species (Ligand or Molecules) to Characterize")
            self.speclist = [spec[1] for spec in unique_species] # spec is a list in which item 1 is the actual unique specie

        # drive_get_poscharges adds posible charges to the metal, ligand, and molecule objects of all species in the unit cell
        # also, it retrieves "Selected_charge_states", 
        # which is a tuple with [the actual charge state, and the protonation it belongs to] for all objects except metals        
        selected_charge_states, self.is_empty_poscharges = drive_get_poscharges(unique_species, debug=debug)
        if self.is_empty_poscharges: return None # Empty list of possible charges received for molecule or ligand. Stopping
        else : # Find possible charge distribution(s)
            final_charge_distribution = balance_charge(unique_indices, unique_species, debug=debug)

            if len(final_charge_distribution) > 1:
                if debug >= 1: print("More than one Possible Distribution Found:", final_charge_distribution)
                self.is_multiple_distrib = True
                self.is_empty_distrib = False
                pp_mols, pp_idx, pp_opt = prepare_unresolved(unique_indices, unique_species, final_charge_distribution, debug=debug)
                self.data_for_postproc(pp_mols, pp_idx, pp_opt)
                return None
            elif len(final_charge_distribution) == 0: # 
                if debug >= 1: print("No valid Distribution Found", final_charge_distribution)
                self.is_multiple_distrib = False
                self.is_empty_distrib = True
                return None
            else: # Only one possible charge distribution -> getcharge for the repeated species
                if debug >= 1:
                    print(f"\nFINAL Charge Distribution: {final_charge_distribution}\n")
                    print("#########################################")
                    print("Assigning Charges and Preparing Molecules")
                    print("#########################################")
                self.moleclist, self.is_preparemol = prepare_mols(self.moleclist, unique_indices, unique_species, selected_charge_states, final_charge_distribution[0], debug=debug)

                if self.is_preparemol: return None # Error while preparing molecules
                
                for mol in self.moleclist:
                    mol.build_bonds(debug=debug) ## TODO: Adapt build_bonds function to specie class
                
                return self.moleclist

    def assign_spin (self, debug: int=0) -> object:
        if not hasattr(self,"is_preparemol"): self.determine_charge(debug=debug)  
        if self.is_preparemol: return None # Stopping. self.is_preparemol must be false to assign the spin

        for mol in self.moleclist:
            tot_elec = mol.eleccount - mol.totcharge
            if mol.iscomplex == False:                              # Other molecules
                if tot_elec % 2 == 0:   mol.magnetism(1) 
                else:                   mol.magnetism(2)             
            else :
                if len(mol.metals) > 1 :                            # Bi- & Poly-metallic complexes
                    # Bi metllic complexes if one is singlet and the spin of the other will be determined
                    # TODO : make a function to determine the spin of these kinds of complexes
                    if tot_elec % 2 == 0:   mol.magnetism(1) 
                    else:                   mol.magnetism(2)    
                else :                                              # Mono-metallic complexes
                    met = mol.metals[0]
                    valence_elec = met.count_valence_elec (met.totcharge)
                    
                    if elemdatabase.elementperiod[met.label] == 4:  # 3d transition metals
                        if valence_elec in [0, 1, 9, 10]:
                            if tot_elec % 2 == 0:   mol.magnetism(1) 
                            else:                   mol.magnetism(2)

                        elif valence_elec in [2, 3] and met.hapticity == False :
                            if tot_elec % 2 == 0:   mol.magnetism(3) 
                            else:                   mol.magnetism(4)

                        elif valence_elec in [4, 5, 6, 7, 8] or (valence_elec in [2, 3] and met.hapticity == True) :
                            # Predict spin multiplicity of metal based on Random forest model
                            # TODO: Implement the Random Forest model
                            feature = generate_feature_vector (met)
                            path_rf = os.path.join( os.path.abspath(os.path.dirname(__file__)), "total_spin_3131.pkl")
                            rf = pickle.load(open(path_rf, 'rb'))
                            predictions = rf.predict(feature)
                            spin_rf = predictions[0]
                            mol.magnetism(spin_rf)
                        else :
                            return None
                            print("Error: valence_elec is not in the range of 0-10", valence_elec)
                    else :                                          # 4d and 5d transition metals
                        if tot_elec % 2 == 0:   mol.magnetism(1) 
                        else:                   mol.magnetism(2)

        return self.moleclist

    def __repr__(self):
        to_print  = f'---------------------------------------------------\n'
        to_print +=  '   >>> Cell2mol CELL Object >>>                    \n'
        to_print += f'---------------------------------------------------\n'
        to_print += f' Version               = {self.version}\n'
        to_print += f' Type                  = {self.type}\n'
        to_print += f' Refcode               = {self.refcode}\n'
        to_print += f' Num Atoms             = {self.natoms}\n'
        to_print += f' Cell Parameters a:c   = {self.cellparam[0:3]}\n'
        to_print += f' Cell Parameters al:ga = {self.cellparam[3:6]}\n'
        #to_print += f' New variable          = {self.newvar}\n'
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