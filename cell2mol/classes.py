import numpy as np
import sys
from cell2mol.connectivity import get_adjacency_types, get_element_count, labels2electrons, labels2formula, labels2ratio, get_adjmatrix, compare_atoms, compare_species
from cell2mol.connectivity import get_metal_idxs, split_species, get_radii 
from cell2mol.cell_reconstruct import *
from cell2mol.cell_operations import cart2frac, frac2cart_fromcellvec, frac2cart_fromparam
from cell2mol.yuri_formal_charge import *
from cell2mol.yuri_spin import *
from cell2mol.other import extract_from_list, compute_centroid, get_dist, get_angle
from cell2mol.elementdata import ElementData
elemdatabase = ElementData()
import pickle

################################
####  BASIS FOR CELL2MOL 2  ####
################################
class specie(object):
    def __init__(self, labels: list, coord: list, parent_indices: list=None, radii: list=None, parent: object=None) -> None:

       # Sanity Checks
        assert len(labels) == len(coord)
        if parent_indices is not None: assert len(labels) == len(parent_indices)
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

        if parent_indices is not None: self.parent_indices = parent_indices                     ## Species are often created from the parent object. Ligands are created from molecules.
        else:                          self.parent_indices = [*range(0,self.natoms,1)]          ## Indices might refer to those of the parent object. e.g. atoms 3, 4, 10 of parent molecule make one ligand
        self.indices   = [*range(0,self.natoms,1)]                                              ## Or, indices might be the atom ordering within a given specie. e.g. 1st, 2nd, 3rd atom of a specie.

        self.cov_factor   = 1.3
        self.metal_factor = 1.0

    ############
    def get_centroid(self):
        from cell2mol.other import compute_centroid 
        self.centroid = compute_centroid(self.coord)
        if hasattr(self,"frac_coord"): self.frac_centroid = compute_centroid(self.frac_coord) # If fractional coordinates exists, then also computes their centroid
        return self.centroid
    
    ############
    def get_fractional_coord(self, cell_vector=None) -> None:
        if cell_vector is None:
            if hasattr(self,"parent"):
                if hasattr(self.parent,"cellvec"): cell_vector = self.parent.cellvec.copy()
            else:     print("CLASS_SPECIE: get_fractional coordinates. Missing cell vector. Please provide it"); return None
        else:         
            self.frac_coord = cart2frac(self.coord, cell_vector)
        return self.frac_coord

    ############
    def get_atomic_numbers(self):
        if not hasattr(self,"atoms"): self.set_atoms()
        self.atnums = []
        for at in self.atoms:
            self.atnums.append(at.atnum)
        return self.atnums

    ############
    def set_element_count(self, heavy_only: bool=False):
        self.element_count = get_element_count(self.labels, heavy_only=heavy_only)
        return self.element_count

    ############
    def set_adj_types(self):
        if not hasattr(self,"adjmat"): self.get_adjmatrix()
        self.adj_types = get_adjacency_types(self.labels, self.adjmat)
        return self.adj_types

    ############
    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        # Stores the covalentradii factor and metal factor that were used to generate the molecule
        self.cov_factor   = cov_factor
        self.metal_factor = metal_factor

    ############
    def reset_charge(self):
        if hasattr(self,"totcharge"):      delattr(self,"totcharge")
        if hasattr(self,"atomic_charges"): delattr(self,"atomic")
        if hasattr(self,"smiles"):         delattr(self,"smiles")
        if hasattr(self,"rdkit_mol"):      delattr(self,"rdkit_mol")
        if hasattr(self,"poscharges"):     delattr(self,"poscharges") 
        for a in self.atoms:               
            a.reset_charge() 

    ############
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

    ############
    def set_atoms(self, atomlist=None, overwrite_parent: bool=False):
        ## If the atom objects already exist, and you want to set them in self from a different specie
        if atomlist is not None: 
            self.atoms = atomlist
            if overwrite_parent: 
                for at in self.atoms:
                    at.parent = self
        ## If not, that is, if the atom objects must be created from scratch....
        else: 
            self.atoms = []
            for idx, l in enumerate(self.labels):
                ## For each l in labels, create an atom class object.
                ismetal = elemdatabase.elementblock[l] == "d" or elemdatabase.elementblock[l] == "f"
                if ismetal: newatom = atom(l, self.coord[idx], parent=self, index=idx, radii=self.radii[idx])
                else:       newatom = metal(l, self.coord[idx], parent=self, index=idx, radii=self.radii[idx])
                self.atoms.append(newatom)
        ## Irrespectively of the choice, we update the adjacencies of the atom to match the new parent
        if not hasattr(self,"adjmat"):  self.get_adjmatrix()
        if not hasattr(self,"madjmat"): self.get_metal_adjmatrix()
        if self.adjmat is not None and self.madjmat is not None: 
            for at in self.atoms: 
                newatom.set_adjacencies(self.adjmat[idx],self.madjmat[idx],self.adjnum[idx],self.madjnum[idx])
    
    ############
    def get_adjmatrix(self):
        isgood, adjmat, adjnum = get_adjmatrix(self.labels, self.coord, self.cov_factor, self.radii)
        if isgood:
            self.adjmat = adjmat
            self.adjnum = adjnum
        else:
            self.adjmat = None
            self.adjnum = None
        return self.adjmat, self.adjnum

    ############
    def get_metal_adjmatrix(self):
        isgood, madjmat, madjnum = get_adjmatrix(self.labels, self.coord, self.cov_factor, self.radii, metal_only=True)
        if isgood:
            self.madjmat = madjmat
            self.madjnum = madjnum
        else:
            self.madjmat = None
            self.madjnum = None
        return self.madjmat, self.madjnum

    ############
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

    ############
    def print_xyz(self):
        print(self.natoms)
        print("")
        for idx, l in enumerate(self.labels):
            print("%s  %.6f  %.6f  %.6f" % (l, self.coord[idx][0], self.coord[idx][1], self.coord[idx][2]))

    ############
    ## This defines the sum operation between two species. To be implemented
    def __add__(self, other):
        if not isinstance(other, type(self)): return self
        return self

    ############
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
    def __init__(self, labels: list, coord: list, parent_indices: list=None, radii: list=None, parent: object=None) -> None:
        self.subtype = "molecule"
        specie.__init__(self, labels, coord, parent_indices, radii, parent)

    ############
    def get_spin(self):
        if self.iscomplex:  self.spin = assign_spin_complexes(self) 
        else :              self.spin = 1
                
    ############
    def reset_charge(self):
        specie.reset_charge(self)      ## First uses the generic specie class function for itself and its atoms
        if hasattr(self,"ligands"):    ## Second removes for the child classes 
            for lig in self.ligands:
                lig.reset_charge()
        if hasattr(self,"metals"):    
            for met in self.metals:
                met.reset_charge()

    ############
    def split_complex(self, debug: int=0):
        if not hasattr(self,"atoms"): self.set_atoms()
        if not self.iscomplex:        self.ligands = None; self.metals = None
        else: 
            self.ligands = []
            self.metals  = []
            # Identify Metals and the rest
            metal_idx = get_metal_idxs(self.labels, debug=debug)
            rest_idx  = list(idx for idx in self.indices if idx not in metal_idx) 
            if debug > 0: 
                print(f"SPLIT COMPLEX: found metals in indices {metal_idx}")
            #rest_idx  = list(idx for idx in self.indices if idx not in metal_idx)
            # Split the "rest" to obtain the ligands
            rest_labels  = extract_from_list(rest_idx, self.labels, dimension=1)
            rest_coord  = extract_from_list(rest_idx, self.coord, dimension=1)
            rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)
            rest_radii   = extract_from_list(rest_idx, self.radii, dimension=1)
            if hasattr(self,"frac_coord"): rest_frac = extract_from_list(rest_idx, self.frac_coord, dimension=1)
            if debug > 0: print(f"SPLIT COMPLEX: rest labels: {rest_labels}")
            if debug > 0: print(f"SPLIT COMPLEX: splitting species with {len(rest_labels)} atoms in block")
            if hasattr(self,"cov_factor"): blocklist = split_species(rest_labels, rest_coord, radii=rest_radii, indices=rest_indices, cov_factor=self.cov_factor, debug=debug)
            else:                          blocklist = split_species(rest_labels, rest_coord, radii=rest_radii, indices=rest_indices, debug=debug)
            
            ## Arranges Ligands
            for b in blocklist:
                if debug > 0: print(f"PREPARING BLOCK: {b}")
                lig_labels  = extract_from_list(b, rest_labels, dimension=1) 
                lig_coord   = extract_from_list(b, rest_coord, dimension=1) 
                lig_radii   = extract_from_list(b, rest_radii, dimension=1) 
                lig_atoms   = extract_from_list(b, self.atoms, dimension=1) 
                newligand   = ligand(lig_labels, lig_coord, indices=b, radii=lig_radii, parent=self)
                newligand.set_atoms(atomlist=lig_atoms, overwrite_parent=True)
                # If fractional coordinates are available...
                if hasattr(self,"frac_coord"): 
                    lig_frac_coord = extract_from_list(b, rest_frac, dimension=1)
                    newligand.set_fractional_coord(lig_frac_coord)
                    #newligand = check_metal_coordinating_atoms (newligand, debug=debug)  
                    # newligand.check_metal_coordinating_atoms(debug=debug) # TODO: Implement this function to ligand class
                self.ligands.append(newligand)

            ## Arranges Metals
            for m in metal_idx:
                newmetal    = metal(self.labels[m], self.coord[m], self.indices[m], self.radii[m], parent=self)
                self.metals.append(newmetal)                            
        return self.ligands, self.metals

    def get_hapticity(self, debug: int=0):
        if not hasattr(self,"ligands"): self.split_complex(debug=debug)
        self.is_haptic = False 
        self.haptic_type = []
        if self.iscomplex: 
            for lig in self.ligands:
                if not hasattr(lig,"is_haptic"):  lig.get_hapticity(debug=debug)
                if lig.is_haptic: self.is_haptic = True
                for entry in lig.haptic_type:
                    if entry not in self.haptic_type: self.haptic_type.append(entry)
        return self.haptic_type

###############
### LIGAND ####
###############
class ligand(specie):
    def __init__(self, labels: list, coord: list, parent_indices: list=None, radii: list=None, parent: object=None) -> None:
        self.subtype  = "ligand"
        specie.__init__(self, labels, coord, parent_indices, radii, parent)
        self.evaluate_as_nitrosyl()
        
    # def check_coordination (self, debug: int=0) -> list : 
    #     if not hasattr(self,"groups"): self.split_ligand(debug=debug)
    #     remove = []
    #     for group in self.groups :
    #         remove = group.check_coordination(debug=debug)        
    #     self.reset_adjacencies_lig_metalist_v2 (remove, debug=debug)    

    def get_connected_metals(self, debug: int=0):
        # metal.groups will be used for the calculation of the relative metal radius 
        # and define the coordination geometry of the metal /hapicitiy/ hapttype    
        self.metals = []
        for met in self.parent.metals:
            tmplabels = self.labels.copy()
            tmpcoord  = self.coord.copy()
            tmplabels.append(met.label)
            tmpcoord.append(met.coord)
            isgood, tmpadjmat, tmpadjnum = get_adjmatrix(tmplabels, tmpcoord, metal_only=True)
            if isgood and any(tmpadjnum) > 0: self.groups.append(group)
        return self.metals

    #######################################################
    def evaluate_as_nitrosyl(self, debug: int=0):
        self.is_nitrosyl = False
        if self.natoms == 2 and "N" in self.labels and "O" in self.labels: 
            self.is_nitrosyl = True 
            self.get_nitrosyl_geom(debug=debug)
        return self.is_nitrosyl 

    #######################################################
    def get_nitrosyl_geom(self: object, thres: float=160, debug: int=0) -> str:
        # Function that determines whether the M-N-O angle of a Nitrosyl "ligand" is "Bent" or "Linear"
        # Each case is treated differently
        #:return NO_type: "Linear" or "Bent"
        if not hasattr(self,"atoms"):  self.set_atoms()
        if not hasattr(self,"metals"): self.get_connected_metals()

        for idx, a in enumerate(self.atoms):
            if a.label == "N": central = a.coord.copy()
            if a.label == "O": extreme = a.coord.copy()

        dist = []
        for idx, met in enumerate(self.metals):
            metal = np.array(met.coord)
            dist.append(np.linalg.norm(central - metal))
        tgt = np.argmin(dist)
        metal = self.metals[tgt].coord.copy()
        if debug >= 2: print("LIGAND.GET_NITRO_GEOM: coords:", central, extreme, metal)

        vector1 = np.subtract(np.array(central), np.array(extreme))
        vector2 = np.subtract(np.array(central), np.array(metal))
        if debug >= 2: print("LIGAND.GET_NITRO_GEOM: NITRO Vectors:", vector1, vector2)

        angle = get_angle(vector1, vector2)
        if debug >= 2: print("NITRO ANGLE:", angle, np.degrees(angle))

        if np.degrees(angle) > float(thres): self.NO_type = "Linear"
        else:                                self.NO_type = "Bent"

        return self.NO_type 

    #######################################################
    def get_connected_idx(self, debug: int=0):
       self.connected_idx = [] 
       if not hasattr(self.parent,"madjmat"): madjmat, madjnum = self.parent.get_metal_adjmatrix()
       self.madjmat = extract_from_list(self.parent_indices, madjmat, dimension=2)    ## Here we use parent_indices because we're operating with a molecule variable (madjmat and madjnum)
       self.madjnum = extract_from_list(self.parent_indices, madjnum, dimension=1)
       for idx, con in enumerate(self.madjnum):
           if con > 0: self.connected_idx.append(self.indices[idx]) 
       return self.connected_idx 

    #######################################################
    def get_connected_atoms(self, debug: int=0):
        if not hasattr(self,"atoms"):         self.set_atoms()
        if not hasattr(self,"connected_idx"): self.get_connected_idx()
        self.connected_atoms = []
        for idx, at in enumerate(self.atoms):
            if idx in self.get_connected_idx: self.connected_atoms.append(at) 
        return self.connected_atoms

    #######################################################
    def get_denticity(self, debug: int=0):
        if not hasattr(self,"groups"):      self.split_ligand()
        if debug > 0: print(f"LIGAND.Get_denticity: checking connectivity of ligand {self.formula}")
        if debug > 0: print(f"LIGAND.Get_denticity: initial connectivity is {len(self.connected_idx)}")
        
        self.denticity = 0
        for g in self.grouplist:
            self.denticity += g.get_denticity()      ## A check is also performed at the group level
        return self.denticity 

    #######################################################
    def split_ligand(self, debug: int=0):
        self.groups = []
        # Identify Connected and Unconnected atoms (to the metal)
        if not hasattr(self,"connected_idx"): self.get_connected_idx()
        conn_idx = self.connected_idx
        rest_idx = list(idx for idx in self.indices if idx not in conn_idx)

        # Split the "ligand to obtain the groups
        conn_labels  = extract_from_list(conn_idx, self.labels, dimension=1)
        conn_coord  = extract_from_list(conn_idx, self.coord, dimension=1)
        conn_indices = extract_from_list(conn_idx, self.indices, dimension=1)

        if hasattr(self,"cov_factor"): blocklist = split_species(conn_labels, conn_coord, indices=conn_indices, cov_factor=self.cov_factor)
        else:                          blocklist = split_species(conn_labels, conn_coord, indices=conn_indices)
        ## Arranges Groups 
        for b in blocklist:
            gr_labels  = extract_from_list(b, self.labels, dimension=1)
            gr_coord   = extract_from_list(b, self.coord, dimension=1)
            gr_radii   = extract_from_list(b, self.radii, dimension=1)
            gr_atoms   = extract_from_list(b, self.atoms, dimension=1)
            newgroup   = group(gr_labels, gr_coord, b, gr_radii, parent=self)
            newgroup.set_atoms(atomlist=gr_atoms, overwrite_parent=False)
            newgroup.get_closest_metal()
            self.groups.append(newgroup)
            
    #######################################################
    def get_hapticity(self, debug: int=0):
        if not hasattr(self,"groups"): self.split_ligand(debug=debug)
        self.is_haptic = False 
        self.haptic_type = []
        for gr in self.groups:
            if not hasattr(gr,"is_haptic"): gr.get_hapticity(debug=debug)
            if gr.is_haptic: self.is_haptic = True; self.haptic_type = gr.haptic_type
            for entry in gr.haptic_type:
                if entry not in self.haptic_type: self.haptic_type.append(entry)
        return self.haptic_type

###############
#### GROUP ####
###############
class group(specie):
    def __init__(self, labels: list, coord: list, parent_indices: list=None, radii: list=None, parent: object=None) -> None:
        self.subtype = "group"
        specie.__init__(self, labels, coord, parent_indices, radii, parent)

    #######################################################
    def get_closest_metal(self, debug: int=0):
        apos = compute_centroid(np.array(self.coord))
        dist = []
        for met in self.parent.parent.metals:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        # finds the closest Metal Atom (tgt)
        self.closest_metal = self.parent.parent.metals[np.argmin(dist)]
        return self.closest_metal
    
    #######################################################
    def get_hapticity(self, debug: int=0):
        if not hasattr(self,"atoms"): self.set_atoms()
        self.is_haptic   = False ## old self.hapticity
        self.haptic_type = []    ## old self.hapttype
    
        numC  = self.labels.count("C")  # Carbon is the most common connected atom in ligands with hapticity
        numAs = self.labels.count("As") # I've seen one case of a Cp but with As instead of C (VENNEH, Fe dataset)
        numP  = self.labels.count("P")  
        numO  = self.labels.count("O")  # For h4-Enone
        numN  = self.labels.count("N")
    
        ## Carbon-based Haptic Ligands
        if   numC == 2:                   self.haptic_type = ["h2-Benzene", "h2-Butadiene", "h2-ethylene"]; self.is_haptic = True
        elif numC == 3 and numO == 0:     self.haptic_type = ["h3-Allyl", "h3-Cp"];                         self.is_haptic = True
        elif numC == 3 and numO == 1:     self.haptic_type = ["h4-Enone"];                                  self.is_haptic = True
        elif numC == 4:                   self.haptic_type = ["h4-Butadiene", "h4-Benzene"];                self.is_haptic = True
        elif numC == 5:                   self.haptic_type = ["h5-Cp"];                                     self.is_haptic = True
        elif numC == 6:                   self.haptic_type = ["h6-Benzene"];                                self.is_haptic = True
        elif numC == 7:                   self.haptic_type = ["h7-Cycloheptatrienyl"];                      self.is_haptic = True
        elif numC == 8:                   self.haptic_type = ["h8-Cyclooctatetraenyl"];                     self.is_haptic = True
        # Other less common types of haptic ligands
        elif numC == 0 and numAs == 5:    self.haptic_type = ["h5-AsCp"];                                   self.is_haptic = True
        elif numC == 0 and numP == 5:     self.haptic_type = ["h5-Pentaphosphole"];                         self.is_haptic = True
        return self.haptic_type 

    #######################################################
    def check_denticity(self, debug: int=0):
        from cell2mol.connectivity import add_atom
        if not hasattr(self,"is_haptic"): self.get_hapticity()
        if not hasattr(self,"atoms"):     self.set_atoms()
        for idx, a in enumerate(self.atoms):
            if debug > 0: print(f"GROUP.check_denticity: connectivity={a.mconnec} in atom idx={idx}, label={a.label}")
            isadded, newlab, newcoord = add_atom(self.parent.labels, self.parent.coord, self.parent_indices[idx], self.parent, self.parent.parent.metals, "H", debug=debug)
            if isadded:
                if debug > 0: print(f"GROUP.check_denticity: connectivity verified for atom {idx} with label {a.label}")
            else:
                if debug > 0: print(f"GROUP.check_denticity: corrected mconnec of atom {idx} with label {a.label}")
                a.mconnec = 0                                                                        # Corrects data of atom object in self
                self.parent.atoms[self.parent_indices[idx]].mconnec = 0                              # Corrects data of atom object in ligand class
                self.parent.madjnum[self.parent_indices[idx]] = 0                                    # Corrects data in metal_adjacency number of the ligand class
                self.parent.parent.atoms[self.parent.parent_indices[self.parent_indices[idx]]] = 0   # Corrects data of atom object in molecule class
                self.parent.parent.madjnum[self.parent.parent_indices[self.parent_indices[idx]]] = 0 # Corrects data in metal_adjacency number of the molecule class
        self.checked_denticity = True

    #######################################################
    def get_denticity(self, debug: int=0):
        if not hasattr(self,"checked_denticity"): self.check_denticity(debug=debug) 
        self.denticity = 0
        for a in self.atoms: 
            self.denticity += a.mconnec      
        return self.denticity

###############
### BOND ######
###############
class bond(object):
    def __init__(self, atom1: object, atom2: object, bond_order: int=1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.order = bond_order

###############
### ATOM ######
###############
class atom(object):
    def __init__(self, label: str, coord: list, parent_index: int=None, radii: float=None, parent: object=None, frac_coord: list=None) -> None:
        self.type          = "atom"
        self.version       = "0.1"
        self.label         = label
        self.coord         = coord
        self.atnum         = elemdatabase.elementnr[label]
        self.block         = elemdatabase.elementblock[label]

        if parent_index is not None:      self.parent_index = parent_index
        if radii is None:          self.radii = get_radii(label)
        else:                      self.radii = radii
        if parent is not None:     self.parent  = parent
        if parent is not None:     self.occurence = parent.get_occurrence(self)
        if frac_coord is not None: self.frac_coord = frac_coord

    #######################################################
    def add_bond(self,newbond: object):
        if not hasattr(self,"bonds"): self.bonds = []
        self.bonds.append(newbond)

    #######################################################
    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        self.cov_factor   = cov_factor
        self.metal_factor = metal_factor

    #######################################################
    def reset_charge(self) -> None:
        if hasattr(self,"charge"):    delattr(self,"charge")
        if hasattr(self,"poscharges"): delattr(self,"charge")

    #######################################################
    def set_charge(self, charge: int) -> None:
        self.charge = charge 

    #######################################################
    def set_adjacencies(self, adjmat, madjmat, connectivity: int, metal_connectivity: int=0):
        self.connec  = int(connectivity)
        self.mconnec = int(metal_connectivity)
        self.adjacency       = []
        self.metal_adjacency = []
        for idx, c in enumerate(adjmat):   ## The atom only receives one row of adjmat, so this is not a matrix anymore. Keep in mind that the idx are the indices of parent
            if c > 0: self.adjacency.append(idx)
        for idx, c in enumerate(madjmat):  ## The atom only receives one row of madjmat, so this is not a matrix anymore
            if c > 0: self.metal_adjacency.append(idx)

    #######################################################
    def get_closest_metal(self, metalist: list, debug: int=0):
        ## Here, the list of metal atoms must be provided
        apos = self.coord
        dist = []
        for met in metalist:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        self.closest_metal = metalist[np.argmin(dist)]
        return self.closest_metal

    #######################################################
    def information(self, cov_factor: float, metal_factor: float) -> None:
        self.cov_factor   = cov_factor
        self.metal_factor = metal_factor

    #######################################################
    def __repr__(self):
        to_print  = f'---------------------------------------------------\n'
        to_print +=  '   >>> Cell2mol ATOM Object >>>                    \n'
        to_print += f'---------------------------------------------------\n'
        to_print += f' Version               = {self.version}\n'
        to_print += f' Type                  = {self.type}\n'
        if hasattr(self,'subtype'): to_print += f' Sub-Type              = {self.subtype}\n'
        to_print += f' Label                 = {self.label}\n'
        to_print += f' Atomic Number         = {self.atnum}\n'
        to_print += f' Index in Parent       = {self.parent_index}\n'
        if hasattr(self,"occurrence"): to_print += f' Occurrence in Parent  = {self.occurrence}\n'
        if hasattr(self,"charge"):     to_print += f' Atom Charge           = {self.charge}\n'
        to_print += '----------------------------------------------------\n'
        return to_print

###############
#### METAL ####
###############
class metal(atom):
    def __init__(self, label: str, coord: list, parent_index: int=None, radii: float=None, parent: object=None, frac_coord: list=None) -> None:
        self.subtype      = "metal"
        atom.__init__(self, label, coord, parent_index=parent_index, radii=radii, parent=parent, frac_coord=frac_coord)

    #######################################################
    def count_valence_elec (self, m_ox: int):
        """ Count valence electrons for a given transition metal and metal oxidation state """
        v_elec = elemdatabase.valenceelectrons[self.label] - m_ox      
        if v_elec >= 0 :
            self.valence_elec = v_elec
        else :
            self.valence_elec = elemdatabase.elementgroup[self.label] - m_ox
        return self.valence_elec

    #######################################################
    def get_coord_sphere(self):
        if not hasattr(self,"parent"): return None
        if not hasattr(self.parent,"adjmat"): self.parent.get_adjmatrix()
        adjmat = self.parent.adjmat.copy()
        
        ## Cordination sphere defined as a collection of atoms
        self.coord_sphere = []
        for idx, at in enumerate(adjmat[self.parent_index]):
            if at >= 1:
                self.coord_sphere.append(self.parent.atoms[idx])
        return self.coord_sphere

    #######################################################
    def get_connected_groups(self, debug: int=0):
        # metal.groups will be used for the calculation of the relative metal radius 
        # and define the coordination geometry of the metal /hapicitiy/ hapttype    
        if not hasattr(self,"parent"): return None
        self.groups = []
        for group in self.parent.ligand.groups:
            tmplabels = self.label.copy()
            tmpcoord  = self.coord.copy()
            tmplabels.append(group.labels)
            tmpcoord.append(group.coord)
            isgood, tmpadjmat, tmpadjnum = get_adjmatrix(tmplabels, tmpcoord, metal_only=True)
            if isgood and any(tmpadjnum) > 0: self.groups.append(group)
        return self.groups

    #######################################################
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

    def get_spin(self):
        self.spin = assign_spin_metal(self)
    
    def get_poscharges(self, debug: int=0):
        self.poscharges = get_metal_poscharges(self, self.parent, debug=debug)

    ############
    def reset_charge(self):
        atom.reset_charge(self)     ## First uses the generic specie class function for itself and its atoms
        if hasattr(self,"poscharges"):   delattr(self,"poscharge") 

##############
#### CELL ####
##############
class cell(object):
    def __init__(self, name: str, labels: list, pos: list, cellvec: list, cellparam: list) -> None:
        self.version    = "0.1"
        self.type       = "cell"
        self.name       = name
        self.labels     = labels 
        self.coord      = pos
        self.cellvec    = cellvec
        self.cellparam  = cellparam
        self.natoms     = len(labels)

    #######################################################
    def get_fractional_coord(self):
        self.frac_coord = cart2frac(self.coord, self.cellv)
        return self.frac_coord

    #######################################################
    def check_missing_H(self, debug: int=0):
        from cell2mol.missingH import check_missingH
        Warning, ismissingH, Missing_H_in_C, Missing_H_in_CoordWater = check_missingH(self.refmoleclist, debug=debug)
        if ismissingH or Missing_H_in_C or Missing_H_in_CoordWater: self.has_missing_H = True
        else:                                                       self.has_missing_H = False
        return self.has_missing_H 

    #######################################################
    def get_reference_molecules(self, ref_labels: list, ref_fracs: list, cov_factor: float=1.3, metal_factor: float=1.0, debug: int=0):
        # In the info file, the reference molecules only have fractional coordinates. We convert them to cartesian
        ref_pos = frac2cart_fromparam(ref_fracs, self.cellparam)

        # Get reference molecules
        blocklist = split_species(ref_labels, ref_pos, cov_factor=cov_factor)
        self.refmoleclist = []
        for b in blocklist:
            mol_labels       = extract_from_list(b, ref_labels, dimension=1)
            mol_coord       = extract_from_list(b, ref_pos, dimension=1)
            mol_frac_coord  = extract_from_list(b, self.frac_coord, dimension=1)
            newmolec         = molecule(mol_labels, mol_coord)
            newmolec.set_fractional_coord(mol_frac_coord)
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
                        lig.get_denticity(debug=0)

        if isgood: self.has_isolated_H = False
        else:      self.has_isolated_H = True
        return self.refmoleclist

    #######################################################
    def get_moleclist(self, blocklist=None):
        if not hasattr(self,"labels") or not hasattr(self,"pos"): return None
        if len(self.labels) == 0 or len(self.pos) == 0:           return None
        cov_factor = 1.3

        if blocklist is None: blocklist = split_species(self.labels, self.pos, cov_factor=cov_factor)
        self.moleclist = []
        for b in blocklist:
            mol_labels  = extract_from_list(b, self.labels, dimension=1)
            mol_coord  = extract_from_list(b, self.coord, dimension=1)
            mol_radii   = extract_from_list(b, self.radii, dimension=1)
            newmolec    = molecule(mol_labels, mol_coord, b, mol_radii, parent=self)
            # If fractional coordinates are available...
            if hasattr(self,"frac_coord"): 
                mol_frac_coord  = extract_from_list(b, self.frac_coord, dimension=1)
                newmolec.set_fractional_coord(mol_frac_coord)
            # This must be below the frac_coord, so they are carried on to the ligands
            if newmolec.iscomplex: newmolec.split_complex()
            self.moleclist.append(newmolec)
        return self.moleclist
   
    #######################################################
    def arrange_cell_coord(self): 
        ## Updates the cell coordinates preserving the original atom ordering
        ## Do do so, it uses the variable parent_indices stored in each molecule
        self.coord = np.zeros((self.natoms,3))
        for mol in self.moleclist:
            for z in zip(mol.parent_indices, mol.coord):
                for i in range(0,3):
                    self.coord[z[0]][i] = z[1][i]
        self.coord = np.ndarray.tolist(self.coord)

    #######################################################
    def get_occurrence(self, substructure: object) -> int:
        occurrence = 0
        ## Molecules in Cell
        if hasattr(substructure,"subtype") and hasattr(self,"moleclist"): 
            if substructure.subtype == 'molecule':
                for m in self.moleclist:
                    issame = compare_species(substructure, m)
                    if issame: occurrence += 1
        return occurrence
    
    #######################################################
    def data_for_postproc(self, molecules: list, indices: list, options: list):
        self.pp_molecules = molecules
        self.pp_indices = indices
        self.pp_options = options

    #######################################################
    def reconstruct(self, cov_factor: float=None, metal_factor: float=None, debug: int=0):
        from cell2mol.cell_reconstruct import classify_fragments, fragments_reconstruct
        if not hasattr(self,"refmoleclist"): print("CELL.RECONSTRUCT. CELL missing list of reference molecules"); return None
        if not hasattr(self,"moleclist"): self.get_moleclist()
        blocklist    = self.moleclist.copy() # In principle, in moleclist now there are both fragments and molecules
        if cov_factor is None:   cov_factor   = self.refmoleclist[0].cov_factor
        if metal_factor is None: metal_factor = self.refmoleclist[0].metal_factor
        ## Classifies fragments
        for b in blocklist:
            if not hasattr(b,"frac_coord"):       b.get_fractional_coord(self.cellvec)

        moleclist, fraglist, Hlist = classify_fragments(blocklist, moleclist, self.refmoleclist, self.cellvec)

        ## Determines if Reconstruction is necessary
        if len(fraglist) > 0 or len(Hlist) > 0: self.is_fragmented = True
        else:                                   self.is_fragmented = False

        if not self.is_fragmented: return self.moleclist 
        self.moleclist, Warning = fragments_reconstruct(moleclist,fraglist,Hlist,self.refmoleclist,self.cellvec,cov_factor,metal_factor)
        if Warning:      self.is_fragmented = True;  self.error_reconstruction = True 
        else:            self.is_fragmented = False; self.error_reconstruction = False
        return self.moleclist
    
    def reset_charge_assignment(self, debug: int=0):
        if not hasattr(self,"moleclist"): return None
        for mol in self.moleclist:
            mol.reset_charge()
        
    #######################################################
    def assign_charges(self, debug: int=0) -> object:
        
        if not hasattr(self,"is_fragmented"): self.reconstruct(debug=debug)  
        if self.is_fragmented: return None # Stopping. self.is_fragmented must be false to determine the charges of the cell

        # Indentify unique chemical species
        unique_species, unique_indices = identify_unique_species(self.moleclist, debug=debug)
        if debug >= 1: print(f"{len(unique_species)} Species (Metal or Ligand or Molecules) to Characterize")
        self.speclist = [spec[1] for spec in unique_species] # spec is a list in which item 1 is the actual unique specie

        # Gets a preliminary list of possible charge states for each specie
        selected_charge_states, Warning = drive_get_poscharges(unique_species, debug=debug)
        if Warning: self.error_empty_poscharges = True; return None # Empty list of possible charges received for molecule or ligand. Stopping
        else:       self.error_empty_poscharges = False

        # Finds the charge_state that satisfies that the crystal must be neutral
        final_charge_distribution = balance_charge(unique_indices, unique_species, debug=debug)

        if len(final_charge_distribution) > 1:
            if debug >= 1: print("More than one Possible Distribution Found:", final_charge_distribution)
            self.error_multiple_distrib = True
            self.error_empty_distrib    = False
            pp_mols, pp_idx, pp_opt = prepare_unresolved(unique_indices, unique_species, final_charge_distribution, debug=debug)
            self.data_for_postproc(pp_mols, pp_idx, pp_opt)
            return None
        elif len(final_charge_distribution) == 0: # 
            if debug >= 1: print("No valid Distribution Found", final_charge_distribution)
            self.error_multiple_distrib = False
            self.error_empty_distrib    = True
            return None
        else: # Only one possible charge distribution -> getcharge for the repeated species
            if debug >= 1:
                print(f"\nFINAL Charge Distribution: {final_charge_distribution}\n")
                print("#########################################")
                print("Assigning Charges and Preparing Molecules")
                print("#########################################")
            self.moleclist, self.error_prepare_mols = prepare_mols(self.moleclist, unique_indices, unique_species, selected_charge_states, final_charge_distribution[0], debug=debug)

            if self.error_prepare_mols: return None # Error while preparing molecules
            
            for mol in self.moleclist:
                mol.build_bonds(debug=debug) ## TODO: Adapt build_bonds function to specie class
            
            return self.moleclist
            
    #######################################################
    def assign_spin(self, debug: int=0) -> object:
        if not hasattr(self,"is_preparemol"): self.determine_charge(debug=debug)  
        if self.is_preparemol: return None # Stopping. self.is_preparemol must be false to assign the spin
        for mol in self.moleclist:
            if mol.iscomplex:
                for metal in mol.metals:
                    metal.get_spin()
            mol.get_spin()
        return self.moleclist

    def assess_errors(self):
        ### This function might be called to print the possible errors found in the unit cell, during reconstruction, and charge/spin assignment
        return None

    def save(self, path):
        print(f"SAVING cell2mol CELL object to {path}")
        with open(path, "wb") as fil:
            pickle.dump(self,fil)
        
    #######################################################
    def __repr__(self):
        to_print  = f'---------------------------------------------------\n'
        to_print +=  '   >>> Cell2mol CELL Object >>>                    \n'
        to_print += f'---------------------------------------------------\n'
        to_print += f' Version               = {self.version}\n'
        to_print += f' Type                  = {self.type}\n'
        to_print += f' Name (Refcode)        = {self.name}\n'
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