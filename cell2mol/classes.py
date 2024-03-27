import numpy as np
from cell2mol.connectivity import get_adjacency_types, get_element_count, labels2electrons, labels2formula, get_adjmatrix
from cell2mol.connectivity import get_metal_idxs, split_species, get_radii
from cell2mol.connectivity import compare_atoms, compare_species, compare_metals
from cell2mol.cell_reconstruction import classify_fragments, fragments_reconstruct
from cell2mol.cell_operations import cart2frac, frac2cart_fromparam
from cell2mol.charge_assignment import get_protonation_states_specie, get_possible_charge_state, get_metal_poscharges
from cell2mol.charge_assignment import balance_charge, prepare_unresolved, prepare_mols, correct_smiles_ligand
from cell2mol.spin import assign_spin_metal, assign_spin_complexes
from cell2mol.other import extract_from_list, compute_centroid, get_dist, get_angle
from cell2mol.elementdata import ElementData
from cell2mol.coordination_sphere import coordination_correction_for_haptic, coordination_correction_for_nonhaptic
elemdatabase = ElementData()
import pickle

##################################
####  CLASSES FOR CELL2MOL 2  ####
##################################
class specie(object):
    def __init__(self, labels: list, coord: list, radii: list=None) -> None:

       # Sanity Checks
        assert len(labels) == len(coord)

        # Optional Information
        if radii   is not None: self.radii   = radii
        else:                   self.radii   = get_radii(labels)

        self.type              = "specie"
        self.version           = "0.1"
        self.labels            = labels
        self.coord             = coord
        self.formula           = labels2formula(labels)
        self.eleccount         = labels2electrons(labels)   ### Assuming neutral specie (so basically this is the sum of atomic numbers)
        self.natoms            = len(labels)
        self.iscomplex         = any((elemdatabase.elementblock[l] == "d") or (elemdatabase.elementblock[l] == "f") for l in self.labels)
        self.parents           = []
        self.parents_indices   = []
        self.cov_factor        = 1.3
        self.metal_factor      = 1.0
        self.indices           = [*range(0,self.natoms,1)]  ## Indices might be the atom ordering within a given specie. e.g. 1st, 2nd, 3rd atom of a specie.

    ############
    def add_parent(self, parent: object, indices: list, overwrite: bool=True):
        ## associates a parent specie to self. The atom indices of self in parent are given in "indices"
        ## if parent of the same subtype already in self.parent then it is overwritten
        ## this is to avoid having a substructure (e.g. a ligand) in more than one superstructure (e.g. a molecule) 

        # 1st-evaluates parent
        append = True
        for idx, p in enumerate(self.parents):
            if p.subtype == parent.subtype:
                if overwrite: 
                    self.parents[idx]         = parent
                    self.parents_indices[idx] = indices
                append = False
        if append: 
            self.parents.append(parent)
            self.parents_indices.append(indices)

        # 2nd-evaluates parents of parent
        if hasattr(parent,"parents"):
            for jdx, p2 in enumerate(parent.parents):
                append = True
                for idx, p in enumerate(self.parents):
                    if p.subtype == p2.subtype:
                        if overwrite: 
                            self.parents[idx]         = p2
                            self.parents_indices[idx] = parent.get_parent_indices(p2.subtype)
                        append = False
                if append: 
                    self.parents.append(p2)
                    self.parents_indices.append(parent.get_parent_indices(p2.subtype))

    ############
    def check_parent(self, subtype: str):
        ## checks if parent of a given subtype exists
        for p in self.parents:
            if p.subtype == subtype: return True
        return False

    ############
    def get_parent(self, subtype: str):
        ## retrieves parent of a given subtype 
        for p in self.parents:
            if p.subtype == subtype: return p
        return None

    ############
    def get_parent_indices(self, subtype: str):
        ## retrieves parent of a given subtype 
        for idx, p in enumerate(self.parents):
            if p.subtype == subtype: return self.parents_indices[idx]
        return None

    ############
    def get_centroid(self):
        from cell2mol.other import compute_centroid 
        self.centroid = compute_centroid(np.array(self.coord))
        if hasattr(self,"frac_coord"): self.frac_centroid = compute_centroid(np.array(self.frac_coord)) # If fractional coordinates exists, then also computes their centroid
        return self.centroid
    
    ############
    def set_fractional_coord(self, frac_coord: list, debug: int=0) -> None:
        assert len(frac_coord) == len(self.coord)
        self.frac_coord = frac_coord 

    ############
    def get_fractional_coord(self, cell_vector=None, debug: int=0) -> None:
        if cell_vector is None:
            if self.check_parent("cell"):
                cell = self.get_parent("cell")
                if hasattr(cell,"cellvec"): cell_vector = cell.cellvec.copy()
            else:     print("SPECIE.GET_FRACTIONAL_COORD: get_fractional coordinates. Missing cell vector. Please provide it"); return None
        if debug > 1: print(f"SPECIE.GET_FRACTIONAL_COORD: Using cell_vector:{cell_vector}")
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
    def set_charges(self, totcharge: int=None, atomic_charges: list=None, smiles: str=None, rdkit_obj: object=None) -> None:
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
        if smiles is not None:      self.smiles = smiles
        if rdkit_obj is not None:   self.rdkit_obj = rdkit_obj

    ############
    def set_atoms(self, atomlist=None, create_adjacencies: bool=False, debug: int=0):
        debug = 0
        ## If the atom objects already exist, and you want to set them in self from a different specie
        if atomlist is not None: 
            if debug > 0: print(f"SPECIE.SET_ATOMS: received {atomlist=}")
            self.atoms = atomlist.copy()
            for idx, at in enumerate(self.atoms):
                at.add_parent(self, index=idx)

        ## If not, that is, if the atom objects must be created from scratch....
        else: 
            self.atoms = []
            for idx, l in enumerate(self.labels):
                if debug > 0: print(f"SPECIE.SET_ATOMS: creating atom for label {l}")
                ## For each l in labels, create an atom class object.
                ismetal = elemdatabase.elementblock[l] == "d" or elemdatabase.elementblock[l] == "f"
                if debug > 0: print(f"SPECIE.SET_ATOMS: {ismetal=}")
                if ismetal: newatom = metal(l, self.coord[idx], radii=self.radii[idx])
                else:       newatom = atom(l, self.coord[idx], radii=self.radii[idx])
                if debug > 0: print(f"SPECIE.SET_ATOMS: added atom to specie: {self.formula}")
                newatom.add_parent(self,index=idx)
                self.atoms.append(newatom)
        
        if create_adjacencies:
            if not hasattr(self,"adjmat"):  self.get_adjmatrix()
            if not hasattr(self,"madjmat"): self.get_metal_adjmatrix()
            if self.adjmat is not None and self.madjmat is not None: 
                for idx, at in enumerate(self.atoms): 
                    at.set_adjacencies(self.adjmat[idx],self.madjmat[idx],self.adjnum[idx],self.madjnum[idx])
    

    #######################################################
    def inherit_adjmatrix(self, parent_subtype: str, debug: int=0):
        exists  = self.check_parent(parent_subtype)
        if not exists: 
            print(f"SPECIE.INHERIT. {parent_subtype=} does not exist")
            return None
        parent  = self.get_parent(parent_subtype)
        indices = self.get_parent_indices(parent_subtype)
        if not hasattr(parent,"madjnum"): 
            print(f"SPECIE.INHERIT. {parent_subtype=} does not have madjnum")
            return None 
        #print(f"SPECIE.INHERIT. found self in parent with {indices=}")
        #print(f"SPECIE.INHERIT: parent data:\n{parent.labels=}\n{parent.madjmat=}\n{parent.madjnum=}\n{parent.adjmat=}\n{parent.adjnum=}")
        self.madjmat = np.stack(extract_from_list(indices, parent.madjmat, dimension=2), axis=0)
        self.madjnum = np.stack(extract_from_list(indices, parent.madjnum, dimension=1), axis=0)
        self.adjmat  = np.stack(extract_from_list(indices, parent.adjmat, dimension=2), axis=0)
        self.adjnum  = np.stack(extract_from_list(indices, parent.adjnum, dimension=1), axis=0)
        #print(f"SPECIE.INHERIT: self data:\n{self.labels=}\n{self.madjmat=}\n{self.madjnum=}\n{self.adjmat=}\n{self.adjnum=}")

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
    def get_protonation_states(self, debug: int=0):
        # !!! WARNING. FUNCTION defined at the "specie" level, but will only do something for ligands and organic (iscomplex == False) molecules
        if self.subtype == "group": 
            if not hasattr(self,"is_haptic"): self.get_hapticity()
            self.protonation_states = None
        elif self.subtype == "ligand" :
            # if not hasattr(self,"groups"): self.split_ligand()
            if not hasattr(self, "is_haptic"): self.get_hapticity()
            if not hasattr(self, "denticity"): self.get_denticity()
            self.protonation_states = get_protonation_states_specie(self, debug=debug)
        else : 
            if not hasattr(self,"is_haptic"): self.get_hapticity()
            self.protonation_states = get_protonation_states_specie(self, debug=debug)
        return self.protonation_states

    ############
    def get_possible_cs(self, debug: int=0):
        ## Arranges a list of possible charge_states associated with this species, 
        ## which is later managed at the cell level to determine the good one
        if not hasattr(self,"protonation_states"): self.get_protonation_states(debug=debug)
        if self.protonation_states is not None:    self.possible_cs = get_possible_charge_state(self, debug=debug)  
        return self.possible_cs
    
    ############
    def create_bonds(self, debug: int=0):
        if not hasattr(self,"rdkit_mol"): self.parent.assign_charges()
        for idx, atom in enumerate(self.atoms):
            # Security Check. Confirms that the labels are the same
            #if debug >= 2: print("BUILD BONDS: atom", idx, a.label)
            rdkitatom = self.rdkit_mol.GetAtomWithIdx(idx)
            tmp = rdkitatom.GetSymbol()
            if atom.label != tmp: print("Error in Create Bonds. Atom labels do not coincide. GMOL vs. MOL:", atom.label, tmp)
            else:
                # First part. Creates bond information
                for b in rdkitatom.GetBonds():
                    bond_startatom = b.GetBeginAtomIdx()
                    bond_endatom   = b.GetEndAtomIdx()
                    bond_order     = b.GetBondTypeAsDouble()

                    if (self.subtype == "ligand") and (bond_startatom >= self.natoms or bond_endatom >= self.natoms):
                        continue
                    else:
                        if self.atoms[bond_endatom].label != self.rdkit_mol.GetAtomWithIdx(bond_endatom).GetSymbol():
                            if debug >= 1: 
                                print("Error with Bond EndAtom", self.atoms[bond_endatom].label, self.rdkit_mol.GetAtomWithIdx(bond_endatom).GetSymbol())
                        else:
                            if bond_endatom == idx:
                                start = bond_endatom
                                end   = bond_startatom
                            elif bond_startatom == idx:
                                start = bond_startatom
                                end   = bond_endatom

                            ## This has changed. Now there is a bond object, and we send the atom objects, not only the index
                            new_bond = bond(self.atoms[start], self.atoms[end], bond_order) 
                        atom.add_bond(new_bond)

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
    def __repr__(self, indirect: bool=False):
        to_print = ""
        if not indirect: to_print  += f'------------- Cell2mol SPECIE Object --------------\n'
        to_print += f' Version                      = {self.version}\n'
        to_print += f' Type                         = {self.type}\n'
        if hasattr(self,'subtype'): to_print += f' Sub-Type                     = {self.subtype}\n'
        to_print += f' Number of Atoms              = {self.natoms}\n'
        to_print += f' Formula                      = {self.formula}\n'
        if hasattr(self,"adjmat"):     to_print += f' Has Adjacency Matrix         = YES\n'
        else:                          to_print += f' Has Adjacency Matrix         = NO \n'
        if hasattr(self,"totcharge"):  to_print += f' Total Charge                 = {self.totcharge}\n'
        if hasattr(self,"spin"):       to_print += f' Spin                         = {self.spin}\n'
        if hasattr(self,"smiles"):     to_print += f' Smiles                       = {self.smiles}\n'
        if hasattr(self,"origin"):     to_print += f' Origin                       = {self.origin}\n'
        if not indirect: to_print += '---------------------------------------------------\n'
        return to_print

###############
### MOLECULE ##
###############
class molecule(specie):
    def __init__(self, labels: list, coord: list, radii: list=None) -> None:
        self.subtype = "molecule"
        specie.__init__(self, labels, coord, radii)

    def __repr__(self):
        to_print = ""
        to_print += f'------------- Cell2mol MOLECULE Object --------------\n'
        to_print += specie.__repr__(self, indirect=True)
        if hasattr(self,"ligands"):  
            if self.ligands is not None: to_print += f' Number of Ligands            = {len(self.ligands)}\n'
        if hasattr(self,"metals"):   
            if self.metals is not None:  to_print += f' Number of Metals             = {len(self.metals)}\n'
        to_print += '---------------------------------------------------\n'
        return to_print

    ############
    def get_spin(self):
        if self.iscomplex:  self.spin = assign_spin_complexes(self) 
        else :              self.spin = 1
        return self.spin
                
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
    def split_complex(self, debug: int=2):
        if not hasattr(self,"atoms"): self.set_atoms()
        if not self.iscomplex:        self.ligands = None; self.metals = None
        else: 
            self.ligands = []
            self.metals  = []
            # Identify Metals and the rest
            metal_idx = list([self.indices[idx] for idx in get_metal_idxs(self.labels, debug=debug)])
            rest_idx  = list(idx for idx in self.indices if idx not in metal_idx) 
            if debug > 0 :  print(f"MOLECULE.SPLIT COMPLEX: labels={self.labels}")
            if debug > 0 :  print(f"MOLECULE.SPLIT COMPLEX: metal_idx={metal_idx}")
            if debug > 0 :  print(f"MOLECULE.SPLIT COMPLEX: rest_idx={rest_idx}")

            # Split the "rest" to obtain the ligands
            rest_labels  = extract_from_list(rest_idx, self.labels, dimension=1)
            rest_coord   = extract_from_list(rest_idx, self.coord, dimension=1)
            rest_indices = extract_from_list(rest_idx, self.indices, dimension=1)
            rest_radii   = extract_from_list(rest_idx, self.radii, dimension=1)
            rest_atoms   = extract_from_list(rest_idx, self.atoms, dimension=1)
            if debug > 0: 
                print(f"SPLIT COMPLEX: rest labels: {rest_labels}")
                print(f"SPLIT COMPLEX: rest coord: {rest_coord}")
                print(f"SPLIT COMPLEX: rest indices: {rest_indices}")
                print(f"SPLIT COMPLEX: rest radii: {rest_radii}")

            if hasattr(self,"frac_coord"): rest_frac = extract_from_list(rest_idx, self.frac_coord, dimension=1)
            if debug > 0: print(f"SPLIT COMPLEX: splitting species with {len(rest_labels)} atoms in block")
            if hasattr(self,"cov_factor"): blocklist = split_species(rest_labels, rest_coord, radii=rest_radii, cov_factor=self.cov_factor, debug=debug)
            else:                          blocklist = split_species(rest_labels, rest_coord, radii=rest_radii, cov_factor=self.cov_factor, debug=debug)      
            if debug > 0: print(f"SPLIT COMPLEX: received {len(blocklist)} blocks")
            
            ## Arranges Ligands
            for b in blocklist:
                if debug > 0: print(f"PREPARING BLOCK: {b}")
                lig_indices = extract_from_list(b, rest_indices, dimension=1)
                lig_labels  = extract_from_list(b, rest_labels, dimension=1) 
                lig_coord   = extract_from_list(b, rest_coord, dimension=1) 
                lig_radii   = extract_from_list(b, rest_radii, dimension=1) 
                lig_atoms   = extract_from_list(b, rest_atoms, dimension=1) 
                if debug > 0: print(f"CREATING LIGAND: {labels2formula(lig_labels)}")
                # Create Ligand Object
                newligand   = ligand(lig_labels, lig_coord, radii=lig_radii)
                # For debugging
                newligand.origin = "split_complex"
                # Define the molecule as parent of the ligand. Bottom-Up hierarchy
                newligand.add_parent(self, indices=lig_indices)
                # Pass the molecule atoms to the ligand
                newligand.set_atoms(atomlist=lig_atoms)
                # Inherit the adjacencies from molecule
                newligand.inherit_adjmatrix("molecule")
                # If fractional coordinates are available...
                if hasattr(self,"frac_coord"): 
                    lig_frac_coord = extract_from_list(b, rest_frac, dimension=1)
                    newligand.set_fractional_coord(lig_frac_coord)
                # Add ligand to the list. Top-Down hierarchy
                self.ligands.append(newligand)

            ## Arranges Metals
            for m in metal_idx:
                ## We were creating the metal again, but it is already in the list of molecule.atoms
                #newmetal    = metal(self.labels[m], self.coord[m], self.radii[m])
                #newmetal.add_parent(self, index=self.indices[m])
                #self.metals.append(newmetal)                            
                self.metals.append(self.atoms[m])                            
        return self.ligands, self.metals

    #######################################################
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

    #######################################################
    def correct_smiles(self):
        if not hasattr(self,"smiles"): self.smiles = []
        if not self.iscomplex: return self.smiles
        for lig in self.ligands:
            lig.smiles, lig.rdkit_mol = correct_smiles_ligand(lig)
            self.smiles.append(lig.smiles)

###############
### LIGAND ####
###############
class ligand(specie):
    def __init__(self, labels: list, coord: list, radii: list=None) -> None:
        self.subtype  = "ligand"
        specie.__init__(self, labels, coord, radii)
        self.evaluate_as_nitrosyl()
        
    #######################################################
    def __repr__(self):
        to_print = ""
        to_print += f'------------- Cell2mol LIGAND Object --------------\n'
        to_print += specie.__repr__(self, indirect=True)
        if hasattr(self,"groups"): to_print += f' Number of Groups             = {len(self.groups)}\n'
        to_print += '---------------------------------------------------\n'
        return to_print

    #######################################################
    def get_connected_metals(self, debug: int=0):
        # metal.groups will be used for the calculation of the relative metal radius 
        # and define the coordination geometry of the metal /hapicitiy/ hapttype    
        self.metals = []
        mol = self.get_parent("molecule")
        for met in mol.metals:
            tmplabels = self.labels.copy()
            tmpcoord  = self.coord.copy()
            tmplabels.append(met.label)
            tmpcoord.append(met.coord)
            isgood, tmpadjmat, tmpadjnum = get_adjmatrix(tmplabels, tmpcoord, metal_only=True)
            if isgood and any(tmpadjnum) > 0: self.metals.append(met)
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
        ## Remember madjmat should not be computed at the ligand level. Since the metal is not there.
        ## Now we operate at the molecular level. We get the parent molecule, and the indices of the ligand atoms in the molecule
        self.connected_idx = [] 
        if not hasattr(self,"madjnum"): self.inherit_adjmatrix("molecule")
        for idx, con in enumerate(self.madjnum):
            if con > 0: self.connected_idx.append(idx)
        return self.connected_idx 

    #######################################################
    def get_connected_atoms(self, debug: int=0):
        if not hasattr(self,"atoms"):         self.set_atoms()
        if not hasattr(self,"connected_idx"): self.get_connected_idx()
        self.connected_atoms = []
        for idx, at in enumerate(self.atoms):
            if idx in self.connected_idx and at.mconnec > 0: 
                self.connected_atoms.append(at) 
            elif idx in self.connected_idx and at.mconnec == 0:
                print("WARNING: Atom appears in connected_idx, but has mconnec=0")
        return self.connected_atoms

    #######################################################
    def check_coordination(self, debug: int=0):
        if not hasattr(self,"groups"):      self.split_ligand(debug=debug)
        for g in self.groups:
            if not hasattr(g,"checked_coordination"): g.check_coordination(debug=debug)

    #######################################################
    def get_denticity(self, debug: int=0):
        if not hasattr(self,"groups"):      self.split_ligand(debug=debug)
        if debug > 0: print(f"LIGAND.Get_denticity: checking connectivity of ligand {self.formula}")
        if debug > 0: print(f"LIGAND.Get_denticity: initial connectivity is {len(self.connected_idx)}")
        self.denticity = 0
        for g in self.groups:
            #if debug > 0: print(f"LIGAND.Get_denticity: checking denticity of group \n{g}\n{g.madjnum=}\n{g.madjmat=}")
            self.denticity += g.get_denticity(debug=debug)      ## A check is also performed at the group level
        if debug > 0: print(f"LIGAND.Get_denticity: final connectivity is {self.denticity}")
        return self.denticity 

    #######################################################
    def split_ligand(self, debug: int=2):
        # Split the "ligand to obtain the groups
        self.groups = []
        # Identify Connected and Unconnected atoms (to the metal)
        if not hasattr(self,"connected_idx"): self.get_connected_idx()

        ## Creates the list of variables
        conn_idx     = self.connected_idx
        # if debug > 0: 
        print(f"LIGAND.SPLIT_LIGAND: {self.indices=}")
        # if debug > 0: 
        print(f"LIGAND.SPLIT_LIGAND: {conn_idx=}")
        conn_labels  = extract_from_list(conn_idx, self.labels, dimension=1)
        conn_coord   = extract_from_list(conn_idx, self.coord, dimension=1)
        conn_radii   = extract_from_list(conn_idx, self.radii, dimension=1)
        conn_atoms   = extract_from_list(conn_idx, self.atoms, dimension=1)
        print(f"LIGAND.SPLIT_LIGAND: {conn_labels=}")
        print(f"LIGAND.SPLIT_LIGAND: {conn_coord=}")
        print(f"LIGAND.SPLIT_LIGAND: {conn_radii=}")
        # print(f"LIGAND.SPLIT_LIGAND: {conn_atoms=}")
        if hasattr(self,"cov_factor"): blocklist = split_species(conn_labels, conn_coord, radii=conn_radii, cov_factor=self.cov_factor, debug=debug)
        else:                          blocklist = split_species(conn_labels, conn_coord, radii=conn_radii, debug=debug)      
        print(f"blocklist={blocklist}")
        ## Arranges Groups 
        for b in blocklist:
            # if debug > 0: 
            print(f"LIGAND.SPLIT_LIGAND: block={b}")
            gr_indices = extract_from_list(b, conn_idx, dimension=1)
            # if debug > 0: print(f"LIGAND.SPLIT_LIGAND: {gr_indices=}")
            gr_labels  = extract_from_list(b, conn_labels, dimension=1)
            gr_coord   = extract_from_list(b, conn_coord, dimension=1)
            gr_radii   = extract_from_list(b, conn_radii, dimension=1)
            gr_atoms   = extract_from_list(b, conn_atoms, dimension=1)
            # Create Group Object
            newgroup = group(gr_labels, gr_coord, radii=gr_radii)
            # For debugging
            newgroup.origin = "split_ligand"
            # Define the ligand as parent of the group. Bottom-Up hierarchy
            newgroup.add_parent(self, indices=gr_indices)
            # Pass the ligand atoms to the groud
            newgroup.set_atoms(atomlist=gr_atoms)
            # Inherit the adjacencies from molecule
            newgroup.inherit_adjmatrix("ligand")
            # Associate the Groups with the Metals
            newgroup.get_connected_metals()
            newgroup.get_closest_metal()
            # Top-down hierarchy
            self.groups.append(newgroup)
        return self.groups

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
    def __init__(self, labels: list, coord: list, radii: list=None) -> None:
        self.subtype = "group"
        specie.__init__(self, labels, coord, radii)

    #######################################################
    def __repr__(self):
        to_print = ""
        to_print += f'------------- Cell2mol GROUP Object --------------\n'
        to_print += specie.__repr__(self, indirect=True)
        if hasattr(self,"metals"):   to_print += f' Number of Metals             = {len(self.metals)}\n'
        to_print += '---------------------------------------------------\n'
        return to_print

    #######################################################
    def remove_atom(self, index: int, debug: int=0):
        if debug > 0: print(f"GROUP.REMOVE_ATOM: deleting atom {index=} from group with {self.natoms} atoms")
        if index > self.natoms: return None
        if not hasattr(self,"atoms"): self.set_atoms()
        self.atoms.pop(index)
        self.labels.pop(index)
        self.coord.pop(index)
        self.radii.pop(index)
        self.formula   = labels2formula(self.labels)
        self.eleccount = labels2electrons(self.labels)   ### Assuming neutral specie (so basically this is the sum of atomic numbers)
        self.natoms    = len(self.labels)
        self.iscomplex = any((elemdatabase.elementblock[l] == "d") or (elemdatabase.elementblock[l] == "f") for l in self.labels)
        if debug > 0: print("GROUP.REMOVE_ATOM. Group after removing atom:")
        if debug > 0: print(self)
        if self.natoms > 0:
            if hasattr(self,"closest_metal"): self.get_closest_metal()
            if hasattr(self,"is_haptic"):     self.get_hapticity()
            if hasattr(self,"centroid"):      self.get_centroid()
            if hasattr(self,"frac_coord"):    self.frac_coord.pop(index)
            if hasattr(self,"adjmat"):        self.get_adjmatrix()
            if hasattr(self,"madjmat"):       self.get_metal_adjmatrix()

    #######################################################
    def get_closest_metal(self, debug: int=0):
        apos = compute_centroid(np.array(self.coord))
        dist = []
        mol  = self.get_parent("molecule")
        for met in mol.metals:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        # finds the closest Metal Atom (tgt)
        self.closest_metal = mol.metals[np.argmin(dist)]
        return self.closest_metal

    #######################################################
    def get_connected_metals(self, debug: int=0):
        # metal.groups will be used for the calculation of the relative metal radius 
        # and define the coordination geometry of the metal /hapicitiy/ hapttype    
        self.metals = []
        lig = self.get_parent("ligand")
        if not hasattr(lig,"metals"): lig.get_connected_metals()
        for met in lig.metals:
            tmplabels = self.labels.copy()
            tmpcoord  = self.coord.copy()
            tmplabels.append(met.label)
            tmpcoord.append(met.coord)
            isgood, tmpadjmat, tmpadjnum = get_adjmatrix(tmplabels, tmpcoord, metal_only=True)
            if isgood and any(tmpadjnum) > 0: self.metals.append(met)
        return self.metals
    
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
    def check_coordination(self, debug: int=2):
        from cell2mol.connectivity import add_atom
        if not hasattr(self,"is_haptic"): self.get_hapticity()
        if not hasattr(self,"atoms"):     self.set_atoms()
        if self.is_haptic:                self = coordination_correction_for_haptic(self, debug=debug)
        if self.is_haptic == False:       self = coordination_correction_for_nonhaptic(self, debug=debug)
        self.checked_coordination = True

    #######################################################
    def get_denticity(self, debug: int=0):
        if not hasattr(self,"checked_coordination"): self.check_coordination(debug=debug) 
        self.denticity = 0
        for a in self.atoms: 
            self.denticity += a.mconnec      
        return self.denticity

###############
### BOND ######
###############
class bond(object):
    def __init__(self, atom1: object, atom2: object, bond_order: int=1):
        self.type       = "bond"
        self.version    = "0.1"
        self.atom1      = atom1
        self.atom2      = atom2
        self.order      = bond_order

    def __repr__(self):
        to_print += f'------------- Cell2mol BOND Object --------------\n'
        to_print += f' Version               = {self.version}\n'
        to_print += f' Type                  = {self.type}\n'
        idx1 = self.atom1.get_parent_index("molecule")
        idx2 = self.atom2.get_parent_index("molecule")
        to_print += f' Molecule Atom 1       = {idx1}\n'
        to_print += f' Molecule Atom 2       = {idx2}\n'
        to_print += f' Bond Order            = {self.order}\n'
        to_print += '----------------------------------------------------\n'
        return to_print

###############
### ATOM ######
###############
class atom(object):
    def __init__(self, label: str, coord: list, radii: float=None, frac_coord: list=None) -> None:
        self.type            = "atom"
        self.version         = "0.1"
        self.label           = label
        self.coord           = coord
        self.atnum           = elemdatabase.elementnr[label]
        self.block           = elemdatabase.elementblock[label]
        self.parents         = []
        self.parents_index   = []

        if radii is None:                 self.radii = get_radii(label)
        else:                             self.radii = radii
        if frac_coord is not None:        self.frac_coord = frac_coord

        ############
    def add_parent(self, parent: object, index: int, overwrite: bool=True):
        ## associates a parent specie to self. The atom indices of self in parent are given in "indices"
        ## if parent of the same subtype already in self.parent then it is overwritten
        ## this is to avoid having a substructure (e.g. a ligand) in more than one superstructure (e.g. a molecule) 
        append = True
        for idx, p in enumerate(self.parents):
            if p.subtype == parent.subtype:
                if overwrite: 
                    self.parents[idx]         = parent
                    self.parents_index[idx]   = index
                append = False
        if append: 
            self.parents.append(parent)
            self.parents_index.append(index)

    ############
    def check_parent(self, subtype: str):
        ## checks if parent of a given subtype exists
        for p in self.parents:
            if p.subtype == subtype: return True
        return False

    ############
    def get_parent(self, subtype: str):
        ## retrieves parent of a given subtype 
        for p in self.parents:
            if p.subtype == subtype: return p
        return None

    ############
    def get_parent_index(self, subtype: str):
        ## retrieves parent of a given subtype 
        for idx, p in enumerate(self.parents):
            if p.subtype == subtype: return self.parents_index[idx]
        return None

    #######################################################
    def check_connectivity(self, other: object, debug: int=0):
        ## Checks whether two atoms are connected (through the adjacency)
        if not isinstance(other, type(self)): return False
        labels = list([self.label,other.label]) 
        coords = list([self.coord,other.coord]) 
        isgood, adjmat, adjnum = get_adjmatrix(labels, coords)
        if isgood and adjnum[0] > 0: return True
        else:                        return False

    #######################################################
    def add_bond(self, newbond: object, debug: int=0):
        if not hasattr(self,"bonds"): self.bonds = []
        at1 = newbond.atom1
        at2 = newbond.atom2
        found = False
        for b in self.bonds:
            if (b.atom1 == at1 and b.atom2 == at2) or (b.atom1 == at2 and b.atom2 == at1): 
                if debug > 0: print(f"ATOM.ADD_BOND found the same bond with atoms:") 
                if debug > 0: print(f"atom1: {b.atom1}") 
                if debug > 0: print(f"atom2: {b.atom2}") 
                found = True    ### It means that the same bond has already been defined
        if not found: self.bonds.append(newbond)

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
    def set_adjacencies(self, adjmat, madjmat, adjnum: int, madjnum: int):
        self.connec  = int(adjnum)
        self.mconnec = int(madjnum)
        self.adjacency       = []
        self.metal_adjacency = []
        for idx, c in enumerate(adjmat):   ## The atom only receives one row of adjmat, so this is not a matrix anymore. Keep in mind that the idx are the indices of parent
            if c > 0: self.adjacency.append(idx)
        for idx, c in enumerate(madjmat):  ## The atom only receives one row of madjmat, so this is not a matrix anymore
            if c > 0: self.metal_adjacency.append(idx)

    #######################################################
    def get_connected_metals(self, metalist: list, debug: int=0):
        self.metals = []
        for met in metalist:
            tmplabels = self.label.copy()
            tmpcoord  = self.coord.copy()
            tmplabels.append(met.label)
            tmpcoord.append(met.coord)
            isgood, tmpadjmat, tmpadjnum = get_adjmatrix(tmplabels, tmpcoord, metal_only=True)
            if isgood and any(tmpadjnum) > 0: self.metals.append(met)
        return self.metals

    #######################################################
    def get_closest_metal(self, debug: int=0):
        ## Here, the list of metal atoms must be provided
        apos = self.coord
        dist = []
        mol = self.get_parent("molecule")
        for met in mol.metals:
            bpos = np.array(met.coord)
            dist.append(np.linalg.norm(apos - bpos))
        self.closest_metal = mol.metals[np.argmin(dist)]
        return self.closest_metal

    #######################################################
    def information(self, cov_factor: float, metal_factor: float) -> None:
        self.cov_factor   = cov_factor
        self.metal_factor = metal_factor

    #######################################################
    def __repr__(self, indirect: bool=False):
        to_print = ""
        if not indirect: to_print += f'------------- Cell2mol ATOM Object ----------------\n'
        to_print += f' Version                      = {self.version}\n'
        to_print += f' Type                         = {self.type}\n'
        if hasattr(self,'subtype'): to_print += f' Sub-Type                     = {self.subtype}\n'
        to_print += f' Label                        = {self.label}\n'
        to_print += f' Atomic Number                = {self.atnum}\n'
        idx = self.get_parent_index("molecule") 
        if idx is not None: to_print += f' Index in Molecule            = {idx}\n'
        idx = self.get_parent_index("ligand") 
        if idx is not None: to_print += f' Index in Ligand              = {idx}\n'
        if hasattr(self,"occurrence"): to_print += f' Occurrence in Parent         = {self.occurrence}\n'
        if hasattr(self,"mconnec"):    to_print += f' Metal Adjacency (mconnec)    = {self.mconnec}\n'
        if hasattr(self,"connec"):     to_print += f' Regular Adjacencies (connec) = {self.mconnec}\n'
        if hasattr(self,"charge"):     to_print += f' Atom Charge                  = {self.charge}\n'
        if not indirect: to_print += '----------------------------------------------------\n'
        return to_print
    
    #######################################################
    def reset_mconnec(self, met, diff: int=-1, debug: int=0):
        if debug > 0: print(f"ATOM.RESET_MCONN: resetting mconnec (and connec) for atom {self.label=}")
        self.mconnec += diff
        self.connec  += diff
        exists = self.check_parent("ligand")
        if exists:
            lig     = self.get_parent("ligand")
            lig_idx = self.get_parent_index("ligand")
            met_idx = met.get_parent_index("ligand")
            if debug > 0: print(f"ATOM.RESET_MCONN: updating ligand atoms and madjnum")
            if debug > 0: print(f"ATOM.RESET_MCONN: initial {lig.madjnum=}") 
            if debug > 0: print(f"ATOM.RESET_MCONN: initial {lig.madjmat=}") 
            # Correct Ligand Data
            #lig.atoms[lig_idx].mconnec += diff           # Corrects data of atom object in ligand class
            #lig.atoms[lig_idx].connec  += diff           # Corrects data of atom object in ligand class
            lig.madjnum[lig_idx] += diff                    # Corrects data in metal_adjacency number of the ligand class
            lig.madjmat[lig_idx,met_idx] += diff            # Corrects data in metal_adjacency matrix
            lig.madjmat[met_idx,lig_idx] += diff            # Corrects data in metal_adjacency matrix
            lig.adjnum[lig_idx]  += diff                    # Corrects data in adjacency number of the ligand class
            lig.adjmat[lig_idx,met_idx]  += diff            # Corrects data in adjacency matrix
            lig.adjmat[met_idx,lig_idx]  += diff            # Corrects data in adjacency matrix
            # Correct Metal Data
            #met.mconnec += diff                             # Corrects data of metal object
            #met.connec  += diff                             # Corrects data of metal object
            # we should delete the adjacencies, but not a priority 
            if debug > 0: print(f"ATOM.RESET_MCONN: final {lig.madjnum=}") 
            if debug > 0: print(f"ATOM.RESET_MCONN: final {lig.madjmat=}") 
            lig.get_connected_idx(debug=debug)
            lig.get_connected_atoms(debug=debug)

        exists = self.check_parent("molecule")
        if exists:
            mol     = self.get_parent("molecule")
            mol_idx = self.get_parent_index("molecule")
            met_idx = met.get_parent_index("molecule")
            if debug > 0: print(f"ATOM.RESET_MCONN: updating molecule atoms and madjnum")
            # Correct Molecule Data
            #mol.atoms[mol_idx].mconnec += diff              # Corrects data of atom object in molecule class
            #mol.atoms[mol_idx].connec  += diff              # Corrects data of atom object in molecule class
            mol.madjnum[mol_idx] += diff                    # Corrects data in metal_adjacency number of the molecule class
            mol.madjmat[mol_idx,met_idx] += diff            # Corrects data in metal_adjacency matrix
            mol.madjmat[met_idx,mol_idx] += diff            # Corrects data in metal_adjacency matrix
            mol.adjnum[mol_idx]  += diff                    # Corrects data in adjacency number of the molecule class
            mol.adjmat[mol_idx,met_idx]  += diff            # Corrects data in adjacency matrix
            mol.adjmat[met_idx,mol_idx]  += diff            # Corrects data in adjacency matrix

###############
#### METAL ####
###############
class metal(atom):
    def __init__(self, label: str, coord: list, radii: float=None, frac_coord: list=None) -> None:
        self.subtype = "metal"
        atom.__init__(self, label, coord, radii=radii, frac_coord=frac_coord)

    #######################################################
    def get_valence_elec (self, m_ox: int):
        """ Count valence electrons for a given transition metal and metal oxidation state """
        v_elec = elemdatabase.valenceelectrons[self.label] - m_ox      
        if v_elec >= 0 :  self.valence_elec = v_elec
        else :            self.valence_elec = elemdatabase.elementgroup[self.label] - m_ox
        return self.valence_elec

    #######################################################
    def get_coord_sphere(self):
        if not self.check_parent("molecule"): return None
        mol = self.get_parent("molecule")
        pidx = self.get_parent_index("molecule")
        if not hasattr(mol,"adjmat"): mol.get_adjmatrix()
        adjmat = mol.adjmat.copy()
        
        ## Cordination sphere defined as a collection of atoms
        self.coord_sphere = []
        for idx, at in enumerate(adjmat[pidx]):
            if at >= 1: self.coord_sphere.append(mol.atoms[idx])
        return self.coord_sphere

    #######################################################
    def get_coord_sphere_formula(self):
        if not hasattr(self,"coord_sphere"): self.get_coord_sphere()
        self.coord_sphere_formula = labels2formula(list([at.label for at in self.coord_sphere])) 
        return self.coord_sphere_formula 

    #######################################################
    def get_connected_groups(self, debug: int=0):
        # metal.groups will be used for the calculation of the relative metal radius 
        # and define the coordination geometry of the metal /hapicitiy/ hapttype    
        if not self.check_parent("molecule"): return None
        mol = self.get_parent("molecule")
        self.groups = []
        for group in mol.ligand.groups:
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

    #######################################################
    def get_possible_cs(self, debug: int=0):
        self.possible_cs = get_metal_poscharges(self)
        return self.possible_cs

    #######################################################
    def get_spin(self):
        self.spin = assign_spin_metal(self)

    ############
    def reset_charge(self):
        atom.reset_charge(self)     ## First uses the generic atom class function for itself
        if hasattr(self,"poscharges"):   delattr(self,"poscharge") 

    def __repr__(self):
        to_print = ""
        to_print += f'------------- Cell2mol METAL Object --------------\n'
        to_print += atom.__repr__(self, indirect=True)
        if hasattr(self,"coord_sphere_formula"): to_print += f' Coordination Sphere Formula  = {self.coord_sphere_formula}\n'
        if hasattr(self,"possible_cs"):          to_print += f' Possible Charges             = {self.possible_cs}\n'
        to_print += '----------------------------------------------------\n'
        return to_print

##############
#### CELL ####
##############
class cell(object):
    def __init__(self, name: str, labels: list, pos: list, cellvec: list, cellparam: list) -> None:
        self.version    = "0.1"
        self.type       = "cell"
        self.subtype    = "cell"
        self.name       = name
        self.labels     = labels 
        self.coord      = pos
        self.cellvec    = cellvec
        self.cellparam  = cellparam
        self.natoms     = len(labels)
        self.frac_coord = cart2frac(self.coord, self.cellvec)
        
    #######################################################
    def get_unique_species(self, debug: int=0): 
        if not hasattr(self,"is_fragmented"): self.reconstruct(debug=debug)  
        if self.is_fragmented: return None # Stopping. self.is_fragmented must be false to determine the charges of the cell

        self.unique_species = []
        self.unique_indices = []

        typelist_mols = [] # temporary variable  
        typelist_ligs = [] # temporary variable
        typelist_mets = [] # temporary variable

        specs_found = -1
        for idx, mol in enumerate(self.moleclist):
            if debug >= 2: print(f"Molecule {idx} formula={mol.formula}")
            if not mol.iscomplex:
                found = False
                for ldx, typ in enumerate(typelist_mols):   # Molecules
                    issame = compare_species(mol, typ[0], debug=debug)
                    if issame :
                        found = True ; kdx = typ[1]
                        if debug >= 2: print(f"Molecule {idx} is the same with {ldx} in typelist")
                if not found:
                    specs_found += 1 ; kdx = specs_found
                    typelist_mols.append(list([mol, kdx]))
                    self.unique_species.append(mol)
                    if debug >= 2: print(f"New molecule found with: formula={mol.formula} and added in position {kdx}")
                self.unique_indices.append(kdx)
                mol.unique_index = kdx

            else:
                if not hasattr(mol,"ligands"): mol.split_complex(debug=debug)
                for jdx, lig in enumerate(mol.ligands):     # ligands
                    found = False
                    for ldx, typ in enumerate(typelist_ligs):
                        issame = compare_species(lig, typ[0], debug=debug)
                        if issame :
                            found = True ; kdx = typ[1]
                            if debug >= 2: print(f"ligand {jdx} is the same with {ldx} in typelist")
                    if not found:
                        specs_found += 1 ; kdx = specs_found
                        typelist_ligs.append(list([lig, kdx]))
                        self.unique_species.append(lig)
                        if debug >= 2: print(f"New ligand found with: formula {lig.formula} added in position {kdx}")
                    self.unique_indices.append(kdx)
                    lig.unique_index = kdx

                for jdx, met in enumerate(mol.metals):      #  metals
                    found = False
                    for ldx, typ in enumerate(typelist_mets):
                        issame = compare_metals(met, typ[0], debug=debug)
                        if issame :
                            found = True ; kdx = typ[1]
                            if debug >= 2: print(f"Metal {jdx} is the same with {ldx} in typelist")
                    if not found:
                        specs_found += 1 ; kdx = specs_found
                        typelist_mets.append(list([met, kdx]))
                        self.unique_species.append(met)
                        if debug >= 2: print(f"New Metal Center found with: labels {met.label} and added in position {kdx}")
                    self.unique_indices.append(kdx)
                    met.unique_index = kdx

        return self.unique_species

    #######################################################
    def get_fractional_coord(self):
        self.frac_coord = cart2frac(self.coord, self.cellvec)
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
            mol_coord        = extract_from_list(b, ref_pos, dimension=1)
            mol_frac_coord   = extract_from_list(b, self.frac_coord, dimension=1)
            newmolec         = molecule(mol_labels, mol_coord)
            newmolec.add_parent(self, indices=b)
            newmolec.set_fractional_coord(mol_frac_coord)
            newmolec.set_atoms(create_adjacencies=True, debug=debug)
            # This must be below the frac_coord, so they are carried on to the ligands
            if newmolec.iscomplex: newmolec.split_complex()
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
                   ref.get_hapticity(debug=debug)                          ### Former "get_hapticity(ref)" function
                   # ref.get_coordination_geometry(debug=debug)                ### Former "get_coordination_Geometry(ref)" function 
                   for lig in ref.ligands:
                       lig.get_denticity(debug=2)

        if isgood: self.has_isolated_H = False
        else:      self.has_isolated_H = True
        return self.refmoleclist

    #######################################################
    def get_moleclist(self, blocklist=None, debug: int=0):
        if debug > 0: print(f"Entered CELL.MOLECLIST with debug={debug}")
        if not hasattr(self,"labels") or not hasattr(self,"coord"): 
            if debug > 0: print(f"CELL.MOLECLIST. Labels or coordinates not found. Returning None")
            return None
        if len(self.labels) == 0 or len(self.coord) == 0:           
            if debug > 0: print(f"CELL.MOLECLIST. Empty labels or coordinates. Returning None")
            return None
        if debug > 0: print(f"CELL.MOLECLIST passed initial checks")
        cov_factor = 1.3

        if blocklist is None: blocklist = split_species(self.labels, self.coord, cov_factor=cov_factor)

        self.moleclist = []
        for b in blocklist:
            if debug > 0: print(f"CELL.MOLECLIST: doing block={b}")
            mol_labels  = extract_from_list(b, self.labels, dimension=1)
            mol_coord   = extract_from_list(b, self.coord, dimension=1)
            # Creates Molecule Object
            newmolec    = molecule(mol_labels, mol_coord)
            # For debugging
            newmolec.origin = "cell.get_moleclist"
            # Creates The atom objects with adjacencies
            newmolec.set_atoms(create_adjacencies=True, debug=debug)
            # Adds cell as parent of the molecule, with indices b
            newmolec.add_parent(self, indices=b)
            # If fractional coordinates are available...
            if hasattr(self,"frac_coord"): 
                assert len(self.frac_coord) == len(self.coord)
                mol_frac_coord  = extract_from_list(b, self.frac_coord, dimension=1)
                newmolec.set_fractional_coord(mol_frac_coord, debug=debug)
            # The split_complex must be below the frac_coord, so they are carried on to the ligands
            if newmolec.iscomplex: 
                if debug > 0: print(f"CELL.MOLECLIST: splitting complex")
                newmolec.split_complex(debug=debug)
            self.moleclist.append(newmolec)

        return self.moleclist
   
    #######################################################
    def arrange_cell_coord(self): 
        ## Updates the cell coordinates preserving the original atom ordering
        ## Do do so, it uses the variable parent_indices stored in each molecule
        self.coord = np.zeros((self.natoms,3))
        for mol in self.moleclist:
            idx = mol.get_parent_indices("cell")
            for z in zip(idx, mol.coord):
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
        if not hasattr(self,"refmoleclist"): print("CELL.RECONSTRUCT. CELL missing list of reference molecules"); return None
        if cov_factor is None:   cov_factor   = self.refmoleclist[0].cov_factor
        if metal_factor is None: metal_factor = self.refmoleclist[0].metal_factor

        ## Get the fragments, which is the moleclist of a fragmented cell
        fragments = self.get_moleclist().copy() 
        ## Classifies fragments
        for f in fragments:
            if not hasattr(f,"frac_coord"):       f.get_fractional_coord(self.cellvec)
        molecules, fragments, hydrogens = classify_fragments(fragments, self.refmoleclist, debug=debug)

        ## Determines if Reconstruction is necessary
        if len(fragments) > 0 or len(hydrogens) > 0: self.is_fragmented = True
        else:                                        self.is_fragmented = False
        
        self.moleclist = []
        if not self.is_fragmented: 
            for mol in molecules:
                self.moleclist.append(mol)
                #########################################
                ## In principle, this is not necessary ##
                #########################################
                #newmolec = molecule(mol.labels, mol.coord)
                #newmolec.add_parent(self,mol_indices)
                #newmolec.set_fractional_coord(mol.frac_coord)
                #newmolec.set_atoms(debug=debug, create_adjacencies=True)          
                #if newmolec.iscomplex: 
                #    newmolec.split_complex()
                #    newmolec.get_hapticity()
                #self.moleclist.append(newmolec)   
            return self.moleclist     
        else :
            reconstructed_molecules, Warning = fragments_reconstruct(molecules, fragments, hydrogens, self.refmoleclist, self.cellvec, cov_factor, metal_factor)
            
            if Warning:
                self.is_fragmented = True
                self.error_reconstruction = True 
            else :
                self.is_fragmented = False
                self.error_reconstruction = False 

            ## For consistency, we create the molecules once again, even if mol is already a molecule-class object.
            ## One must follow the same structure as in self.get_moleclist()
            for mol in reconstructed_molecules:
                newmolec = molecule(mol.labels, mol.coord)
                newmolec.origin = "cell.reconstruct"
                newmolec.set_atoms(create_adjacencies=True, debug=debug)
                newmolec.add_parent(self,mol.cell_indices) 
                newmolec.set_fractional_coord(mol.frac_coord)
                if newmolec.iscomplex: newmolec.split_complex()
                self.moleclist.append(newmolec)         
            return self.moleclist
    
    def reset_charge_assignment(self, debug: int=0):
        if not hasattr(self,"moleclist"): return None
        for mol in self.moleclist:
            mol.reset_charge()
        
    #######################################################
    def assign_charges(self, debug: int=0) -> object:
    #########
    # CHARGE#
    #########
    # This function drives the determination of the charge the species in the unit cell
    # The whole process is done by 4 functions, which are run at the specie class level:
    # 1) spec.get_protonation_states(), which determines which atoms of the specie must have added elements (see above) to have a meaningful Lewis structure
    # 2) spec.get_possible_cs(), which retrieves the possible charge states associated with the specie
    # 3) spec.get_charge(), which generates one connectivity for a set of charges
    # 4) cell.select_charge_distr() chooses the best connectivity among the generated ones.

    # Basically, this function connects these other three functions,
    # while managing some key information for those
    # Initiates variables
    ############
        
        # (0) Makes sure the cell is reconstructed
        if not hasattr(self,"is_fragmented"): self.reconstruct(debug=debug)  
        if self.is_fragmented: return None # Stopping. self.is_fragmented must be false to determine the charges of the cell

        # (1) Indentify unique chemical species
        if not hasattr(self,"unique_species"): self.get_unique_species(debug=debug)  
        if debug >= 1: print(f"{len(self.unique_species)} Species (Metal or Ligand or Molecules) to Characterize")

        # (2) Gets a preliminary list of possible charge states for each specie (former drive_poscharge)
        selected_cs = []
        for idx, spec in enumerate(self.unique_species):
            tmp = spec.get_possible_cs(debug=debug)
            if tmp is None: 
                self.error_empty_poscharges = True
                return None # Empty list of possible charges received. Stopping
            if spec.subtype != "metal":
                selected_cs.append(list([cs.corr_total_charge for cs in spec.possible_cs]))
            else :
                selected_cs.append(spec.possible_cs)   
        self.error_empty_poscharges = False

        # Finds the charge_state that satisfies that the crystal must be neutral
        final_charge_distribution = balance_charge(self.unique_indices, self.unique_species, debug=debug)

        if len(final_charge_distribution) > 1:
            if debug >= 1: print("More than one Possible Distribution Found:", final_charge_distribution)
            self.error_multiple_distrib = True
            self.error_empty_distrib    = False
            pp_mols, pp_idx, pp_opt = prepare_unresolved(self.unique_indices, self.unique_species, final_charge_distribution, debug=debug)
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
            self.moleclist, self.error_prepare_mols = prepare_mols(self.moleclist, self.unique_indices, self.unique_species, selected_cs, final_charge_distribution[0], debug=debug)
            if self.error_prepare_mols: return None # Error while preparing molecules
            
            return self.moleclist

    #######################################################
    def create_bonds(self, debug: int=0):
        if not hasattr(self,"error_prepare_mols"): self.assign_charges(debug=debug)  
        if self.error_prepare_mols: return None # Stopping. self.error_prepare_mols must be false to create the spin
        for mol in self.moleclist:
            # First part
            if not mol.iscomplex: 
                mol.create_bonds(debug=debug)          ### Creates bonds between molecule.atoms using the molecule.rdkit_object
            
            # Second part
            if mol.iscomplex:
                for lig in mol.ligands:
                    lig.create_bonds(debug=debug)      ### Creates bonds between ligand.atoms, which also belong to molecule.atoms, using the ligand.rdkit_object
                
            # Third Part. Adds Metal-Ligand Bonds, with an arbitrary 0.5 order:
            if mol.iscomplex:
                for lig in mol.ligands:
                    for at in lig.atoms:
                        count = 0
                        for met in mol.metals: 
                            isconnected = at.check_connectivity(met, debug=debug)
                            # isconnected = check_connectivity(at, met)
                            if isconnected: 
                                newbond = bond(at, met, 0.5)
                                at.add_bond(newbond)
                                met.add_bond(newbond)
                                count += 1 
                        if count != at.mconnec: 
                            print(f"CELL.CREATE_BONDS: error creating bonds for atom: \n{atom}\n of ligand: \n{lig}\n")
                            print(f"CELL.CREATE_BONDS: count differs from atom.mconnec: {count}, {at.mconnec}")

            # Adds Metal-Metal Bonds, with an arbitrary 0.5 order:
            if mol.iscomplex:
                for idx, met1 in mol.metals:
                    for jdx, met2 in mol.metals:
                        if idx <= jdx: continue
                        isconnected = met1.check_connectivity(met2, debug=debug)
                        # isconnected = check_connectivity(met1, met2)
                        if isconnected:
                            newbond = bond(met1, met2, 0.5)
                            met1.add_bond(newbond) 
                            met2.add_bond(newbond) 


                # Fourth part : correction smiles of ligands
                mol.smiles_with_H = [lig.smiles for lig in mol.ligands]
                mol.smiles = mol.correct_smiles()

    #######################################################
    def assign_spin(self, debug: int=0) -> object:
        if not hasattr(self,"error_prepare_mols"): self.assign_charges(debug=debug)  
        if self.error_prepare_mols: return None # Stopping. self.error_prepare_mols must be false to assign the spin
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
        to_print  = f'------------- Cell2mol CELL Object ----------------\n'
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
