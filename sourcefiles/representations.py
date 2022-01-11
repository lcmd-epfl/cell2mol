#!/usr/bin/env python
import numpy as np
from rdkit import Chem

def dist_matrix(coord):
    import numpy as np
    natoms = len(coord)
    print(natoms, coord)
    d = np.zeros((natoms, natoms))
    for idx, a in enumerate(coord):
        for jdx, b in enumerate(coord):
            print("a", a)
            print("b", b)
            if idx < jdx:
                d[idx, jdx] = np.linalg.norm(a-b)
                d[jdx, idx] = d[idx, jdx]
            #else:
            #    continue
    return d

def coulomb_matrix_from_gmol(atnums, pos):
    natoms = len(atnums)
    print(natoms, coord)
    d = dist_matrix(pos) 
    m = np.zeros((natoms, natoms))
    for i in atnums:
        for j in atnums:
            if i == j:
                m[i, j] = 0.5 * z[i] ** 2.4
            elif i < j:
                m[i, j] = (z[i] * z[j]) / d[i, j]
                m[j, i] = m[i, j]
            #else:
            #    continue
    return m

def coulomb_matrix_from_mol(mol):
    #from rdkit import Chem
    """
    Generate Coulomb matrices for each conformer of the given molecule.
    Parameters
    ----------
    mol : RDKit Mol
    Molecule.
    """
    if self.remove_hydrogens:
        mol = Chem.RemoveHs(mol)
    n_atoms = mol.GetNumAtoms()
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    rval = []
    for conf in mol.GetConformers():
        d = self.get_interatomic_distances(conf)
        m = np.zeros((n_atoms, n_atoms))
        for i in xrange(mol.GetNumAtoms()):
            for j in xrange(mol.GetNumAtoms()):
                if i == j:
                    m[i, j] = 0.5 * z[i] ** 2.4
                elif i < j:
                    m[i, j] = (z[i] * z[j]) / d[i, j]
                    m[j, i] = m[i, j]
                else:
                    continue
        if self.randomize:
            for random_m in self.randomize_coulomb_matrix(m):
                random_m = pad_array(random_m, self.max_atoms)
                rval.append(random_m)
        else:
            m = pad_array(m, self.max_atoms)
            rval.append(m)
    rval = np.asarray(rval)
    return rval

