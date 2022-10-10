#!/usr/env python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment as lsa
def center(X):
    X = np.array(X)
    C = X.mean(axis=0)
    X -= C
    return X, C
def reorder_hungarian(z1, z2, coord1, coord2):
    unique_atoms = np.unique(z1)
    map12 = np.zeros_like(z1, dtype=int)
    map12 -= 1
    for atom in unique_atoms:
        (aidx1,) = np.where(z1 == atom)
        (aidx2,) = np.where(z2 == atom)
        acoord1 = coord1[aidx1]
        acoord2 = coord2[aidx2]
        v = hungarian(acoord1, acoord2)
        map12[aidx1] = aidx2[v]
    return map12
def hungarian(a, b):
    distances = cdist(a, b, "euclidean")
    ia, ib = lsa(distances)
    return ib
def reorder(z1, z2, coord1, coord2):
    z1 = np.array(z1)
    z2 = np.array(z2)
    coord1, c1 = center(coord1)
    coord2, c2 = center(coord2)
    assert len(z1) == len(z2)
    assert coord1.shape == coord2.shape
    map12 = reorder_hungarian(z1, z2, coord1, coord2)
    z2 = z2[map12]
    coord2 = coord2[map12, :]
    return list(z2), list(coord2 + c2), map12
def test_reorder():
    # Two water molecules with different order!
    A = np.array([[1.5, 1.0], [1.0, 2.0], [2.0, 1.5]])
    B = np.array([[1.0, 2.0], [1.5, 1.0], [2.0, 1.5]])
    rn = np.random.random(3)
    for i in range(3):
        A[i] *= 1 + (0.1 * rn[i])
    rn = np.random.random(3)
    for i in range(3):
        B[i] *= 1 + (0.1 * rn[i])
    za = ["O", "H", "H"]
    zb = ["H", "O", "H"]
    # We attempt to reorder them
    zb2, B2, mapab = reorder(za, zb, A, B)
    assert za == zb2
    for i, idx in enumerate(mapab):
        print(za[i], zb2[i], zb[idx])
        print(A[i], B2[i], B[idx])
if __name__ == "__main__":
    test_reorder()
