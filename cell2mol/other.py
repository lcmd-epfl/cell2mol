## File for small functions
import numpy as np

#######################################################
def additem(item, vector):
    if item not in vector:
        vector.append(item)
    return vector

#######################################################
def absolute_value(num):
    sum = 0
    for i in num:
        sum += np.abs(i)
    return abs(sum)

#######################################################
def det3(mat):
    """ Calculate the determinant of a 3x3 matrix
    
    Args:   
        mat (list): list of 3x3 matrix
    Returns:
        determinant (float): determinant of the matrix
    """    
    return (
        (mat[0][0] * mat[1][1] * mat[2][2])
        + (mat[0][1] * mat[1][2] * mat[2][0])
        + (mat[0][2] * mat[1][0] * mat[2][1])
        - (mat[0][2] * mat[1][1] * mat[2][0])
        - (mat[0][1] * mat[1][0] * mat[2][2])
        - (mat[0][0] * mat[1][2] * mat[2][1])
    )

####################################
def inv(perm: list) -> list:

    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

#######################################################
def compute_centroid(arr: np.array) -> list:
    # Get centroid of a set of coordinates
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    centroid = np.around(np.array([sum_x/length, sum_y/length, sum_z/length]),7)
    return np.array(centroid)

#######################################################
def get_dist (atom1_pos: list, atom2_pos: list) -> float :
    dist = np.linalg.norm(np.array(atom1_pos) - np.array(atom2_pos))
    return round(dist, 3)

#######################################################
def handle_error(case: int):
    print(f"Cell2mol terminated with error number {case}. Message:")
    if case == 1: print("The cell object has isolated H atoms in the reference molecules list. This typically indicates an error. STOPPING") 
    if case == 2: print("We detected that H atoms are likely missing. This will cause errors in the charge prediction, so STOPPING pre-emptively.") 
    if case == 3: print("After reconstruction of the unit cell, we still detected some fragments. STOPPING pre-emptively.") 
    sys.exit(1)