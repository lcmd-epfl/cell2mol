import numpy
from numpy.linalg import eig

# Python3 compatibility
if isinstance(u"abc",str):   #Python 3
    unicode = str

def aug_append(current,extra):
    """Add the contents of extra to current"""
    have_list = isinstance(current,list)
    if have_list:
        if not isinstance(extra, list):
            #append a single element
            return current + [extra]
        else:
            newlist = current[:]
            newlist.append(extra)
            return newlist
    elif isinstance(current,numpy.ndarray):
        if current.ndim == extra.ndim + 1:
            extra = numpy.expand_dims(extra,axis=0)
        elif current.ndim == extra.ndim:
            extra = numpy.expand_dims(extra,axis=0)
            current = numpy.expand_dims(current,axis=0)
        else:
            raise ValueError('Arrays have mismatching sizes for concatenating: %d and %d' % (current.ndim,extra.ndim))
        return numpy.concatenate((current,extra))
    raise ValueError("Cannot append %s to %s" % (repr(extra),repr(current)))

def aug_add(current,extra):
    """Sum the contents of extra to current"""
    have_list = isinstance(current,list)
    if have_list:
        if isinstance(extra, (float,int)):
           # requires numpy
           return numpy.array(current) + extra
        elif isinstance(extra, list):
           return numpy.array(current) + numpy.array(extra)
    else:
        return current + extra

def aug_sub(current,extra):
   have_list = isinstance(current,(list,numpy.ndarray))
   if have_list:
        if isinstance(extra, (float,int)):
           # requires numpy
           return numpy.array(current) - extra
        elif isinstance(extra, (list,numpy.ndarray)):
           return numpy.array(current) - numpy.array(extra)
   else:
        return current - extra

def aug_remove(current,extra):
    """Remove extra from current. Not in formal
       specifications. Allowed to fail silently."""
    have_list = isinstance(current,list)
    if have_list:
        if extra in current:
            # not efficient as we modify in place here
            current.remove(extra)
            return current
        else:
            print('Removal Warning: %s not in %s' % (repr(extra),repr(current)))
            return current
    else:
        raise ValueError("Cannot remove %s from %s" % (repr(extra),repr(current)))

def drel_dot(first_arg,second_arg):
    """Perform a multiplication on two unknown types"""
    print("Multiply %s and %s" % (repr(first_arg),repr(second_arg)))
    def make_numpy(input_arg):
        if hasattr(input_arg,'__iter__'):
            try:
                return numpy.matrix(input_arg),True
            except ValueError:
                raise ValueError('Attempt to multiply non-matrix object %s' % (repr(input_arg)))
        return input_arg,False
    fa,first_matrix = make_numpy(first_arg)
    sa,second_matrix = make_numpy(second_arg)
    if first_matrix and second_matrix:  #mult of 2 non-scalars
        if sa.shape[0] == 1:  #is a row vector
           as_column = sa.T
           result = (fa * as_column).T
        else:
           result = fa * sa
       # detect scalars
        if result.size == 1:
            return result.item(0)
       # remove extra dimension
        elif result.ndim == 2 and 1 in result.shape:  #vector
            return numpy.array(result).squeeze()
        else:
            return result
    return fa * sa

def drel_add(first_arg,second_arg):
    """Separate string addition from the rest"""
    if isinstance(first_arg,(unicode,str)) and isinstance(second_arg,(unicode,str)):
        return first_arg+second_arg
    else:
        result = numpy.add(first_arg,second_arg)
        return result


def drel_eigen(in_matrix):
    """Return 3 lists of form [a,v1,v2,v3], corresponding to the 3 eigenvalues 
       and eigenvector components of a 3x3 matrix"""
    vals,vects = eig(in_matrix)
    move = list(numpy.argsort(vals))
    move.reverse()
    vals = vals[move]
    vects = vects[move]
    vects = list([[a]+list(numpy.asarray(v).ravel()) for a,v in zip(vals,vects)]) #Eigen returns 4-list
    return vects

def drel_int(in_val):
    """Return in_val as an integer"""
    try:
        return in_val.astype('int')
    except:
        return int(in_val)

def drel_strip(in_list,element):
    """Return the nth element from the list"""
    return [a[element] for a in in_list]

