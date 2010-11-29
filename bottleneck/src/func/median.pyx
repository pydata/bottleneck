
# proposed avg. O(n) replacement for NumPy's median
# (C) 2009 Sturla Molden
# SciPy license

"""
import numpy as np

try:
    from quickselect import select 

except ImportError:

    def _select(a, k):
        ''' Python quickselect for reference only '''
        l = 0
        r = a.shape[0] - 1
        while l < r:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j

    def select(a, k, inplace=False):
        '''
    Wirth's version of Hoare's quick select

    Parameters
    ----------
    a : array_like
    k : integer
    inplace : boolean
        The partial sort is done inplace if a is a
        contiguous ndarray and ndarray and inplace=True. Default: False. 
    
    Returns
    -------
    out : ndarray
        Partially sorted a such that out[k] is
        the k largest element. Elements smaller than
        out[k] are unsorted in out[:k]. Elements larger
        than out[k] are unsorted in out[k:].

    Python version for reference only!
        
        '''
        
        if inplace:
            _a = np.ascontiguousarray(a)
        else:
            _a = np.array(a)
        _select(_a,k)    
        return _a
"""

def _median(x, inplace):
    assert(x.ndim == 1)
    n = x.shape[0]
    if n > 3:
        k = n >> 1
        s = select(x, k, inplace=inplace)
        if n & 1:
            return s[k]
        else:
            return 0.5*(s[k]+s[:k].max())      
    elif n == 0:
        return np.nan
    elif n == 2:
        return 0.5*(x[0]+x[1])        
    else: # n == 3
        s = select(x, 1, inplace=inplace)
        return s[1]

        

def median(a, axis=None, out=None, overwrite_input=False):
    """
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {None, int}, optional
        Axis along which the medians are computed. The default (axis=None)
        is to compute the median along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : {False, True}, optional
       If True, then allow use of memory of input array (a) for
       calculations. The input array will be modified by the call to
       median. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. Note that, if `overwrite_input` is True and the input
       is not already an ndarray, an error will be raised.

    Returns
    -------
    median : ndarray
        A new array holding the result (unless `out` is specified, in
        which case that array is returned instead).  If the input contains
        integers, or floats of smaller precision than 64, then the output
        data-type is float64.  Otherwise, the output data-type is the same
        as that of the input.

    See Also
    --------
    mean

    Notes
    -----
    Given a vector V of length N, the median of V is the middle value of
    a sorted copy of V, ``V_sorted`` - i.e., ``V_sorted[(N-1)/2]``, when N is
    odd.  When N is even, it is the average of the two middle values of
    ``V_sorted``.

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.median(a)
    3.5
    >>> np.median(a, axis=0)
    array([ 6.5,  4.5,  2.5])
    >>> np.median(a, axis=1)
    array([ 7.,  2.])
    >>> m = np.median(a, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.median(a, axis=0, out=m)
    array([ 6.5,  4.5,  2.5])
    >>> m
    array([ 6.5,  4.5,  2.5])
    >>> b = a.copy()
    >>> np.median(b, axis=1, overwrite_input=True)
    array([ 7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.median(b, axis=None, overwrite_input=True)
    3.5
    >>> assert not np.all(a==b)

    """

    if overwrite_input and not isinstance(a, np.ndarray):
         raise ValueError, 'a must be ndarray when overwrite_input is True'

    a = np.asarray(a)

    if a.ndim == 1:
        if axis:
            raise ValueError, 'axis out of bounds'
        retv = _median(a, overwrite_input)

    elif a.ndim == 2:
        if axis is None:
            retv = _median(a.ravel(), overwrite_input)
        elif axis == 0:
            n = a.shape[1]
            retv = np.array([_median(a[:,i], overwrite_input) for i in xrange(n)])
        elif axis == 1:            
            n = a.shape[0]
            retv = np.array([_median(a[i,:], overwrite_input) for i in xrange(n)])
        else:            
            raise ValueError, 'axis out of bounds'
       
    else:
        if axis:
            retv = np.apply_along_axis(_median, axis, a, overwrite_input)
        else:
            retv = _median(a.ravel(), overwrite_input)

    if out is not None:
        if np.isscalar(retv):
            out[:] = [retv]
        else:
            out[:] = retv[:]
    else:
        return retv



@cython.boundscheck(False)
@cython.wraparound(False)
def _select_uint8(np.ndarray[np.uint8_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.uint8_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_int8(np.ndarray[np.int8_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.int8_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_uint16(np.ndarray[np.uint16_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.uint16_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_int16(np.ndarray[np.int16_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.int16_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_uint32(np.ndarray[np.uint32_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.uint32_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_int32(np.ndarray[np.int32_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.int32_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_int64(np.ndarray[np.int64_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.int64_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_uint64(np.ndarray[np.uint64_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.uint64_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_float32(np.ndarray[np.float32_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.float32_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






@cython.boundscheck(False)
@cython.wraparound(False)
def _select_float64(np.ndarray[np.float64_t, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef np.float64_t x, tmp
    l = 0
    r = a.shape[0] - 1 
    with nogil:       
        while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j





_jumptab = {
    np.uint8 : _select_uint8,
    np.int8 : _select_int8,
    np.uint16 : _select_uint16,
    np.int16 : _select_int16,
    np.uint32 : _select_uint32,
    np.int32 : _select_int32,
    np.int64 : _select_int64,
    np.uint64 : _select_uint64,
    np.float32 : _select_float32,
    np.float64 : _select_float64,
}





@cython.boundscheck(False)
@cython.wraparound(False)
def _select_pyobject(np.ndarray[object, ndim=1, mode="c"] a, np.npy_intp k):
    cdef np.npy_intp i, j, l, r
    cdef object x, tmp
    l = 0
    r = a.shape[0] - 1 
    while l < k:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j






def select(a, k, inplace=False):
    '''
    Wirth's version of Hoare's quick select

    Parameters
    ----------
    a : array_like
    k : integer
    inplace : boolean
        The partial sort is done inplace if a is a
        contiguous ndarray and inplace=True.
        Default: False. 
    
    Returns
    -------
    out : ndarray
        Partially sorted a such that out[k] is
        the k largest element. Elements smaller than
        out[k] are unsorted in out[:k]. Elements larger
        than out[k] are unsorted in out[k:].
    
    '''
    
    if inplace:
        _a = np.ascontiguousarray(a)
    else:
        _a = np.array(a)
    try:
        _select = _jumptab[_a.dtype.type] 
    except KeyError:
        _select = _select_pyobject
    _select(_a,k)    
    return _a
