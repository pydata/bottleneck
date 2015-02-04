#cython: embedsignature=True

import numpy as np
cimport numpy as np
import cython

from numpy cimport float64_t, float32_t, int64_t, int32_t, intp_t
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INTP as NPY_intp

from numpy cimport PyArray_ITER_DATA as pid
from numpy cimport PyArray_ITER_NOTDONE
from numpy cimport PyArray_ITER_NEXT
from numpy cimport PyArray_ITER_RESET
from numpy cimport PyArray_IterAllButAxis

from numpy cimport PyArray_TYPE
from numpy cimport PyArray_NDIM
from numpy cimport NPY_CORDER
from numpy cimport PyArray_Copy
from numpy cimport PyArray_EMPTY
from numpy cimport PyArray_Ravel
from numpy cimport PyArray_ArgSort
from numpy cimport NPY_QUICKSORT
from numpy cimport PyArray_FillWithScalar

from numpy cimport ndarray
from numpy cimport import_array
import_array()

import bottleneck.slow.nonreduce_axis as slow

cdef double NAN = <double> np.nan


# partsort ------------------------------------------------------------------

def partsort(arr, int n, axis=-1):
    """
    Partial sorting of array elements along given axis.

    A partially sorted array is one in which the `n` smallest values appear
    (in any order) in the first `n` elements. The remaining largest elements
    are also unordered. Due to the algorithm used (Wirth's method), the nth
    smallest element is in its sorted position (at index `n-1`).

    Shuffling the input array may change the output. The only guarantee is
    that the first `n` elements will be the `n` smallest and the remaining
    element will appear in the remainder of the output.

    This functions is not protected against NaN. Therefore, you may get
    unexpected results if the input contains NaN.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    n : int
        The `n` smallest elements will appear (unordered) in the first `n`
        elements of the output array.
    axis : {int, None}, optional
        Axis along which the partial sort is performed. The default (axis=-1)
        is to sort along the last axis.

    Returns
    -------
    y : ndarray
        A partially sorted copy of the input array where the `n` smallest
        elements will appear (unordered) in the first `n` elements.

    See Also
    --------
    bottleneck.argpartsort: Indices that would partially sort an array

    Notes
    -----
    Unexpected results may occur if the input array contains NaN.

    Examples
    --------
    Create a numpy array:

    >>> a = np.array([1, 0, 3, 4, 2])

    Partially sort array so that the first 3 elements are the smallest 3
    elements (note, as in this example, that the smallest 3 elements may not
    be sorted):

    >>> bn.partsort(a, n=3)
    array([1, 0, 2, 4, 3])

    Now partially sort array so that the last 2 elements are the largest 2
    elements:

    >>> bn.partsort(a, n=a.shape[0]-2)
    array([1, 0, 2, 3, 4])

    """
    try:
        return nonreducer_axis(arr, axis,
                               partsort_float64,
                               partsort_float32,
                               partsort_int64,
                               partsort_int32,
                               n)
    except TypeError:
        return slow.partsort(arr, n, axis)


cdef ndarray partsort_DTYPE0(ndarray a, int axis,
                             int a_ndim, np.npy_intp* y_dims, int n):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef np.npy_intp i, j = 0, l, r, k = n-1
    cdef DTYPE0_t x, tmpi, tmpj
    cdef ndarray y = PyArray_Copy(a)
    cdef Py_ssize_t stride = y.strides[axis]
    cdef Py_ssize_t length = y.shape[axis]
    if length == 0:
        return y
    if (n < 1) or (n > length):
        raise ValueError("`n` (=%d) must be between 1 and %d, inclusive." %
                         (n, length))
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    while PyArray_ITER_NOTDONE(ity):
        l = 0
        r = length - 1
        while l < r:
            x = (<DTYPE0_t*>((<char*>pid(ity)) + k*stride))[0]
            i = l
            j = r
            while 1:
                while (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0] < x: i += 1
                while x < (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0]: j -= 1
                if i <= j:
                    tmpi = (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0]
                    tmpj = (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0]
                    (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0] = tmpj
                    (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0] = tmpi
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        PyArray_ITER_NEXT(ity)
    return y


# argpartsort ---------------------------------------------------------------

def argpartsort(arr, int n, axis=-1):
    """
    Return indices that would partially sort an array.

    A partially sorted array is one in which the `n` smallest values appear
    (in any order) in the first `n` elements. The remaining largest elements
    are also unordered. Due to the algorithm used (Wirth's method), the nth
    smallest element is in its sorted position (at index `n-1`).

    Shuffling the input array may change the output. The only guarantee is
    that the first `n` elements will be the `n` smallest and the remaining
    element will appear in the remainder of the output.

    This functions is not protected against NaN. Therefore, you may get
    unexpected results if the input contains NaN.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    n : int
        The indices of the `n` smallest elements will appear in the first `n`
        elements of the output array along the given `axis`.
    axis : {int, None}, optional
        Axis along which the partial sort is performed. The default (axis=-1)
        is to sort along the last axis.

    Returns
    -------
    y : ndarray
        An array the same shape as the input array containing the indices
        that partially sort `arr` such that the `n` smallest elements will
        appear (unordered) in the first `n` elements.

    See Also
    --------
    bottleneck.partsort: Partial sorting of array elements along given axis.

    Notes
    -----
    Unexpected results may occur if the input array contains NaN.

    Examples
    --------
    Create a numpy array:

    >>> a = np.array([1, 0, 3, 4, 2])

    Find the indices that partially sort that array so that the first 3
    elements are the smallest 3 elements:

    >>> index = bn.argpartsort(a, n=3)
    >>> index
    array([0, 1, 4, 3, 2])

    Let's use the indices to partially sort the array (note, as in this
    example, that the smallest 3 elements may not be in order):

    >>> a[index]
    array([1, 0, 2, 4, 3])

    """
    try:
        return nonreducer_axis(arr, axis,
                               argpartsort_float64,
                               argpartsort_float32,
                               argpartsort_int64,
                               argpartsort_int32,
                               n)
    except TypeError:
        return slow.argpartsort(arr, n, axis)


cdef ndarray argpartsort_DTYPE0(ndarray a, int axis,
                                int a_ndim, np.npy_intp* y_dims, int n):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]

    cdef np.npy_intp i, j = 0, l, r, k = n-1
    cdef DTYPE0_t x, tmpi, tmpj
    cdef intp_t itmpi, itmpj
    cdef ndarray y = PyArray_Copy(a)
    cdef Py_ssize_t stride = y.strides[axis]
    cdef Py_ssize_t length = y.shape[axis]

    cdef ndarray index = PyArray_EMPTY(a_ndim, y_dims, NPY_intp, 0)
    cdef Py_ssize_t istride = index.strides[axis]
    cdef np.flatiter iti = PyArray_IterAllButAxis(index, &axis)
    while PyArray_ITER_NOTDONE(iti):
        for i in range(length):
            (<intp_t*>((<char*>pid(iti)) + i*istride))[0] = i
        PyArray_ITER_NEXT(iti)
    PyArray_ITER_RESET(iti)

    if length == 0:
        return index
    if (n < 1) or (n > length):
        raise ValueError("`n` (=%d) must be between 1 and %d, inclusive." %
                         (n, length))

    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    while PyArray_ITER_NOTDONE(ity):
        l = 0
        r = length - 1
        while l < r:
            x = (<DTYPE0_t*>((<char*>pid(ity)) + k*stride))[0]
            i = l
            j = r
            while 1:
                while (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0] < x: i += 1
                while x < (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0]: j -= 1
                if i <= j:
                    tmpi = (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0]
                    tmpj = (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0]
                    (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0] = tmpj
                    (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0] = tmpi
                    itmpi = (<intp_t*>((<char*>pid(iti)) + i*istride))[0]
                    itmpj = (<intp_t*>((<char*>pid(iti)) + j*istride))[0]
                    (<intp_t*>((<char*>pid(iti)) + i*istride))[0] = itmpj
                    (<intp_t*>((<char*>pid(iti)) + j*istride))[0] = itmpi
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        PyArray_ITER_NEXT(ity)
        PyArray_ITER_NEXT(iti)

    return index


# nanrankdata ---------------------------------------------------------------

def nanrankdata(arr, axis=None):
    """
    Ranks the data, dealing with ties and NaNs appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    NaNs in the input array are returned as NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the elements of the array are ranked. The default
        (axis=None) is to rank the elements of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`. The dtype is 'float64'.

    See also
    --------
    bottleneck.rankdata: Ranks the data, dealing with ties and appropriately.

    Examples
    --------
    >>> bn.nanrankdata([np.nan, 2, 2, 3])
    array([ nan,  1.5,  1.5,  3. ])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]])
    array([ nan,  1.5,  1.5,  3. ])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]], axis=0)
    array([[ nan,   1.],
           [  1.,   2.]])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]], axis=1)
    array([[ nan,   1.],
           [  1.,   2.]])

    """
    try:
        return nonreducer_axis(arr, axis,
                               nanrankdata_float64,
                               nanrankdata_float32,
                               rankdata_int64,
                               rankdata_int32)
    except TypeError:
        return slow.nanrankdata(arr, axis)


cdef ndarray nanrankdata_DTYPE0(ndarray a, int axis,
                                int a_ndim, np.npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64', 'float64', 'intp'], ['float32', 'float64', 'intp']]

    cdef Py_ssize_t j=0, k, idx, dupcount=0, i
    cdef DTYPE1_t old, new, averank, sumranks = 0
    cdef Py_ssize_t length = a.shape[axis]

    cdef np.flatiter ita = PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t astride = a.strides[axis]

    cdef ndarray ivec = PyArray_ArgSort(a, axis, NPY_QUICKSORT)
    cdef np.flatiter iti = PyArray_IterAllButAxis(ivec, &axis)
    cdef Py_ssize_t istride = ivec.strides[axis]

    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    if length == 0:
        PyArray_FillWithScalar(y, NAN)
        return y

    while PyArray_ITER_NOTDONE(ita):
        idx = (<DTYPE2_t*>((<char*>pid(iti)) + 0*istride))[0]
        old = (<DTYPE0_t*>((<char*>pid(ita)) + idx*astride))[0]
        sumranks = 0
        dupcount = 0
        for i in range(length - 1):
            sumranks += i
            dupcount += 1
            k = i + 1
            idx = (<DTYPE2_t*>((<char*>pid(iti)) + k*istride))[0]
            new = (<DTYPE0_t*>((<char*>pid(ita)) + idx*astride))[0]
            if old != new:
                if old == old:
                    averank = sumranks / dupcount + 1
                    for j in range(k - dupcount, k):
                        idx = (<DTYPE2_t*>((<char*>pid(iti)) + j*istride))[0]
                        (<DTYPE1_t*>((<char*>pid(ity)) + idx*ystride))[0] = averank
                else:
                    idx = (<DTYPE2_t*>((<char*>pid(iti)) + i*istride))[0]
                    (<DTYPE1_t*>((<char*>pid(ity)) + idx*ystride))[0] = NAN
                sumranks = 0
                dupcount = 0
            old = new
        sumranks += (length - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        if old == old:
            for j in range(length - dupcount, length):
                idx = (<DTYPE2_t*>((<char*>pid(iti)) + j*istride))[0]
                (<DTYPE1_t*>((<char*>pid(ity)) + idx*ystride))[0] = averank
        else:
            idx = (<DTYPE2_t*>((<char*>pid(iti)) + (length - 1)*istride))[0]
            (<DTYPE1_t*>((<char*>pid(ity)) + idx*ystride))[0] = NAN
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
        PyArray_ITER_NEXT(iti)
    return y


# rankdata ------------------------------------------------------------------

def rankdata(arr, axis=None):
    """
    Ranks the data, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the elements of the array are ranked. The default
        (axis=None) is to rank the elements of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`. The dtype is 'float64'.

    See also
    --------
    bottleneck.nanrankdata: Ranks the data dealing with ties and NaNs.

    Examples
    --------
    >>> bn.rankdata([0, 2, 2, 3])
    array([ 1. ,  2.5,  2.5,  4. ])
    >>> bn.rankdata([[0, 2], [2, 3]])
    array([ 1. ,  2.5,  2.5,  4. ])
    >>> bn.rankdata([[0, 2], [2, 3]], axis=0)
    array([[ 1.,  1.],
           [ 2.,  2.]])
    >>> bn.rankdata([[0, 2], [2, 3]], axis=1)
    array([[ 1.,  2.],
           [ 1.,  2.]])

    """
    try:
        return nonreducer_axis(arr, axis,
                               rankdata_float64,
                               rankdata_float32,
                               rankdata_int64,
                               rankdata_int32)
    except TypeError:
        return slow.rankdata(arr, axis)


cdef ndarray rankdata_DTYPE0(ndarray a, int axis,
                             int a_ndim, np.npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64', 'float64', 'intp'], ['float32', 'float64', 'intp'], ['int64', 'float64', 'intp'], ['int32', 'float64', 'intp']]

    cdef Py_ssize_t j=0, k, idx, dupcount=0, i
    cdef DTYPE1_t old, new, averank, sumranks = 0
    cdef Py_ssize_t length = a.shape[axis]

    cdef np.flatiter ita = PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t astride = a.strides[axis]

    cdef ndarray ivec = PyArray_ArgSort(a, axis, NPY_QUICKSORT)
    cdef np.flatiter iti = PyArray_IterAllButAxis(ivec, &axis)
    cdef Py_ssize_t istride = ivec.strides[axis]

    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    if length == 0:
        PyArray_FillWithScalar(y, NAN)
        return y

    while PyArray_ITER_NOTDONE(ita):
        idx = (<DTYPE2_t*>((<char*>pid(iti)) + 0*istride))[0]
        old = (<DTYPE0_t*>((<char*>pid(ita)) + idx*astride))[0]
        sumranks = 0
        dupcount = 0
        for i in range(length - 1):
            sumranks += i
            dupcount += 1
            k = i + 1
            idx = (<DTYPE2_t*>((<char*>pid(iti)) + k*istride))[0]
            new = (<DTYPE0_t*>((<char*>pid(ita)) + idx*astride))[0]
            if old != new:
                averank = sumranks / dupcount + 1
                for j in range(k - dupcount, k):
                    idx = (<DTYPE2_t*>((<char*>pid(iti)) + j*istride))[0]
                    (<DTYPE1_t*>((<char*>pid(ity)) + idx*ystride))[0] = averank
                sumranks = 0
                dupcount = 0
            old = new
        sumranks += (length - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in range(length - dupcount, length):
            idx = (<DTYPE2_t*>((<char*>pid(iti)) + j*istride))[0]
            (<DTYPE1_t*>((<char*>pid(ity)) + idx*ystride))[0] = averank
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
        PyArray_ITER_NEXT(iti)
    return y


# nonreduce_axis ------------------------------------------------------------

ctypedef ndarray (*nra_t)(ndarray, int, int, np.npy_intp*, int)


cdef ndarray nonreducer_axis(arr, axis,
                             nra_t nra_float64,
                             nra_t nra_float32,
                             nra_t nra_int64,
                             nra_t nra_int32,
                             int int_input=0):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # input array
    cdef int dtype = PyArray_TYPE(a)
    cdef int a_ndim

    # axis
    cdef int axis_int
    if axis is None:
        a = PyArray_Ravel(a, NPY_CORDER)
        axis_int = 0
        a_ndim = 1
    else:
        a_ndim = PyArray_NDIM(a)
        axis_int = <int>axis
        if axis_int < 0:
            axis_int += a_ndim
            if axis_int < 0:
                raise ValueError("axis(=%d) out of bounds" % axis)
        elif axis_int >= a_ndim:
            raise ValueError("axis(=%d) out of bounds" % axis)

    # output array
    cdef ndarray y
    cdef np.npy_intp *y_dims = np.PyArray_DIMS(a)

    # calc
    if dtype == NPY_float64:
        y = nra_float64(a, axis_int, a_ndim, y_dims, int_input)
    elif dtype == NPY_float32:
        y = nra_float32(a, axis_int, a_ndim, y_dims, int_input)
    elif dtype == NPY_int64:
        y = nra_int64(a, axis_int, a_ndim, y_dims, int_input)
    elif dtype == NPY_int32:
        y = nra_int32(a, axis_int, a_ndim, y_dims, int_input)
    else:
        raise TypeError("Unsupported dtype (%s)." % a.dtype)

    return y
