"var"

cdef dict var_dict = {}

#     Dim dtype axis
var_dict[(1, f64, 0)] = var_1d_float64_axis0
var_dict[(1, f64, N)] = var_1d_float64_axis0
var_dict[(2, f64, 0)] = var_2d_float64_axis0
var_dict[(2, f64, 1)] = var_2d_float64_axis1
var_dict[(2, f64, N)] = var_2d_float64_axisNone
var_dict[(3, f64, 0)] = var_3d_float64_axis0
var_dict[(3, f64, 1)] = var_3d_float64_axis1
var_dict[(3, f64, 2)] = var_3d_float64_axis2
var_dict[(3, f64, N)] = var_3d_float64_axisNone

var_dict[(1, i32, 0)] = var_1d_int32_axis0
var_dict[(1, i32, N)] = var_1d_int32_axis0
var_dict[(2, i32, 0)] = var_2d_int32_axis0
var_dict[(2, i32, 1)] = var_2d_int32_axis1
var_dict[(2, i32, N)] = var_2d_int32_axisNone
var_dict[(3, i32, 0)] = var_3d_int32_axis0
var_dict[(3, i32, 1)] = var_3d_int32_axis1
var_dict[(3, i32, 2)] = var_3d_int32_axis2
var_dict[(3, i32, N)] = var_3d_int32_axisNone

var_dict[(1, i64, 0)] = var_1d_int64_axis0
var_dict[(1, i64, N)] = var_1d_int64_axis0
var_dict[(2, i64, 0)] = var_2d_int64_axis0
var_dict[(2, i64, 1)] = var_2d_int64_axis1
var_dict[(2, i64, N)] = var_2d_int64_axisNone
var_dict[(3, i64, 0)] = var_3d_int64_axis0
var_dict[(3, i64, 1)] = var_3d_int64_axis1
var_dict[(3, i64, 2)] = var_3d_int64_axis2
var_dict[(3, i64, N)] = var_3d_int64_axisNone


def var(arr, axis=None, int ddof=0):
    """
    Variance along the specified axis, ignoring NaNs.
    
    """
    func, arr = var_selector(arr, axis)
    return func(arr, ddof)

def var_selector(arr, axis):
    "Return var function that matches `arr` and `axis` and return `arr`."
    cdef np.ndarray a = np.array(arr, copy=False)
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    cdef int size = a.size
    if axis != None:
        if axis < 0:
            axis += ndim
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = var_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func, a

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def var_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a, int ddof=0):
    "var of 1d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0]
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        ai = a[i]
        if ai == ai:
            asum += ai
    amean = asum / a0
    asum = 0
    for i in range(a0):
        ai = a[i]
        if ai == ai:
            ai -= amean
            asum += (ai * ai)
    return np.float64(asum / (a0 - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def var_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a, int ddof=0):
    "var of 1d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0]
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        ai = a[i]
        if ai == ai:
            asum += ai
    amean = asum / a0
    asum = 0
    for i in range(a0):
        ai = a[i]
        if ai == ai:
            ai -= amean
            asum += (ai * ai)
    return np.float64(asum / (a0 - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def var_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a, int ddof=0):
    "var of 1d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], count = 0
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        amean = asum / count
        asum = 0
        for i in range(a0):
            ai = a[i]
            if ai == ai:
                ai -= amean
                asum += (ai * ai)
        return np.float64(asum / (count - ddof))
    else:
        return np.float64(NAN)

# Two dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a1, dtype=np.float64)
    for j in range(a1):
        asum = 0
        for i in range(a0):
            asum += a[i,j]
        amean = asum / a0
        asum = 0
        for i in range(a0):
            ai = a[i,j]
            ai -= amean
            asum += (ai * ai)
        y[j] = asum / (a0 - ddof)
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.int32 along axis=1."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a0, dtype=np.float64)
    for i in range(a0):
        asum = 0
        for j in range(a1):
            asum += a[i,j]
        amean = asum / a1
        asum = 0
        for j in range(a1):
            ai = a[i,j]
            ai -= amean
            asum += (ai * ai)
        y[i] = asum / (a1 - ddof)
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], a01 = a0 * a1
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        for j in range(a1):
            asum += a[i,j]
    amean = asum / a01
    asum = 0
    for i in range(a0):
        for j in range(a1):
            ai = a[i,j]
            ai -= amean
            asum += (ai * ai)
    return np.float64(asum / (a01 - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a1, dtype=np.float64)
    for j in range(a1):
        asum = 0
        for i in range(a0):
            asum += a[i,j]
        amean = asum / a0
        asum = 0
        for i in range(a0):
            ai = a[i,j]
            ai -= amean
            asum += (ai * ai)
        y[j] = asum / (a0 - ddof)
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.int64 along axis=1."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a0, dtype=np.float64)
    for i in range(a0):
        asum = 0
        for j in range(a1):
            asum += a[i,j]
        amean = asum / a1
        asum = 0
        for j in range(a1):
            ai = a[i,j]
            ai -= amean
            asum += (ai * ai)
        y[i] = asum / (a1 - ddof)
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], a01 = a0 * a1
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        for j in range(a1):
            asum += a[i,j]
    amean = asum / a01
    asum = 0
    for i in range(a0):
        for j in range(a1):
            ai = a[i,j]
            ai -= amean
            asum += (ai * ai)
    return np.float64(asum / (a01 - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a1, dtype=np.float64)
    for j in range(a1):
        asum = 0
        count = 0
        for i in range(a0):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
        if count > 0:
            amean = asum / count
            asum = 0
            for i in range(a0):
                ai = a[i,j]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
            y[j] = asum / (count - ddof)
        else:
            y[j] = NAN
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a0, dtype=np.float64)
    for i in range(a0):
        asum = 0
        count = 0
        for j in range(a1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
        if count > 0:
            amean = asum / count
            asum = 0
            for j in range(a1):
                ai = a[i,j]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
            y[i] = asum / (count - ddof)
        else:
            y[i] = NAN
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a, int ddof=0):
    "var of 2d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count = 0
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        for j in range(a1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
    if count > 0:
        amean = asum / count
        asum = 0
        for i in range(a0):
            for j in range(a1):
                ai = a[i,j]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
        return np.float64(asum / (count - ddof))
    else:
        return np.float64(NAN)

# Three dimensional ---------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a1, a2),
                                                       dtype=np.float64)
    for j in range(a1):
        for k in range(a2):
            asum = 0
            for i in range(a0):
                asum += a[i,j,k]
            amean = asum / a0
            asum = 0
            for i in range(a0):
                ai = a[i,j,k]
                ai -= amean
                asum += (ai * ai)
            y[j,k] = asum / (a0 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, a2),
                                                       dtype=np.float64)
    for i in range(a0):
        for k in range(a2):
            asum = 0
            for j in range(a1):
                asum += a[i,j,k]
            amean = asum / a1
            asum = 0
            for j in range(a1):
                ai = a[i,j,k]
                ai -= amean
                asum += (ai * ai)
            y[i,k] = asum / (a1 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.int32 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, a1),
                                                       dtype=np.float64)
    for i in range(a0):
        for j in range(a1):
            asum = 0
            for k in range(a2):
                asum += a[i,j,k]
            amean = asum / a2
            asum = 0
            for k in range(a2):
                ai = a[i,j,k]
                ai -= amean
                asum += (ai * ai)
            y[i,j] = asum / (a2 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2]
    cdef int a012 = a0 * a1 * a2
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        for j in range(a1):
            for k in range(a2):
                asum += a[i,j,k]
    amean = asum / a012
    asum = 0
    for i in range(a0):
        for j in range(a1):
            for k in range(a2):
                ai = a[i,j,k]
                ai -= amean
                asum += (ai * ai)
    return np.float64(asum / (a012 - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a1, a2),
                                                       dtype=np.float64)
    for j in range(a1):
        for k in range(a2):
            asum = 0
            for i in range(a0):
                asum += a[i,j,k]
            amean = asum / a0
            asum = 0
            for i in range(a0):
                ai = a[i,j,k]
                ai -= amean
                asum += (ai * ai)
            y[j,k] = asum / (a0 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, a2),
                                                       dtype=np.float64)
    for i in range(a0):
        for k in range(a2):
            asum = 0
            for j in range(a1):
                asum += a[i,j,k]
            amean = asum / a1
            asum = 0
            for j in range(a1):
                ai = a[i,j,k]
                ai -= amean
                asum += (ai * ai)
            y[i,k] = asum / (a1 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.int64 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2]
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, a1),
                                                       dtype=np.float64)
    for i in range(a0):
        for j in range(a1):
            asum = 0
            for k in range(a2):
                asum += a[i,j,k]
            amean = asum / a2
            asum = 0
            for k in range(a2):
                ai = a[i,j,k]
                ai -= amean
                asum += (ai * ai)
            y[i,j] = asum / (a2 - ddof)
    return y 


@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2]
    cdef int a012 = a0 * a1 * a2
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        for j in range(a1):
            for k in range(a2):
                asum += a[i,j,k]
    amean = asum / a012
    asum = 0
    for i in range(a0):
        for j in range(a1):
            for k in range(a2):
                ai = a[i,j,k]
                ai -= amean
                asum += (ai * ai)
    return np.float64(asum / (a012 - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a1, a2),
                                                       dtype=np.float64)
    for j in range(a1):
        for k in range(a2):
            asum = 0
            count = 0
            for i in range(a0):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:
                amean = asum / count
                asum = 0
                for i in range(a0):
                    ai = a[i,j,k]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[j,k] = asum / (count - ddof)
            else:
                y[j,k] = NAN
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, a2),
                                                       dtype=np.float64)
    for i in range(a0):
        for k in range(a2):
            asum = 0
            count = 0
            for j in range(a1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:
                amean = asum / count
                asum = 0
                for j in range(a1):
                    ai = a[i,j,k]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[i,k] = asum / (count - ddof)
            else:
                y[i,k] = NAN
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.float64 along axis=2."
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef np.float64_t asum, amean, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, a1),
                                                       dtype=np.float64)
    for i in range(a0):
        for j in range(a1):
            asum = 0
            count = 0
            for k in range(a2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:
                amean = asum / count
                asum = 0
                for k in range(a2):
                    ai = a[i,j,k]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[i,j] = asum / (count - ddof)
            else:
                y[i,j] = NAN
    return y            

@cython.boundscheck(False)
@cython.wraparound(False)
def var_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a, int ddof=0):
    "var of 3d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count = 0
    cdef np.float64_t asum = 0, amean, ai
    for i in range(a0):
        for j in range(a1):
            for k in range(a2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
    if count > 0:
        amean = asum / count
        asum = 0
        for i in range(a0):
            for j in range(a1):
                for k in range(a2):
                    ai = a[i,j,k]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
        return np.float64(asum / (count - ddof))
    else:
        return np.float64(NAN)

