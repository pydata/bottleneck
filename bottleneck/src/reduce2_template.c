#include "bottleneck.h"

/*
 * The iterator below is similar to the iterator in move.c. See that
 * file for a brief description.
 */
#define Y_INIT(dt0, dt1) \
    PyObject *y = PyArray_EMPTY(ndim - 1, yshape, dt0, 0); \
    dt1 *py = (dt1 *)PyArray_DATA((PyArrayObject *)y); \

#define INIT \
    Py_ssize_t i; \
    char *pa = PyArray_BYTES(a); \
    const npy_intp *astrides = PyArray_STRIDES(a); \
    const npy_intp *ashape = PyArray_SHAPE(a); \
    npy_intp index = 0; \
    npy_intp size = PyArray_SIZE(a); \
    npy_intp indices[ndim]; \
    memset(indices, 0, ndim * sizeof(npy_intp)); \
    if (length != 0) size /= length;

#define NEXT \
    for (i=ndim - 1; i >= 0; i--) { \
        if (i == axis) continue; \
        if (indices[i] < ashape[i] - 1) { \
            pa += astrides[i]; \
            indices[i]++; \
            break; \
        } \
        pa -= indices[i] * astrides[i]; \
        indices[i] = 0; \
    } \
    index++;

#define NOT_DONE index < size
#define AI(dt) *(dt*)(pa + i * stride)
#define FOR for (i = 0; i < length; i++)

/* function pointers ----------------------------------------------------- */

typedef PyObject *(*fall_t)(PyArrayObject *, int, Py_ssize_t, Py_ssize_t, int);
typedef PyObject *(*fone_t)(PyArrayObject *, int, Py_ssize_t, Py_ssize_t, int,
                            npy_intp*);

/* prototypes ------------------------------------------------------------ */

static PyObject *
reducer(char *name,
        PyObject *args,
        PyObject *kwds,
        fall_t fall_float64,
        fall_t fall_float32,
        fall_t fall_int64,
        fall_t fall_int32,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32,
        int ravel,
        int copy);

/* nansum ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nansum_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim)
{
    INIT
    npy_DTYPE0 ai, asum = 0;
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
            }
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}


static PyObject *
nansum_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    BN_BEGIN_ALLOW_THREADS
    npy_DTYPE0 ai, asum = 0;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR *py++ = 0;
    }
    else {
        while (NOT_DONE) {
            asum = 0;
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai == ai) {
                    asum += ai;
                }
            }
            *py++ = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nansum_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim)
{
    INIT
    npy_DTYPE0 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        FOR {
            asum += AI(npy_DTYPE0);
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}


static PyObject *
nansum_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    BN_BEGIN_ALLOW_THREADS
    npy_DTYPE0 asum = 0;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR *py++ = 0;
    }
    else {
        while (NOT_DONE) {
            asum = 0;
            FOR {
                asum += AI(npy_DTYPE0);
            }
            *py++ = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
nansum(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nansum",
                   args,
                   kwds,
                   nansum_all_float64,
                   nansum_all_float32,
                   nansum_all_int64,
                   nansum_all_int32,
                   nansum_one_float64,
                   nansum_one_float32,
                   nansum_one_int64,
                   nansum_one_int32,
                   0, 0);
}

/* nanmean ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmean_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                   Py_ssize_t length, int ndim)
{
    INIT
    Py_ssize_t count = 0;
    npy_DTYPE0 ai, asum = 0;
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (count > 0) {
        return PyFloat_FromDouble(asum / count);
    } else {
        return PyFloat_FromDouble(BN_NAN);
    }
}


static PyObject *
nanmean_one_DTYPE0(PyArrayObject *a,
                   int axis,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int ndim,
                   npy_intp* yshape)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    BN_BEGIN_ALLOW_THREADS
    Py_ssize_t count;
    npy_DTYPE0 ai, asum;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR *py++ = BN_NAN;
    }
    else {
        while (NOT_DONE) {
            asum = 0;
            count = 0;
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai == ai) {
                    asum += ai;
                    count += 1;
                }
            }
            if (count > 0) {
                asum /= count;
            } else {
                asum = BN_NAN;
            }
            *py++ = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
nanmean_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                   Py_ssize_t length, int ndim)
{
    INIT
    Py_ssize_t total_length = 0;
    npy_DTYPE1 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        FOR {
            asum += AI(npy_DTYPE0);
        }
        total_length += length;
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (total_length > 0) {
        return PyFloat_FromDouble(asum / total_length);
    } else {
        return PyFloat_FromDouble(BN_NAN);
    }
}


static PyObject *
nanmean_one_DTYPE0(PyArrayObject *a,
                   int axis,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int ndim,
                   npy_intp* yshape)
{
    Y_INIT(NPY_DTYPE1, npy_DTYPE1)
    INIT
    BN_BEGIN_ALLOW_THREADS
    npy_DTYPE1 asum = 0;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR *py++ = BN_NAN;
    }
    else {
        while (NOT_DONE) {
            asum = 0;
            FOR {
                asum += AI(npy_DTYPE0);
            }
            if (length > 0) {
                asum /= length;
            } else {
                asum = BN_NAN;
            }
            *py++ = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
nanmean(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nanmean",
                   args,
                   kwds,
                   nanmean_all_float64,
                   nanmean_all_float32,
                   nanmean_all_int64,
                   nanmean_all_int32,
                   nanmean_one_float64,
                   nanmean_one_float32,
                   nanmean_one_int64,
                   nanmean_one_int32,
                   0, 0);
}

/* nanmin ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmin_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim)
{
    INIT
    npy_DTYPE0 ai, amin = BN_INFINITY;
    int allnan = 1;
    if (PyArray_SIZE(a) == 0) {
        VALUE_ERR("numpy.nanmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai <= amin) {
                amin = ai;
                allnan = 0;
            }
        }
        NEXT
    }
    if (allnan) {
        amin = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(amin);
}


static PyObject *
nanmin_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    npy_DTYPE0 ai, amin = 0;
    int allnan;
    if (length == 0) {
        VALUE_ERR("numpy.nanmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        amin = BN_INFINITY;
        allnan = 1;
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai <= amin) {
                amin = ai;
                allnan = 0;
            }
        }
        if (allnan) {
            amin = BN_NAN;
        }
        *py++ = amin;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nanmin_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim)
{
    INIT
    npy_DTYPE0 ai, amin = NPY_MAX_DTYPE0;
    if (PyArray_SIZE(a) == 0) {
        VALUE_ERR("numpy.nanmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai <= amin) {
                amin = ai;
            }
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(amin);
}


static PyObject *
nanmin_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    npy_DTYPE0 ai, amin;
    if (length == 0) {
        VALUE_ERR("numpy.nanmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        amin = NPY_MAX_DTYPE0;
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai <= amin) {
                amin = ai;
            }
        }
        *py++ = amin;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
nanmin(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nanmin",
                   args,
                   kwds,
                   nanmin_all_float64,
                   nanmin_all_float32,
                   nanmin_all_int64,
                   nanmin_all_int32,
                   nanmin_one_float64,
                   nanmin_one_float32,
                   nanmin_one_int64,
                   nanmin_one_int32,
                   0, 0);
}

/* nanmax ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmax_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim)
{
    INIT
    npy_DTYPE0 ai, amax = -BN_INFINITY;
    int allnan = 1;
    if (PyArray_SIZE(a)== 0) {
        VALUE_ERR("numpy.nanmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai >= amax) {
                amax = ai;
                allnan = 0;
            }
        }
        NEXT
    }
    if (allnan) {
        amax = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(amax);
}


static PyObject *
nanmax_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    npy_DTYPE0 ai, amax = 0;
    int allnan;
    if (length == 0) {
        VALUE_ERR("numpy.nanmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        amax = -BN_INFINITY;
        allnan = 1;
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai >= amax) {
                amax = ai;
                allnan = 0;
            }
        }
        if (allnan) {
            amax = BN_NAN;
        }
        *py++ = amax;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nanmax_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim)
{
    INIT
    npy_DTYPE0 ai, amax = NPY_MIN_DTYPE0;
    if (PyArray_SIZE(a)== 0) {
        VALUE_ERR("numpy.nanmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai >= amax) {
                amax = ai;
            }
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(amax);
}


static PyObject *
nanmax_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    npy_DTYPE0 ai, amax;
    if (length == 0) {
        VALUE_ERR("numpy.nanmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    while (NOT_DONE) {
        amax = NPY_MIN_DTYPE0;
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai >= amax) {
                amax = ai;
            }
        }
        *py++ = amax;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
nanmax(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nanmax",
                   args,
                   kwds,
                   nanmax_all_float64,
                   nanmax_all_float32,
                   nanmax_all_int64,
                   nanmax_all_int32,
                   nanmax_one_float64,
                   nanmax_one_float32,
                   nanmax_one_int64,
                   nanmax_one_int32,
                   0, 0);
}

/* python strings -------------------------------------------------------- */

PyObject *pystr_arr = NULL;
PyObject *pystr_axis = NULL;
PyObject *pystr_ddof = NULL;

static int
intern_strings(void) {
    pystr_arr = PyString_InternFromString("arr");
    pystr_axis = PyString_InternFromString("axis");
    pystr_ddof = PyString_InternFromString("ddof");
    return pystr_arr && pystr_axis && pystr_ddof;
}

/* reducer --------------------------------------------------------------- */

static BN_INLINE int
parse_args(PyObject *args,
           PyObject *kwds,
           PyObject **arr,
           PyObject **axis)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (kwds) {
        const Py_ssize_t nkwds = PyDict_Size(kwds);
        if (nkwds == 1) {
            if (nargs == 0) {
                *arr = PyDict_GetItem(kwds, pystr_arr);
                if (!*arr) {
                    TYPE_ERR("can't find `arr` input");
                    return 0;
                }
            }
            else {
                *axis = PyDict_GetItem(kwds, pystr_axis);
                if (!*axis) {
                    TYPE_ERR("can't find `axis` input");
                    return 0;
                }
                if (nargs == 1) {
                    *arr = PyTuple_GET_ITEM(args, 0);
                }
                else {
                    TYPE_ERR("wrong number of inputs");
                    return 0;
                }
            }
        }
        else if (nkwds == 2) {
            if (nargs != 0) {
                TYPE_ERR("wrong number of inputs");
                return 0;
            }
            *arr = PyDict_GetItem(kwds, pystr_arr);
            if (!*arr) {
                TYPE_ERR("can't find `arr` input");
                return 0;
            }
            *axis = PyDict_GetItem(kwds, pystr_axis);
            if (!*axis) {
                TYPE_ERR("can't find `axis` input");
                return 0;
            }
        }
        else {
            TYPE_ERR("wrong number of inputs");
            return 0;
        }
    }
    else if (nargs == 1) {
        *arr = PyTuple_GET_ITEM(args, 0);
    }
    else if (nargs == 2) {
        *arr = PyTuple_GET_ITEM(args, 0);
        *axis = PyTuple_GET_ITEM(args, 1);
    }
    else {
        TYPE_ERR("wrong number of inputs");
        return 0;
    }

    return 1;

}


static PyObject *
reducer(char *name,
        PyObject *args,
        PyObject *kwds,
        fall_t fall_float64,
        fall_t fall_float32,
        fall_t fall_int64,
        fall_t fall_int32,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32,
        int ravel,
        int copy)
{

    int ndim;
    int reduce_all = 0;
    int axis;
    int dtype;

    Py_ssize_t i;
    Py_ssize_t j = 0;
    Py_ssize_t stride;
    Py_ssize_t length;

    npy_intp *shape;
    npy_intp *strides;

    PyArrayObject *a;

    PyObject *arr_obj;
    PyObject *axis_obj = Py_None;

    if (!parse_args(args, kwds, &arr_obj, &axis_obj)) {
        return NULL;
    }

    /* convert to array if necessary */
    if PyArray_Check(arr_obj) {
        a = (PyArrayObject *)arr_obj;
    } else {
        a = (PyArrayObject *)PyArray_FROM_O(arr_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        return slow(name, args, kwds);
    }

    /* input array
     TODO
    if (copy == 1) {
        a = PyArray_Copy(a);
    }
    */

    ndim = PyArray_NDIM(a);

    /* defend against 0d beings */
    if (ndim == 0) {
        if (axis_obj == Py_None ||
            axis_obj == PyInt_FromLong(0) ||
            axis_obj == PyInt_FromLong(-1))
            return slow(name, args, kwds);
        else {
            VALUE_ERR("axis out of bounds for 0d input");
            return NULL;
        }
    }

    /* does user want to reduce over all axes? */
    if (axis_obj == Py_None) {
        reduce_all = 1;
        axis = -1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer or None");
            return NULL;
        }
        if (axis < 0) {
            axis += ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis);
                return NULL;
            }
        }
        else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            return NULL;
        }
        if (ndim == 1 && axis == 0) {
            reduce_all = 1;
        }
    }

    dtype = PyArray_TYPE(a);
    shape = PyArray_SHAPE(a);
    strides = PyArray_STRIDES(a);

    if (reduce_all == 1) {
        /* we are reducing the array along all axes */
        if (ravel == 0) {
            axis = 0;
            stride = strides[0];
            for (i=1; i < ndim; i++) {
                if (strides[i] < stride) {
                    axis = i;
                }
            }
            stride = strides[axis];
            length = shape[axis];
        }
        else {
            /* TODO a = PyArray_Ravel(a, NPY_ANYORDER);*/
            axis = 0; /* TODO find min stride axis */
            stride = PyArray_STRIDE(a, 0);
            length = PyArray_SIZE(a);
        }
        if (dtype == NPY_FLOAT64) {
            return fall_float64(a, axis, stride, length, ndim);
        }
        else if (dtype == NPY_FLOAT32) {
            return fall_float32(a, axis, stride, length, ndim);
        }
        else if (dtype == NPY_INT64) {
            return fall_int64(a, axis, stride, length, ndim);
        }
        else if (dtype == NPY_INT32) {
            return fall_int32(a, axis, stride, length, ndim);
        }
        else {
            return slow(name, args, kwds);
        }
    }
    else {

        /* we are reducing an array with ndim > 1 over a single axis */

        npy_intp yshape[ndim - 1];

        /* shape of output */
        for (i=0; i < ndim; i++) {
            if (i != axis) yshape[j++] = shape[i];
        }

        stride = strides[axis];
        length = shape[axis];

        if (dtype == NPY_FLOAT64) {
            return fone_float64(a, axis, stride, length, ndim, yshape);
        }
        else if (dtype == NPY_FLOAT32) {
            return fone_float32(a, axis, stride, length, ndim, yshape);
        }
        else if (dtype == NPY_INT64) {
            return fone_int64(a, axis, stride, length, ndim, yshape);
        }
        else if (dtype == NPY_INT32) {
            return fone_int32(a, axis, stride, length, ndim, yshape);
        }
        else {
            return slow(name, args, kwds);
        }
    }

}

/* docstrings ------------------------------------------------------------- */

static char reduce_doc[] =
"Bottleneck functions that reduce the input array along a specified axis.";

static char nansum_doc[] =
/* MULTILINE STRING BEGIN
nansum(arr, axis=None)

Sum of array elements along given axis treating NaNs as zero.

The data type (dtype) of the output is the same as the input. On 64-bit
operating systems, 32-bit input is NOT upcast to 64-bit accumulator and
return values.

Parameters
----------
arr : array_like
    Array containing numbers whose sum is desired. If `arr` is not an
    array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the sum is computed. The default (axis=None) is to
    compute the sum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned.

Notes
-----
No error is raised on overflow.

If positive or negative infinity are present the result is positive or
negative infinity. But if both positive and negative infinity are present,
the result is Not A Number (NaN).

Examples
--------
>>> bn.nansum(1)
1
>>> bn.nansum([1])
1
>>> bn.nansum([1, np.nan])
1.0
>>> a = np.array([[1, 1], [1, np.nan]])
>>> bn.nansum(a)
3.0
>>> bn.nansum(a, axis=0)
array([ 2.,  1.])

When positive infinity and negative infinity are present:

>>> bn.nansum([1, np.nan, np.inf])
inf
>>> bn.nansum([1, np.nan, np.NINF])
-inf
>>> bn.nansum([1, np.nan, np.inf, np.NINF])
nan

MULTILINE STRING END */

static char nanmean_doc[] =
/* MULTILINE STRING BEGIN
nanmean(arr, axis=None)

Mean of array elements along given axis ignoring NaNs.

`float64` intermediate and return values are used for integer inputs.

Parameters
----------
arr : array_like
    Array containing numbers whose mean is desired. If `arr` is not an
    array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the means are computed. The default (axis=None) is to
    compute the mean of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned.
    `float64` intermediate and return values are used for integer inputs.

See also
--------
bottleneck.nanmedian: Median along specified axis, ignoring NaNs.

Notes
-----
No error is raised on overflow. (The sum is computed and then the result
is divided by the number of non-NaN elements.)

If positive or negative infinity are present the result is positive or
negative infinity. But if both positive and negative infinity are present,
the result is Not A Number (NaN).

Examples
--------
>>> bn.nanmean(1)
1.0
>>> bn.nanmean([1])
1.0
>>> bn.nanmean([1, np.nan])
1.0
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.nanmean(a)
2.0
>>> bn.nanmean(a, axis=0)
array([ 1.,  4.])

When positive infinity and negative infinity are present:

>>> bn.nanmean([1, np.nan, np.inf])
inf
>>> bn.nanmean([1, np.nan, np.NINF])
-inf
>>> bn.nanmean([1, np.nan, np.inf, np.NINF])
nan

MULTILINE STRING END */

static char nanmin_doc[] =
/* MULTILINE STRING BEGIN
nanmin(arr, axis=None)

Minimum values along specified axis, ignoring NaNs.

When all-NaN slices are encountered, NaN is returned for that slice.

Parameters
----------
arr : array_like
    Input array. If `arr` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the minimum is computed. The default (axis=None) is
    to compute the minimum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned. The
    same dtype as `arr` is returned.

See also
--------
bottleneck.nanmax: Maximum along specified axis, ignoring NaNs.
bottleneck.nanargmin: Indices of minimum values along axis, ignoring NaNs.

Examples
--------
>>> bn.nanmin(1)
1
>>> bn.nanmin([1])
1
>>> bn.nanmin([1, np.nan])
1.0
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.nanmin(a)
1.0
>>> bn.nanmin(a, axis=0)
array([ 1.,  4.])

MULTILINE STRING END */

static char nanmax_doc[] =
/* MULTILINE STRING BEGIN
nanmax(arr, axis=None)

Maximum values along specified axis, ignoring NaNs.

When all-NaN slices are encountered, NaN is returned for that slice.

Parameters
----------
arr : array_like
    Input array. If `arr` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the maximum is computed. The default (axis=None) is
    to compute the maximum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned. The
    same dtype as `arr` is returned.

See also
--------
bottleneck.nanmin: Minimum along specified axis, ignoring NaNs.
bottleneck.nanargmax: Indices of maximum values along axis, ignoring NaNs.

Examples
--------
>>> bn.nanmax(1)
1
>>> bn.nanmax([1])
1
>>> bn.nanmax([1, np.nan])
1.0
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.nanmax(a)
4.0
>>> bn.nanmax(a, axis=0)
array([ 1.,  4.])

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
reduce_methods[] = {
    {"nansum", (PyCFunction)nansum, VARKEY, nansum_doc},
    {"nanmean", (PyCFunction)nanmean, VARKEY, nanmean_doc},
    {"nanmin", (PyCFunction)nanmin, VARKEY, nanmin_doc},
    {"nanmax", (PyCFunction)nanmax, VARKEY, nanmax_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
reduce_def = {
   PyModuleDef_HEAD_INIT,
   "reduce2",
   reduce_doc,
   -1,
   reduce_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_reduce2(void)
#else
#define RETVAL
initreduce2(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&reduce_def);
    #else
        PyObject *m = Py_InitModule3("reduce2", reduce_methods, reduce_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
