#include "bottleneck.h"

/*
 * All of the moving window functions contain the expression
 *
 *       ((PyArrayIterObject *)(ita))->coordinates[i]
 *
 * where ita is a numpy iterator. Using a numpy iterator in this way is
 * not part of the numpy C API. As such it could break in a new release
 * of numpy even if the release makes no changes to the C API.
 *
 * The reason we use it is to avoid the overhead of creating a second
 * iterator to iterate over the output array. It also speeds things up
 * when the input array is long and narrow and the moving window is
 * along the narrow dimension of the array.
 */

/* function pointers ----------------------------------------------------- */

typedef PyObject *(*move_t)(PyObject *, int, int, int, PyObject *, Py_ssize_t,
                            Py_ssize_t, int, npy_intp*);

/* prototypes ------------------------------------------------------------ */

static PyObject *
mover(char *name,
      PyObject *args,
      PyObject *kwds,
      move_t,
      move_t,
      move_t,
      move_t);

/* move_sum -------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
move_sum_DTYPE0(PyObject *a, int window, int min_count, int axis,
                PyObject *ita, Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape)
{
    Py_ssize_t i, count;
    npy_DTYPE0 asum, ai, aold, yi;
    PyObject *y = PyArray_EMPTY(ndim, shape, NPY_DTYPE0, 0);
    char *p0 = PyArray_BYTES((PyArrayObject *)y);
    char *p = p0;
    npy_intp *ystrides = PyArray_STRIDES((PyArrayObject *)y);
    npy_intp ystride = ystrides[axis];
    BN_BEGIN_ALLOW_THREADS
    while (PyArray_ITER_NOTDONE(ita)) {
        asum = 0;
        count = 0;
        for (i=0; i < min_count - 1; i++) {
            ai = *(npy_DTYPE0*)(PID(ita) + i*stride);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            *(npy_DTYPE0*)(p + i*ystride) = NAN;
        }
        for (i = min_count - 1; i < window; i++) {
            ai = *(npy_DTYPE0*)(PID(ita) + i*stride);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            if (count >= min_count) {
                yi = asum;
            }
            else {
                yi = NAN;
            }
            *(npy_DTYPE0*)(p + i*ystride) = yi;
        }
        for (i = window; i < length; i++) {
            ai = *(npy_DTYPE0*)(PID(ita) + i*stride);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            aold = *(npy_DTYPE0*)(PID(ita) + (i-window)*stride);
            if (aold == aold) {
                asum -= aold;
                count -= 1;
            }
            if (count >= min_count) {
                yi = asum;
            }
            else {
                yi = NAN;
            }
            *(npy_DTYPE0*)(p + i*ystride) = yi;
        }
        PyArray_ITER_NEXT(ita);
        p = p0;
        for (i=0; i < ndim; i++) {
            p += ((PyArrayIterObject *)(ita))->coordinates[i] * ystrides[i];
        }
    }
    BN_END_ALLOW_THREADS
    Py_DECREF(ita);
    return y;
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
move_sum_DTYPE0(PyObject *a, int window, int min_count, int axis,
                PyObject *ita, Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape)
{
    Py_ssize_t i;
    npy_DTYPE1 asum;
    PyObject *y = PyArray_EMPTY(ndim, shape, NPY_DTYPE1, 0);
    char *p0 = PyArray_BYTES((PyArrayObject *)y);
    char *p = p0;
    npy_intp *ystrides = PyArray_STRIDES((PyArrayObject *)y);
    npy_intp ystride = ystrides[axis];
    BN_BEGIN_ALLOW_THREADS
    while PyArray_ITER_NOTDONE(ita) {
        asum = 0;
        for (i=0; i < min_count - 1; i++) {
            asum += *(npy_DTYPE0*)(PID(ita) + i*stride);
            *(npy_DTYPE1*)(p + i*ystride) = NAN;
        }
        for (i = min_count - 1; i < window; i++) {
            asum += *(npy_DTYPE0*)(PID(ita) + i*stride);
            *(npy_DTYPE1*)(p + i*ystride) = (npy_DTYPE0)asum;
        }
        for (i = window; i < length; i++) {
            asum += *(npy_DTYPE0*)(PID(ita) + i*stride);
            asum -= *(npy_DTYPE0*)(PID(ita) + (i-window)*stride);
            *(npy_DTYPE1*)(p + i*ystride) = (npy_DTYPE0)asum;
        }
        PyArray_ITER_NEXT(ita);
        p = p0;
        for (i=0; i < ndim; i++) {
            p += ((PyArrayIterObject *)(ita))->coordinates[i] * ystrides[i];
        }
    }
    BN_END_ALLOW_THREADS
    Py_DECREF(ita);
    return y;
}
/* dtype end */


static PyObject *
move_sum(PyObject *self, PyObject *args, PyObject *kwds)
{
    return mover("move_sum",
                 args,
                 kwds,
                 move_sum_float64,
                 move_sum_float32,
                 move_sum_int64,
                 move_sum_int32);
}

/* python strings -------------------------------------------------------- */

PyObject *pystr_arr = NULL;
PyObject *pystr_window = NULL;
PyObject *pystr_min_count = NULL;
PyObject *pystr_axis = NULL;
PyObject *pystr_ddof = NULL;

static int
intern_strings(void) {
    pystr_arr = PyString_InternFromString("arr");
    pystr_window = PyString_InternFromString("window");
    pystr_min_count = PyString_InternFromString("min_count");
    pystr_axis = PyString_InternFromString("axis");
    pystr_ddof = PyString_InternFromString("ddof");
    return pystr_arr && pystr_window && pystr_min_count &&
           pystr_axis && pystr_ddof;
}

/* mover ----------------------------------------------------------------- */

static BN_INLINE int
parse_args(PyObject *args,
           PyObject *kwds,
           PyObject **arr_obj,
           PyObject **window_obj,
           PyObject **min_count_obj,
           PyObject **axis_obj)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (kwds) {
        int nkwds_found = 0;
        const Py_ssize_t nkwds = PyDict_Size(kwds);
        PyObject *tmp;
        switch (nargs) {
            case 3: *min_count_obj = PyTuple_GET_ITEM(args, 2);
            case 2: *window_obj = PyTuple_GET_ITEM(args, 1);
            case 1: *arr_obj = PyTuple_GET_ITEM(args, 0);
            case 0: break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
        switch (nargs) {
            case 0:
                *arr_obj = PyDict_GetItem(kwds, pystr_arr);
                if (*arr_obj == NULL) {
                    TYPE_ERR("Cannot find `arr` keyword input");
                    return 0;
                }
                nkwds_found += 1;
            case 1:
                *window_obj = PyDict_GetItem(kwds, pystr_window);
                if (*window_obj == NULL) {
                    TYPE_ERR("Cannot find `window` keyword input");
                    return 0;
                }
                nkwds_found++;
            case 2:
                tmp = PyDict_GetItem(kwds, pystr_min_count);
                if (tmp != NULL) {
                    *min_count_obj = tmp;
                    nkwds_found++;
                }
            case 3:
                tmp = PyDict_GetItem(kwds, pystr_axis);
                if (tmp != NULL) {
                    *axis_obj = tmp;
                    nkwds_found++;
                }
                break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
        if (nkwds_found != nkwds) {
            TYPE_ERR("wrong number of keyword arguments");
            return 0;
        }
        if (nargs + nkwds_found > 4) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    }
    else {
        switch (nargs) {
            case 4:
                *axis_obj = PyTuple_GET_ITEM(args, 3);
            case 3:
                *min_count_obj = PyTuple_GET_ITEM(args, 2);
            case 2:
                *window_obj = PyTuple_GET_ITEM(args, 1);
                *arr_obj = PyTuple_GET_ITEM(args, 0);
                break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
    }

    return 1;

}


static PyObject *
mover(char *name,
      PyObject *args,
      PyObject *kwds,
      move_t move_float64,
      move_t move_float32,
      move_t move_int64,
      move_t move_int32)
{

    int mc;
    int window;
    int axis;
    int dtype;
    int ndim;

    Py_ssize_t stride;
    Py_ssize_t length;

    PyObject *ita;
    PyArrayObject *a;

    PyObject *y;
    npy_intp *shape;

    PyObject *arr_obj = NULL;
    PyObject *window_obj = NULL;
    PyObject *min_count_obj = Py_None;
    PyObject *axis_obj = NULL;

    if (!parse_args(args, kwds,
                    &arr_obj, &window_obj, &min_count_obj, &axis_obj)) {
        return NULL;
    }

    /* convert to array if necessary */
    if PyArray_Check(arr_obj) {
        a = (PyArrayObject *)arr_obj;
    }
    else {
        a = (PyArrayObject *)PyArray_FROM_O(arr_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        return slow(name, args, kwds);
    }

    /* window */
    window = PyArray_PyIntAsInt(window_obj);
    if (error_converting(window)) {
        TYPE_ERR("`window` must be an integer");
        return NULL;
    }

    /* min_count */
    if (min_count_obj == Py_None) {
        mc = window;
    }
    else {
        mc = PyArray_PyIntAsInt(min_count_obj);
        if (error_converting(mc)) {
            TYPE_ERR("`min_count` must be an integer or None");
            return NULL;
        }
        if (mc > window) {
            PyErr_Format(PyExc_ValueError,
                         "min_count (%d) cannot be greater than window (%d)",
                         mc, window);
            return NULL;
        }
        else if (mc <= 0) {
            VALUE_ERR("`min_count` must be greater than zero.");
            return NULL;
        }
    }

    /* input array */
    dtype = PyArray_TYPE(a);
    ndim = PyArray_NDIM(a);

    /* defend against 0d beings */
    if (ndim == 0) {
        VALUE_ERR("moving window functions require ndim > 0");
        return NULL;
    }

    /* defend against the axis of negativity */
    if (axis_obj == NULL) {
        axis = ndim - 1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer");
            return NULL;
        }
        if (axis < 0) {
            axis += ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
                return NULL;
            }
        }
        else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            return NULL;
        }
    }

    ita = PyArray_IterAllButAxis((PyObject *)a, &axis);
    stride = PyArray_STRIDE(a, axis);
    shape = PyArray_SHAPE(a);
    length = shape[axis];

    if ((window < 1) || (window > length)) {
        PyErr_Format(PyExc_ValueError,
                     "Moving window (=%d) must between 1 and %zu, inclusive",
                     window, length);
        return NULL;
    }

    if (dtype == NPY_float64) {
        y = move_float64((PyObject *)a, window, mc, axis, ita, stride, length,
                         ndim, shape);
    }
    else if (dtype == NPY_float32) {
        y = move_float32((PyObject *)a, window, mc, axis, ita, stride, length,
                         ndim, shape);
    }
    else if (dtype == NPY_int64) {
        y = move_int64((PyObject *)a, window, mc, axis, ita, stride, length,
                       ndim, shape);
    }
    else if (dtype == NPY_int32) {
        y = move_int32((PyObject *)a, window, mc, axis, ita, stride, length,
                       ndim, shape);
    }
    else {
        y = slow(name, args, kwds);
    }

    return y;
}

/* docstrings ------------------------------------------------------------- */

static char move_doc[] =
"Bottleneck functions that reduce the input array along a specified axis.";

static char move_sum_doc[] =
/* MULTILINE STRING BEGIN
move_sum(arr, window, min_count=None, axis=-1)

Moving window sum along the specified axis, optionally ignoring NaNs.

This function cannot handle input arrays that contain Inf. When the
window contains Inf, the output will correctly be Inf. However, when Inf
moves out of the window, the remaining output values in the slice will
incorrectly be NaN.

Parameters
----------
arr : ndarray
    Input array. If `arr` is not an array, a conversion is attempted.
window : int
    The number of elements in the moving window.
min_count: {int, None}, optional
    If the number of non-NaN values in a window is less than `min_count`,
    then a value of NaN is assigned to the window. By default `min_count`
    is None, which is equivalent to setting `min_count` equal to `window`.
axis : int, optional
    The axis over which the window is moved. By default the last axis
    (axis=-1) is used. An axis of None is not allowed.

Returns
-------
y : ndarray
    The moving sum of the input array along the specified axis. The output
    has the same shape as the input.

Examples
--------
>>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_sum(arr, window=2)
array([ nan,   3.,   5.,  nan,  nan])
>>> bn.move_sum(arr, window=2, min_count=1)
array([ 1.,  3.,  5.,  3.,  5.])

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
move_methods[] = {
    {"move_sum", (PyCFunction)move_sum, VARKEY, move_sum_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
move_def = {
   PyModuleDef_HEAD_INIT,
   "move2",
   move_doc,
   -1,
   move_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_move2(void)
#else
#define RETVAL
initmove2(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&move_def);
    #else
        PyObject *m = Py_InitModule3("move2", move_methods, move_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
