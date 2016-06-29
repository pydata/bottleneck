#include "bottleneck.h"

/* function pointers ----------------------------------------------------- */

typedef PyObject *(*fall_ss_t)(char *, npy_intp, npy_intp);
typedef PyObject *(*fall_t)(PyObject *, Py_ssize_t, Py_ssize_t);
typedef PyObject *(*fone_t)(PyObject *, Py_ssize_t, Py_ssize_t, int,
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
        fall_ss_t fall_ss_float64,
        fall_ss_t fall_ss_float32,
        fall_ss_t fall_ss_int64,
        fall_ss_t fall_ss_int32,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32,
        int ravel,
        int copy);

/* nansum ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nansum_all_ss_DTYPE0(char *p, npy_intp stride, npy_intp length)
{
    Py_ssize_t i;
    npy_DTYPE0 ai;
    npy_DTYPE0 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    for (i = 0; i < length; i++) {
        ai = *(npy_DTYPE0*)(p + i * stride);
        if (ai == ai) {
            asum += ai;
        }
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nansum_all_ss_DTYPE0(char *p, npy_intp stride, npy_intp length)
{
    Py_ssize_t i;
    npy_DTYPE0 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    for (i = 0; i < length; i++) {
        asum += *(npy_DTYPE0*)(p + i * stride);
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}
/* dtype end */


/* dtype = [['float64'], ['float32']] */
static PyObject *
nansum_all_DTYPE0(PyObject *ita, Py_ssize_t stride, Py_ssize_t length)
{
    Py_ssize_t i;
    npy_DTYPE0 asum = 0, ai;
    BN_BEGIN_ALLOW_THREADS
    while (PyArray_ITER_NOTDONE(ita)) {
        for (i = 0; i < length; i++) {
            ai = *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ita)) + i*stride);
            if (ai == ai) {
                asum += ai;
            }
        }
        PyArray_ITER_NEXT(ita);
    }
    Py_DECREF(ita);
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nansum_all_DTYPE0(PyObject *ita, Py_ssize_t stride, Py_ssize_t length)
{
    Py_ssize_t i;
    npy_DTYPE0 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    while (PyArray_ITER_NOTDONE(ita)) {
        for (i = 0; i < length; i++) {
            asum += *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ita)) + i*stride);
        }
        PyArray_ITER_NEXT(ita);
    }
    Py_DECREF(ita);
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}
/* dtype end */


/* dtype = [['float64'], ['float32']] */
static PyObject *
nansum_one_DTYPE0(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int ndim,
                   npy_intp* y_dims)
{
    Py_ssize_t i;
    npy_DTYPE0 asum = 0, ai;
    PyObject *y = PyArray_EMPTY(ndim - 1, y_dims, NPY_DTYPE0, 0);
    PyObject *ity = PyArray_IterNew(y);
    BN_BEGIN_ALLOW_THREADS
    if (length == 0) {
        while (PyArray_ITER_NOTDONE(ity)) {
            *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ity))) = asum;
            PyArray_ITER_NEXT(ity);
        }
    }
    else {
        while (PyArray_ITER_NOTDONE(ita)) {
            asum = 0;
            for (i = 0; i < length; i++) {
                ai = (*(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ita)) +
                                                                  i*stride));
                if (ai == ai) {
                    asum += ai;
                }
            }
            *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ity))) = asum;
            PyArray_ITER_NEXT(ita);
            PyArray_ITER_NEXT(ity);
        }
    }
    Py_DECREF(ita);
    Py_DECREF(ity);
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nansum_one_DTYPE0(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int ndim,
                   npy_intp* y_dims)
{
    Py_ssize_t i;
    npy_DTYPE0 asum = 0;
    PyObject *y = PyArray_EMPTY(ndim - 1, y_dims, NPY_DTYPE0, 0);
    PyObject *ity = PyArray_IterNew(y);
    if (length == 0) {
        while (PyArray_ITER_NOTDONE(ity)) {
            *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ity))) = asum;
            PyArray_ITER_NEXT(ity);
        }
    }
    else {
        while (PyArray_ITER_NOTDONE(ita)) {
            asum = 0;
            for (i = 0; i < length; i++) {
                asum += (*(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ita)) +
                                                                  i*stride));
            }
            *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ity))) = asum;
            PyArray_ITER_NEXT(ita);
            PyArray_ITER_NEXT(ity);
        }
    }
    Py_DECREF(ita);
    Py_DECREF(ity);
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
                   nansum_all_ss_float64,
                   nansum_all_ss_float32,
                   nansum_all_ss_int64,
                   nansum_all_ss_int32,
                   nansum_one_float64,
                   nansum_one_float32,
                   nansum_one_int64,
                   nansum_one_int32,
                   0, 0);
}

/* nanmean ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmean_all_ss_DTYPE0(char *p, npy_intp stride, npy_intp length)
{
    Py_ssize_t i;
    Py_ssize_t count = 0;
    npy_DTYPE0 ai;
    npy_DTYPE0 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    for (i = 0; i < length; i++) {
        ai = *(npy_DTYPE0*)(p + i * stride);
        if (ai == ai) {
            asum += ai;
            count += 1;
        }
    }
    BN_END_ALLOW_THREADS
    if (count > 0) { 
        return PyFloat_FromDouble(asum / count);
    } else {
        return PyFloat_FromDouble(NAN);
    }
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
nanmean_all_ss_DTYPE0(char *p, npy_intp stride, npy_intp length)
{
    Py_ssize_t i;
    npy_DTYPE1 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    for (i = 0; i < length; i++) {
        asum += *(npy_DTYPE0*)(p + i * stride);
    }
    BN_END_ALLOW_THREADS
    if (length > 0) { 
        return PyFloat_FromDouble(asum / length);
    } else {
        return PyFloat_FromDouble(NAN);
    }
}
/* dtype end */


/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmean_all_DTYPE0(PyObject *ita, Py_ssize_t stride, Py_ssize_t length)
{
    Py_ssize_t i;
    Py_ssize_t count = 0;
    npy_DTYPE0 asum = 0, ai;
    BN_BEGIN_ALLOW_THREADS
    while (PyArray_ITER_NOTDONE(ita)) {
        for (i = 0; i < length; i++) {
            ai = *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ita)) + i*stride);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
        }
        PyArray_ITER_NEXT(ita);
    }
    Py_DECREF(ita);
    BN_END_ALLOW_THREADS
    if (count > 0) { 
        return PyFloat_FromDouble(asum / count);
    } else {
        return PyFloat_FromDouble(NAN);
    }
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
nanmean_all_DTYPE0(PyObject *ita, Py_ssize_t stride, Py_ssize_t length)
{
    Py_ssize_t i;
    Py_ssize_t size = 0;
    npy_DTYPE1 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    while (PyArray_ITER_NOTDONE(ita)) {
        for (i = 0; i < length; i++) {
            asum += *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ita)) + i*stride);
        }
        size += length;
        PyArray_ITER_NEXT(ita);
    }
    Py_DECREF(ita);
    BN_END_ALLOW_THREADS
    if (size > 0) { 
        return PyFloat_FromDouble(asum / size);
    } else {
        return PyFloat_FromDouble(NAN);
    }
}
/* dtype end */


/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmean_one_DTYPE0(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int ndim,
                   npy_intp* y_dims)
{
    Py_ssize_t i;
    Py_ssize_t count;
    npy_DTYPE0 asum, ai;
    PyObject *y = PyArray_EMPTY(ndim - 1, y_dims, NPY_DTYPE0, 0);
    PyObject *ity = PyArray_IterNew(y);
    BN_BEGIN_ALLOW_THREADS
    if (length == 0) {
        while (PyArray_ITER_NOTDONE(ity)) {
            *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ity))) = NAN;
            PyArray_ITER_NEXT(ity);
        }
    }
    else {
        while (PyArray_ITER_NOTDONE(ita)) {
            asum = 0;
            count = 0;
            for (i = 0; i < length; i++) {
                ai = *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ita)) +
                                                                  i*stride);
                if (ai == ai) {
                    asum += ai;
                    count += 1;
                }
            }
            if (count > 0) { 
                asum /= count;
            } else {
                asum = NAN;
            }
            *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ity))) = asum;
            PyArray_ITER_NEXT(ita);
            PyArray_ITER_NEXT(ity);
        }
    }
    Py_DECREF(ita);
    Py_DECREF(ity);
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
nanmean_one_DTYPE0(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int ndim,
                   npy_intp* y_dims)
{
    Py_ssize_t i;
    npy_DTYPE1 asum = 0;
    PyObject *y = PyArray_EMPTY(ndim - 1, y_dims, NPY_DTYPE1, 0);
    PyObject *ity = PyArray_IterNew(y);
    if (length == 0) {
        while (PyArray_ITER_NOTDONE(ity)) {
            *(npy_DTYPE1*)(((char*)PyArray_ITER_DATA(ity))) = NAN;
            PyArray_ITER_NEXT(ity);
        }
    }
    else {
        while (PyArray_ITER_NOTDONE(ita)) {
            asum = 0;
            for (i = 0; i < length; i++) {
                asum += *(npy_DTYPE0*)(((char*)PyArray_ITER_DATA(ita)) +
                                                                  i*stride);
            }
            if (length > 0) { 
                asum /= length;
            } else {
                asum = NAN;
            }
            *(npy_DTYPE1*)(((char*)PyArray_ITER_DATA(ity))) = asum;
            PyArray_ITER_NEXT(ita);
            PyArray_ITER_NEXT(ity);
        }
    }
    Py_DECREF(ita);
    Py_DECREF(ity);
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
                   nanmean_all_ss_float64,
                   nanmean_all_ss_float32,
                   nanmean_all_ss_int64,
                   nanmean_all_ss_int32,
                   nanmean_one_float64,
                   nanmean_one_float32,
                   nanmean_one_int64,
                   nanmean_one_int32,
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

static inline int
parse_args(PyObject *args,
           PyObject *kwds,
           PyObject **arr_obj,
           PyObject **axis_obj)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (kwds) {
        const Py_ssize_t nkwds = PyDict_Size(kwds);
        if (nkwds == 1) {
            if (nargs == 0) {
                *arr_obj = PyDict_GetItem(kwds, pystr_arr);
                if (!*arr_obj) {
                    PyErr_SetString(PyExc_TypeError, "can't find `arr` input");
                    return 0;
                }
            }
            else {
                *axis_obj = PyDict_GetItem(kwds, pystr_axis);
                if (!*axis_obj) {
                    PyErr_SetString(PyExc_TypeError, "can't find `axis` input");
                    return 0;
                }
                if (nargs == 1) {
                    *arr_obj = PyTuple_GET_ITEM(args, 0);
                }
                else {
                    PyErr_SetString(PyExc_TypeError, "wrong number of inputs");
                    return 0;
                }
            }
        }
        else if (nkwds == 2) {
            if (nargs != 0) {
                PyErr_SetString(PyExc_TypeError, "wrong number of inputs");
                return 0;
            }
            *arr_obj = PyDict_GetItem(kwds, pystr_arr);
            if (!*arr_obj) {
                PyErr_SetString(PyExc_TypeError, "can't find `arr` input");
                return 0;
            }
            *axis_obj = PyDict_GetItem(kwds, pystr_axis);
            if (!*axis_obj) {
                PyErr_SetString(PyExc_TypeError, "can't find `axis` input");
                return 0;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError, "wrong number of inputs");
            return 0;
        }
    }
    else if (nargs == 1) {
        *arr_obj = PyTuple_GET_ITEM(args, 0);
    }
    else if (nargs == 2) {
        *arr_obj = PyTuple_GET_ITEM(args, 0);
        *axis_obj = PyTuple_GET_ITEM(args, 1);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "wrong number of inputs");
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
        fall_ss_t fall_ss_float64,
        fall_ss_t fall_ss_float32,
        fall_ss_t fall_ss_int64,
        fall_ss_t fall_ss_int32,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32,
        int ravel,
        int copy)
{
    PyObject *arr_obj;
    PyObject *axis_obj = Py_None;
    if (!parse_args(args, kwds, &arr_obj, &axis_obj)) {
        return NULL;
    }

    /* convert to array if necessary */
    PyArrayObject *a;
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

    int ndim = PyArray_NDIM(a);

    /* defend against 0d beings */
    if (ndim == 0) {
        if (axis_obj == Py_None ||
            axis_obj == PyInt_FromLong(0) ||
            axis_obj == PyInt_FromLong(-1))
            return slow(name, args, kwds);
        else {
            PyErr_Format(PyExc_ValueError, "axis out of bounds for 0d input");
            return NULL;
        }
    }

    /* does user want to reduce over all axes? */
    int reduce_all = 0;
    int axis;
    if (axis_obj == Py_None) {
        reduce_all = 1;
        axis = -1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            PyErr_SetString(PyExc_TypeError, "`axis` must be an integer");
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

    Py_ssize_t i;
    Py_ssize_t stride;
    Py_ssize_t length;

    PyObject *ita;
    char *p;

    int dtype = PyArray_TYPE(a);
    npy_intp *shape = PyArray_SHAPE(a);
    npy_intp *strides = PyArray_STRIDES(a);

    if (reduce_all == 1) {
        /* reduce over all axes */

        if (ndim==1 || PyArray_CHKFLAGS(a, NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_CHKFLAGS(a, NPY_ARRAY_F_CONTIGUOUS)) {
            /* low function call overhead reduction */
            length = shape[0];
            stride = strides[0];
            for (i=1; i < ndim; i++) {
                length *= shape[i];
                if (strides[i] < stride) {
                    stride = strides[i];
                }
            }
            p = (char *)PyArray_DATA(a);
            if (dtype == NPY_FLOAT64) {
                return fall_ss_float64(p, stride, length);
            }
            else if (dtype == NPY_FLOAT32) {
                return fall_ss_float32(p, stride, length);
            }
            else if (dtype == NPY_INT64) {
                return fall_ss_int64(p, stride, length);
            }
            else if (dtype == NPY_INT32) {
                return fall_ss_int32(p, stride, length);
            }
            else {
                return slow(name, args, kwds);
            }
        }
        else {
            if (ravel == 0) {
                ita = PyArray_IterAllButAxis((PyObject *)a, &axis);
                stride = strides[axis];
                length = shape[axis];
            }
            else {
                /* TODO a = PyArray_Ravel(a, NPY_ANYORDER);*/
                axis = 0;
                ita = PyArray_IterAllButAxis((PyObject *)a, &axis);
                stride = PyArray_STRIDE(a, 0);
                length = PyArray_SIZE(a);
            }
            if (dtype == NPY_FLOAT64) {
                return fall_float64(ita, stride, length);
            }
            else if (dtype == NPY_FLOAT32) {
                return fall_float32(ita, stride, length);
            }
            else if (dtype == NPY_INT64) {
                return fall_int64(ita, stride, length);
            }
            else if (dtype == NPY_INT32) {
                return fall_int32(ita, stride, length);
            }
            else {
                return slow(name, args, kwds);
            }
        }
    }

    /* if we have reached this point then we are reducing an array with
       ndim > 1 over a single axis */

    /* output array */
    npy_intp y_dims[NPY_MAXDIMS];

    /* input iterator */
    ita = PyArray_IterAllButAxis((PyObject *)a, &axis);
    stride = strides[axis];
    length = shape[axis];

    /* reduce over a single axis; ndim > 1 */
    Py_ssize_t j = 0;
    for (i=0; i < ndim; i++) {
        if (i != axis) {
            y_dims[j++] = shape[i];
        }
    }
    if (dtype == NPY_FLOAT64) {
        return fone_float64(ita, stride, length, ndim, y_dims);
    }
    else if (dtype == NPY_FLOAT32) {
        return fone_float32(ita, stride, length, ndim, y_dims);
    }
    else if (dtype == NPY_INT64) {
        return fone_int64(ita, stride, length, ndim, y_dims);
    }
    else if (dtype == NPY_INT32) {
        return fone_int32(ita, stride, length, ndim, y_dims);
    }
    else {
        return slow(name, args, kwds);
    }

}

/* docstrings ------------------------------------------------------------- */

static char module_docstring[] =
    "Bottleneck functions that reduce the input array along a specified axis.";

static char nansum_docstring[] =
    "Sum of array elements along given axis treating NaNs as zero.";
static char nanmean_docstring[] =
    "Mean of array elements along given axis ignoring NaNs.";

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
module_methods[] = {
    {"nansum", (PyCFunction)nansum, VAKW, nansum_docstring},
    {"nanmean", (PyCFunction)nanmean, VAKW, nanmean_docstring},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
   PyModuleDef_HEAD_INIT,
   "reduce2",
   module_docstring,
   -1,
   module_methods
};
#endif


#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyMODINIT_FUNC PyInit_reduce2(void)
#else
#define RETVAL
PyMODINIT_FUNC
initreduce2(void)
#endif
{
#if PY_MAJOR_VERSION >=3
    PyObject *m = PyModule_Create(&moduledef);
#else
    PyObject *m = Py_InitModule3("reduce2", module_methods, module_docstring);
#endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
