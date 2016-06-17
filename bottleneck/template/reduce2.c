#include <Python.h>
#include <numpy/arrayobject.h>

/* docstrings ------------------------------------------------------------- */

static char module_docstring[] =
    "Bottleneck functions that reduce the input array along a specified axis.";

static char nansum_docstring[] =
    "Sum of array elements along given axis treating NaNs as zero.";


/* python strings -------------------------------------------------------- */

PyObject *pystr_arr = NULL;
PyObject *pystr_axis = NULL;


/* function pointers ----------------------------------------------------- */

typedef PyObject *(*fall_ss_t)(char *, npy_intp, npy_intp);
typedef PyObject *(*fall_t)(PyObject *, Py_ssize_t, Py_ssize_t);
typedef PyObject *(*fone_t)(PyObject *, Py_ssize_t, Py_ssize_t, int,
                            npy_intp*);


/* prototypes ------------------------------------------------------------ */

static PyObject *
nansum(PyObject *self, PyObject *args, PyObject *kwds);

static PyObject *
nansum_all_ss_float64(char *p, npy_intp stride, npy_intp length);

static PyObject *
nansum_all_float64(PyObject *ita, Py_ssize_t stride, Py_ssize_t length);

static PyObject *
nansum_one_float64(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int a_ndim,
                   npy_intp* y_dims);

static PyObject *
reducer(PyObject *args,
        PyObject *kwds,
        fall_t fall_float64,
        fall_ss_t fall_ss_float64,
        fone_t fone_float64,
        int ravel,
        int copy);


/* nansum ---------------------------------------------------------------- */

static PyMethodDef
module_methods[] = {
    {"nansum", (PyCFunction)nansum, METH_VARARGS | METH_KEYWORDS,
      nansum_docstring},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
initnansum(void)
{
    PyObject *m = Py_InitModule3("nansum", module_methods, module_docstring);
    if (m == NULL) return;
    import_array();
    pystr_arr = PyString_FromString("arr");
    pystr_axis = PyString_FromString("axis");
}


static PyObject *
nansum(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer(args,
                   kwds,
                   nansum_all_float64,
                   nansum_all_ss_float64,
                   nansum_one_float64,
                   0, 0);
}


static PyObject *
nansum_all_ss_float64(char *p,
                      npy_intp stride,
                      npy_intp length)
{
Py_ssize_t i;
    npy_float64 ai;
    npy_float64 asum = 0;
    for (i = 0; i < length; i++) {
        ai = (*(npy_float64*)(p + i * stride));
        if (ai == ai) {
            asum += ai;
        }
    }
    return PyFloat_FromDouble(asum);
}


static PyObject *
nansum_all_float64(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length)
{
    Py_ssize_t i;
    npy_float64 asum = 0, ai;
    while (PyArray_ITER_NOTDONE(ita)) {
        for (i = 0; i < length; i++) {
            ai = (*(npy_float64*)(((char*)PyArray_ITER_DATA(ita)) + i*stride));
            if (ai == ai) {
                asum += ai;
            }
        }
        PyArray_ITER_NEXT(ita);
    }
    return PyFloat_FromDouble(asum);
}


static PyObject *
nansum_one_float64(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int a_ndim,
                   npy_intp* y_dims)
{
    Py_ssize_t i;
    npy_float64 asum = 0, ai;
    PyObject *y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_FLOAT64, 0);
    PyObject *ity = PyArray_IterNew(y);
    if (length == 0) {
        while (PyArray_ITER_NOTDONE(ity)) {
            (*(npy_float64*)(((char*)PyArray_ITER_DATA(ity)))) = asum;
            PyArray_ITER_NEXT(ity);
        }
    }
    else {
        while (PyArray_ITER_NOTDONE(ita)) {
            asum = 0;
            for (i = 0; i < length; i++) {
                ai = (*(npy_float64*)(((char*)PyArray_ITER_DATA(ita)) +
                                                                  i*stride));
                if (ai == ai) {
                    asum += ai;
                }
            }
            (*(npy_float64*)(((char*)PyArray_ITER_DATA(ity)))) = asum;
            PyArray_ITER_NEXT(ita);
            PyArray_ITER_NEXT(ity);
        }
    }
    return y;
}


/* reducer --------------------------------------------------------------- */

static PyObject *
reducer(PyObject *args,
        PyObject *kwds,
        fall_t fall_float64,
        fall_ss_t fall_ss_float64,
        fone_t fone_float64,
        int ravel,
        int copy)
{
    /* parse inputs: args and kwds */
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    PyObject *arr_obj;
    PyObject *axis_obj = Py_None;
    if (kwds) {
        const Py_ssize_t nkwds = PyDict_Size(kwds);
        if (nkwds == 1) {
            if (nargs == 0) {
                arr_obj = PyDict_GetItem(kwds, pystr_arr);
                if (!arr_obj) {
                    PyErr_SetString(PyExc_TypeError, "can't find `arr` input");
                    return NULL;
                }
            }
            else {
                axis_obj = PyDict_GetItem(kwds, pystr_axis);
                if (!axis_obj) {
                    PyErr_SetString(PyExc_TypeError, "can't find `axis` input");
                    return NULL;
                }
                if (nargs == 1) {
                    arr_obj = PyTuple_GET_ITEM(args, 0);
                }
                else {
                    PyErr_SetString(PyExc_TypeError, "wrong number of inputs");
                    return NULL;
                }
            }
        }
        else if (nkwds == 2) {
            if (nargs != 0) {
                PyErr_SetString(PyExc_TypeError, "wrong number of inputs");
                return NULL;
            }
            arr_obj = PyDict_GetItem(kwds, pystr_arr);
            if (!arr_obj) {
                PyErr_SetString(PyExc_TypeError, "can't find `arr` input");
                return NULL;
            }
            axis_obj = PyDict_GetItem(kwds, pystr_axis);
            if (!axis_obj) {
                PyErr_SetString(PyExc_TypeError, "can't find `axis` input");
                return NULL;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError, "wrong number of inputs");
            return NULL;
        }
    }
    else if (nargs == 1) {
        arr_obj = PyTuple_GET_ITEM(args, 0);
    }
    else if (nargs == 2) {
        arr_obj = PyTuple_GET_ITEM(args, 0);
        axis_obj = PyTuple_GET_ITEM(args, 1);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "wrong number of inputs");
        return NULL;
    }

    /* convert to array if necessary */
    PyObject *a;
    if PyArray_Check(arr_obj) {
        a = arr_obj;
    } else {
        a = PyArray_FROM_O(arr_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        /* TODO: call bn.slow */
        PyErr_SetString(PyExc_TypeError,
                        "Cannot yet handle byte-swapped arrays");
        return NULL;
    }

    /* input array
     TODO
    if (copy == 1) {
        a = PyArray_Copy(a);
    }
    */

    int a_ndim = PyArray_NDIM(a);

    /* defend against 0d beings */
    if (a_ndim == 0) {
        /*
        if (axis == Py_None || (int)axis == 0 || (int)axis == -1) {
            PyErr_SetString(PyExc_ValueError, "TODO: 0d input");
            return NULL;
        } else {
        */
        PyErr_SetString(PyExc_ValueError, "TODO: 0d input");
        return NULL;
    }

    /* does user want to reduce over all axes? */
    int reduce_all = 0;
    int axis;
    int axis_reduce;
    if (axis_obj == Py_None) {
        reduce_all = 1;
        axis_reduce = -1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (axis == -1) {
            PyErr_SetString(PyExc_TypeError, "`axis` must be an integer");
            return NULL;
        }
        if (axis < 0) {
            axis += a_ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis);
                return NULL;
            }
        }
        else if (axis >= a_ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            return NULL;
        }
        if (a_ndim == 1 && axis == 0) {
            reduce_all = 1;
        }
        axis_reduce = axis;
    }

    PyObject *ita;
    Py_ssize_t stride, length, i; /*  , j; */
    int dtype = PyArray_TYPE(a);

    npy_intp *shape = PyArray_DIMS(a);
    npy_intp *strides = PyArray_STRIDES(a);
    char *p;

    if (reduce_all == 1) {
        /* reduce over all axes */

        if (a_ndim==1 || PyArray_CHKFLAGS(a, NPY_C_CONTIGUOUS) ||
            PyArray_CHKFLAGS(a, NPY_F_CONTIGUOUS)) {
            stride = strides[0];
            for (i=1; i < a_ndim; i++) {
                if (strides[i] < stride) {
                    stride = strides[i];
                }
            }
            length = PyArray_SIZE(a);
            p = (char *)PyArray_DATA(a);
            if (dtype == NPY_FLOAT64) {
                return fall_ss_float64(p, stride, length);
            }
            else {
                PyErr_SetString(PyExc_TypeError, "dtype not yet supported");
                return NULL;
            }
        }
        else {
            if (ravel == 0) {
                ita = PyArray_IterAllButAxis(a, &axis_reduce);
                stride = strides[axis_reduce];
                length = shape[axis_reduce];
            }
            else {
                /* TODO a = PyArray_Ravel(a, NPY_ANYORDER);*/
                axis_reduce = 0;
                ita = PyArray_IterAllButAxis(a, &axis_reduce);
                stride = PyArray_STRIDE(a, 0);
                length = PyArray_SIZE(a);
            }
            if (dtype == NPY_FLOAT64) {
                return fall_float64(ita, stride, length);
            }
            else {
                PyErr_SetString(PyExc_TypeError, "dtype not yet supported");
                return NULL;
            }
        }
    }

    /* if we have reached this point then we are reducing an array with
       ndim > 1 over a single axis */

    /* output array
    PyArrayObject *y;
    npy_intp *y_dims[NPY_MAXDIMS];

    input iterator
    ita = PyArray_IterAllButAxis(a, &axis_reduce);
    stride = strides[axis_reduce];
    length = shape[axis_reduce];
    */

    PyErr_SetString(PyExc_TypeError, "can only reduce along all axes");
    return NULL;

    /* TODO reduce over a single axis; a_ndim > 1
    j = 0;
    for (i=0, j=0; i < a_ndim; i++, j++) {
        if (i != axis_reduce) {
            y_dims[j] = shape[i];
        }
    }
    if (dtype == NPY_FLOAT64) {
        y = fone_float64(ita, stride, length, a_ndim, y_dims);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "dtype not yet supported");
        return NULL;
    }
    return y;
    */

}
