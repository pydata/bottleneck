#include <Python.h>
#include <numpy/arrayobject.h>

/* docstrings ------------------------------------------------------------- */

static char module_docstring[] =
    "Bottleneck functions that reduce the input array along a specified axis.";

static char nansum_docstring[] =
    "Sum of array elements along given axis treating NaNs as zero.";

/* function pointers ----------------------------------------------------- */

/* pointer to functions that reduce along ALL axes */
typedef PyObject *(*fall_ss_t)(char *, npy_intp, npy_intp, int);
typedef PyObject *(*fall_t)(PyObject *, Py_ssize_t, Py_ssize_t, int);

/* pointer to functions that reduce along ONE axis */
typedef PyObject *(*fone_t)(PyObject *, Py_ssize_t, Py_ssize_t, int, npy_intp*, int);


/* prototypes ------------------------------------------------------------ */

static PyObject *
nansum(PyObject *self, PyObject *args, PyObject *kwds);

static PyObject *
nansum_all_ss_float64(char *p,
                      npy_intp stride,
                      npy_intp length,
                      int int_input);

static PyObject *
nansum_all_float64(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int int_input);

static PyObject *
nansum_one_float64(PyObject *ita,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int a_ndim,
                   npy_intp* y_dims,
                   int int_input);

static PyObject *
reducer(PyObject *arr,
        PyObject *axis,
        fall_t fall_float64,
        fall_ss_t fall_ss_float64,
        fone_t fone_float64,
        int int_input,
        int ravel,
        int copy);


/* python ---------------------------------------------------------------- */

/* module specification */
static PyMethodDef module_methods[] = {
    {"nansum", (PyCFunction)nansum, METH_VARARGS | METH_KEYWORDS, nansum_docstring},
    {NULL, NULL, 0, NULL}
};

/* initialize the module */
PyMODINIT_FUNC initnansum(void)
{
    PyObject *m = Py_InitModule3("nansum", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* load `numpy` functionality. */
    import_array();
}

/* nansum ---------------------------------------------------------------- */


static PyObject *
nansum(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *arr_obj;
    PyObject *axis_obj = Py_None;
    static char *kwlist[] = {"arr", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                     &arr_obj, &axis_obj)) {
        goto fail;
    }

    return reducer(arr_obj,
                   axis_obj,
                   nansum_all_float64,
                   nansum_all_ss_float64,
                   nansum_one_float64,
                   0, 0, 0);

    fail:
        Py_XDECREF(arr_obj);
        Py_XDECREF(axis_obj);
        return NULL;

}


static PyObject *
nansum_all_ss_float64(char *p,
                      npy_intp stride,
                      npy_intp length,
                      int int_input)
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
                   Py_ssize_t length,
                   int int_input)
{
    Py_ssize_t i;
    npy_float64 asum = 0, ai;
    while (PyArray_ITER_NOTDONE(ita)) {
        for (i = 0; i < length; i++) {
            ai = (*(npy_float64*)(((char*)PyArray_ITER_DATA(ita)) + i * stride));
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
                   npy_intp* y_dims,
                   int int_input)
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
                ai = (*(npy_float64*)(((char*)PyArray_ITER_DATA(ita)) + i*stride));
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
reducer(PyObject *arr,
        PyObject *axis,
        fall_t fall_float64,
        fall_ss_t fall_ss_float64,
        fone_t fone_float64,
        int int_input,
        int ravel,
        int copy)
{

    /* convert to array if necessary */
    PyObject *a;
    if PyArray_Check(arr) {
        a = arr;
    } else {
        a = PyArray_FROM_O(arr);
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

    PyObject *ita;
    Py_ssize_t stride, length, i; /*  , j; */
    int dtype = PyArray_TYPE(a);
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
    int axis_int;
    int axis_reduce;
    if (axis == Py_None) {
        reduce_all = 1;
        axis_reduce = -1;
    }
    else {
        axis_int = PyArray_PyIntAsInt(axis); /* TODO check for -1 returned */
        if (axis_int < 0) {
            axis_int += a_ndim;
            if (axis_int < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis_int);
                return NULL;
            }
        }
        else if (axis_int >= a_ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis_int);
            return NULL;
        }
        if (a_ndim == 1 && axis_int == 0) {
            reduce_all = 1;
        }
        axis_reduce = axis_int;
    }

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
                return fall_ss_float64(p, stride, length, int_input);
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
                return fall_float64(ita, stride, length, int_input);
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
        y = fone_float64(ita, stride, length, a_ndim, y_dims, int_input);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "dtype not yet supported");
        return NULL;
    }
    return y;
    */

}
