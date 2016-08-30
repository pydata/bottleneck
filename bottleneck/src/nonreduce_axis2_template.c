#include "bottleneck.h"

/* iterator -------------------------------------------------------------- */

 /*
 * INIT and NEXT are loosely based on NumPy's PyArray_IterAllButAxis and
 * PyArray_ITER_NEXT.
 */

#define INIT(dt) \
    PyObject *_y = PyArray_EMPTY(ndim, shape, dt, 0); \
    BN_BEGIN_ALLOW_THREADS \
    Py_ssize_t _i = 0; \
    char *_py = PyArray_BYTES((PyArrayObject *)_y); \
    char *_pa = PyArray_BYTES(a); \
    const npy_intp *_ystrides = PyArray_STRIDES((PyArrayObject *)_y); \
    const npy_intp *_astrides = PyArray_STRIDES(a); \
    const npy_intp _ystride = _ystrides[axis]; \
    npy_intp _index = 0; \
    npy_intp _size = PyArray_SIZE(a); \
    npy_intp _indices[ndim]; \
    memset(_indices, 0, ndim * sizeof(npy_intp)); \
    if (length != 0) _size /= length;

#define NEXT \
    for (_i = ndim - 1; _i >= 0; _i--) { \
        if (_i == axis) continue; \
        if (_indices[_i] < shape[_i] - 1) { \
            _pa += _astrides[_i]; \
            _py += _ystrides[_i]; \
            _indices[_i]++; \
            break; \
        } \
        _pa -= _indices[_i] * _astrides[_i]; \
        _py -= _indices[_i] * _ystrides[_i]; \
        _indices[_i] = 0; \
    } \
    _index++;

#define RETURN \
    BN_END_ALLOW_THREADS \
    return _y;

#define  WHILE   while (_index < _size)

#define  A0(dt)    *(dt *)(_pa)
#define  AI(dt)    *(dt *)(_pa + _i * stride)
#define  AOLD(dt)  *(dt *)(_pa + (_i - window) * stride)
#define  AX(dt, x) *(dt *)(_pa + x * stride)
#define  YI(dt)    *(dt *)(_py + _i++ * _ystride)

/* function signatures --------------------------------------------------- */

/* low-level functions such as move_sum_float64 */
#define NRA(name, dtype) \
    static PyObject * \
    name##_##dtype(PyArrayObject *a, int axis, int ndim, int n)

/* top-level functions such as move_sum */
#define NRA_MAIN(name, has_n) \
    static PyObject * \
    name(PyObject *self, PyObject *args, PyObject *kwds) \
    { \
        return nonreducer_axis(#name, \
                               args, \
                               kwds, \
                               name##_float64, \
                               name##_float32, \
                               name##_int64, \
                               name##_int32, \
                               has_n); \
    }

/* typedefs and prototypes ----------------------------------------------- */

/* function pointer for functions passed to nonreducer_axis */
typedef PyObject *(*nra_t)(PyArrayObject *, int, int, int);

static PyObject *
nonreducer_axis(char *name,
                PyObject *args,
                PyObject *kwds,
                nra_t,
                nra_t,
                nra_t,
                nra_t,
                int);

/* partsort -------------------------------------------------------------- */

#define INIT_FOR \
    Py_ssize_t _i; \
    char *_pa = PyArray_BYTES(a); \

#define INIT_CORE \
    INIT_FOR \
    const npy_intp *_astrides = PyArray_STRIDES(a); \
    const npy_intp *_ashape = PyArray_SHAPE(a); \
    npy_intp _index = 0; \
    npy_intp _size = PyArray_SIZE(a); \
    npy_intp _indices[ndim]; \

#define INIT2 \
    INIT_CORE \
    memset(_indices, 0, ndim * sizeof(npy_intp)); \
    if (length != 0) _size /= length; \

#define NEXT2 \
    for (_i = ndim - 1; _i >= 0; _i--) { \
        if (_i == axis) continue; \
        if (_indices[_i] < _ashape[_i] - 1) { \
            _pa += _astrides[_i]; \
            _indices[_i]++; \
            break; \
        } \
        _pa -= _indices[_i] * _astrides[_i]; \
        _indices[_i] = 0; \
    } \
    _index++;

#define B(dtype, i) AX(dtype, i) /* used by PARTITION */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
NRA(partsort, DTYPE0)
{
    a = (PyArrayObject *)PyArray_NewCopy(a, NPY_ANYORDER);
    Py_ssize_t length = PyArray_DIM(a, axis);
    Py_ssize_t stride = PyArray_STRIDE(a, axis);
    npy_intp i;
    INIT2
    if (length == 0) return (PyObject *)a;
    if (n < 1 || n > length) {
        PyErr_Format(PyExc_ValueError,
                     "`n` (=%d) must be between 1 and %zd, inclusive.",
                     n, length);
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    npy_intp j, l, r, k;
    k = n - 1;
    WHILE {
        l = 0;
        r = length - 1;
        PARTITION(npy_DTYPE0)
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return (PyObject *)a;
}
/* dtype end */

NRA_MAIN(partsort, 1)


/* python strings -------------------------------------------------------- */

PyObject *pystr_arr = NULL;
PyObject *pystr_n = NULL;
PyObject *pystr_axis = NULL;

static int
intern_strings(void) {
    pystr_arr = PyString_InternFromString("arr");
    pystr_n = PyString_InternFromString("n");
    pystr_axis = PyString_InternFromString("axis");
    return pystr_arr && pystr_n && pystr_axis;
}

/* nonreducer_axis ------------------------------------------------------- */

static BN_INLINE int
parse_args_n(PyObject *args,
             PyObject *kwds,
             PyObject **arr,
             PyObject **n,
             PyObject **axis)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (kwds) {
        int nkwds_found = 0;
        const Py_ssize_t nkwds = PyDict_Size(kwds);
        PyObject *tmp;
        switch (nargs) {
            case 2: *n = PyTuple_GET_ITEM(args, 1);
            case 1: *arr = PyTuple_GET_ITEM(args, 0);
            case 0: break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
        switch (nargs) {
            case 0:
                *arr = PyDict_GetItem(kwds, pystr_arr);
                if (*arr == NULL) {
                    TYPE_ERR("Cannot find `arr` keyword input");
                    return 0;
                }
                nkwds_found += 1;
            case 1:
                *n = PyDict_GetItem(kwds, pystr_n);
                if (*n == NULL) {
                    TYPE_ERR("Cannot find `n` keyword input");
                    return 0;
                }
                nkwds_found++;
            case 2:
                tmp = PyDict_GetItem(kwds, pystr_axis);
                if (tmp != NULL) {
                    *axis = tmp;
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
        if (nargs + nkwds_found > 3) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    }
    else {
        switch (nargs) {
            case 3:
                *axis = PyTuple_GET_ITEM(args, 2);
            case 2:
                *n = PyTuple_GET_ITEM(args, 1);
                *arr = PyTuple_GET_ITEM(args, 0);
                break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
    }

    return 1;

}

static PyObject *
nonreducer_axis(char *name,
                PyObject *args,
                PyObject *kwds,
                nra_t nra_float64,
                nra_t nra_float32,
                nra_t nra_int64,
                nra_t nra_int32,
                int has_n)
{

    int n;
    int axis;
    int dtype;
    int ndim;

    PyArrayObject *a;

    PyObject *arr_obj = NULL;
    PyObject *n_obj = NULL;
    PyObject *axis_obj = NULL;

    if (has_n) {
        if (!parse_args_n(args, kwds, &arr_obj, &n_obj, &axis_obj)) {
            return NULL;
        }
    }
    else {
        VALUE_ERR("not yet implemented");
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
    dtype = PyArray_TYPE(a);

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        return slow(name, args, kwds);
    }

    /* defend against the axis of negativity */
    ndim = PyArray_NDIM(a);
    if (axis_obj == NULL) {
        axis = ndim - 1;
        if (axis < 0) {
            PyErr_Format(PyExc_ValueError,
                         "axis(=%d) out of bounds", axis);
            return NULL;
        }
    }
    else if (axis_obj == Py_None) {
        a = (PyArrayObject *)PyArray_Ravel(a, NPY_ANYORDER);
        axis = 0;
        ndim = 1;
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
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis);
                return NULL;
            }
        }
        else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            return NULL;
        }
    }

    /* ddof */
    if (n_obj == NULL) {
        n = -1;
    }
    else {
        n = PyArray_PyIntAsInt(n_obj);
        if (error_converting(n)) {
            TYPE_ERR("`n` must be an integer");
            return NULL;
        }
    }

    if (dtype == NPY_float64)      return nra_float64(a, axis, ndim, n);
    else if (dtype == NPY_float32) return nra_float32(a, axis, ndim, n);
    else if (dtype == NPY_int64)   return nra_int64(a, axis, ndim, n);
    else if (dtype == NPY_int32)   return nra_int32(a, axis, ndim, n);
    else                           return slow(name, args, kwds);

}

/* docstrings ------------------------------------------------------------- */

static char nra_doc[] =
"Bottleneck non-reducing functions that operate along an axis.";

static char partsort_doc[] =
/* MULTILINE STRING BEGIN
partsort(arr, n, axis=-1)

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

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
nra_methods[] = {
    {"partsort",    (PyCFunction)partsort,    VARKEY, partsort_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
nra_def = {
   PyModuleDef_HEAD_INIT,
   "nonreduce_axis2",
   nra_doc,
   -1,
   nra_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_nonreduce_axis2(void)
#else
#define RETVAL
initnonreduce_axis2(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&nra_def);
    #else
        PyObject *m = Py_InitModule3("nonreduce_axis2", nra_methods, nra_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
