// Copyright 2010-2019 Keith Goodman
// Copyright 2019 Bottleneck Developers
#include "bottleneck.h"
#include "iterators.h"

/* function signatures --------------------------------------------------- */

/* low-level functions such as move_sum_float64 */
#define NRA(name, dtype) \
    static PyObject * \
    name##_##dtype(PyArrayObject *a, int axis, int n)

/* top-level functions such as move_sum */
#define NRA_MAIN(name, parse) \
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
                               parse); \
    }

/* typedefs and prototypes ----------------------------------------------- */

/* how should input be parsed? */
typedef enum {PARSE_PARTITION, PARSE_RANKDATA, PARSE_PUSH} parse_type;

/* function pointer for functions passed to nonreducer_axis */
typedef PyObject *(*nra_t)(PyArrayObject *, int, int);

static PyObject *
nonreducer_axis(char *name,
                PyObject *args,
                PyObject *kwds,
                nra_t,
                nra_t,
                nra_t,
                nra_t,
                parse_type);

/* partition ------------------------------------------------------------- */

#define B(dtype, i) AX(dtype, i) /* used by PARTITION */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
NRA(partition, DTYPE0) {
    npy_intp i;
    npy_intp j, l, r, k;
    iter it;

    a = (PyArrayObject *)PyArray_NewCopy(a, NPY_ANYORDER);
    init_iter_one(&it, a, axis);

    if (LENGTH == 0) return (PyObject *)a;
    if (n < 0 || n > LENGTH - 1) {
        PyErr_Format(PyExc_ValueError,
                     "`n` (=%d) must be between 0 and %zd, inclusive.",
                     n, LENGTH - 1);
        return NULL;
    }

    BN_BEGIN_ALLOW_THREADS
    k = n;
    WHILE {
        l = 0;
        r = LENGTH - 1;
        PARTITION(DTYPE0)
        NEXT
    }
    BN_END_ALLOW_THREADS

    return (PyObject *)a;
}
/* dtype end */

NRA_MAIN(partition, PARSE_PARTITION)


/* argpartition ----------------------------------------------------------- */

#define BUFFER_NEW(dtype) dtype *B = malloc(LENGTH * sizeof(dtype));
#define BUFFER_DELETE free(B);

#define ARGWIRTH(dtype0, dtype1) \
    x = B[k]; \
    i = l; \
    j = r; \
    do { \
        while (B[i] < x) i++; \
        while (x < B[j]) j--; \
        if (i <= j) { \
            npy_##dtype0 atmp = B[i]; \
            B[i] = B[j]; \
            B[j] = atmp; \
            ytmp = YX(dtype1, i); \
            YX(dtype1, i) = YX(dtype1, j); \
            YX(dtype1, j) = ytmp; \
            i++; \
            j--; \
        } \
    } while (i <= j); \
    if (j < k) l = i; \
    if (k < i) r = j;

#define ARGPARTITION(dtype0, dtype1) \
    while (l < r) { \
        npy_##dtype0 x; \
        npy_##dtype0 al = B[l]; \
        npy_##dtype0 ak = B[k]; \
        npy_##dtype0 ar = B[r]; \
        npy_##dtype1 ytmp; \
        if (al > ak) { \
            if (ak < ar) { \
                if (al < ar) { \
                    B[k] = al; \
                    B[l] = ak; \
                    ytmp = YX(dtype1, k); \
                    YX(dtype1, k) = YX(dtype1, l); \
                    YX(dtype1, l) = ytmp; \
                } else { \
                    B[k] = ar; \
                    B[r] = ak; \
                    ytmp = YX(dtype1, k); \
                    YX(dtype1, k) = YX(dtype1, r); \
                    YX(dtype1, r) = ytmp; \
                } \
            } \
        } else { \
            if (ak > ar) { \
                if (al > ar) { \
                    B[k] = al; \
                    B[l] = ak; \
                    ytmp = YX(dtype1, k); \
                    YX(dtype1, k) = YX(dtype1, l); \
                    YX(dtype1, l) = ytmp; \
                } else { \
                    B[k] = ar; \
                    B[r] = ak; \
                    ytmp = YX(dtype1, k); \
                    YX(dtype1, k) = YX(dtype1, r); \
                    YX(dtype1, r) = ytmp; \
                } \
            } \
        } \
        ARGWIRTH(dtype0, dtype1) \
    }

#define ARGPARTSORT(dtype0, dtype1) \
    for (i = 0; i < LENGTH; i++) { \
        B[i] = AX(dtype0, i); \
        YX(dtype1, i) = i; \
    } \
    l = 0; \
    r = LENGTH - 1; \
    ARGPARTITION(dtype0, dtype1)

/* dtype = [['float64', 'intp'], ['float32', 'intp'],
            ['int64',   'intp'], ['int32',   'intp']] */
NRA(argpartition, DTYPE0) {
    npy_intp i;
    PyObject *y = PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a),
                                NPY_DTYPE1, 0);
    iter2 it;
    init_iter2(&it, a, y, axis);
    if (LENGTH == 0) return y;
    if (n < 0 || n > LENGTH - 1) {
        PyErr_Format(PyExc_ValueError,
                     "`n` (=%d) must be between 0 and %zd, inclusive.",
                     n, LENGTH - 1);
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    BUFFER_NEW(npy_DTYPE0)
    npy_intp j, l, r, k;
    k = n;
    WHILE {
        l = 0;
        r = LENGTH - 1;
        ARGPARTSORT(DTYPE0, DTYPE1)
        NEXT2
    }
    BUFFER_DELETE
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

NRA_MAIN(argpartition, PARSE_PARTITION)


/* rankdata -------------------------------------------------------------- */

/* dtype = [['float64', 'float64', 'intp'], ['float32', 'float64', 'intp'],
            ['int64',   'float64', 'intp'], ['int32',   'float64', 'intp']] */
NRA(rankdata, DTYPE0) {
    Py_ssize_t j=0, k, idx, dupcount=0, i;
    npy_DTYPE1 old, new, averank, sumranks = 0;

    PyObject *z = PyArray_ArgSort(a, axis, NPY_QUICKSORT);
    PyObject *y = PyArray_EMPTY(PyArray_NDIM(a),
                                PyArray_SHAPE(a), NPY_DTYPE1, 0);

    iter3 it;
    init_iter3(&it, a, y, z, axis);

    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        Py_ssize_t size = PyArray_SIZE((PyArrayObject *)y);
        npy_DTYPE1 *py = (npy_DTYPE1 *)PyArray_DATA(a);
        for (i = 0; i < size; i++) YPP = BN_NAN;
    } else {
        WHILE {
            idx = ZX(DTYPE2, 0);
            old = AX(DTYPE0, idx);
            sumranks = 0;
            dupcount = 0;
            for (i = 0; i < LENGTH - 1; i++) {
                sumranks += i;
                dupcount++;
                k = i + 1;
                idx = ZX(DTYPE2, k);
                new = AX(DTYPE0, idx);
                if (old != new) {
                    averank = sumranks / dupcount + 1;
                    for (j = k - dupcount; j < k; j++) {
                        idx = ZX(DTYPE2, j);
                        YX(DTYPE1, idx) = averank;
                    }
                    sumranks = 0;
                    dupcount = 0;
                }
                old = new;
            }
            sumranks += (LENGTH - 1);
            dupcount++;
            averank = sumranks / dupcount + 1;
            for (j = LENGTH - dupcount; j < LENGTH; j++) {
                idx = ZX(DTYPE2, j);
                YX(DTYPE1, idx) = averank;
            }
            NEXT3
        }
    }
    BN_END_ALLOW_THREADS

    Py_DECREF(z);
    return y;
}
/* dtype end */

NRA_MAIN(rankdata, PARSE_RANKDATA)


/* nanrankdata ----------------------------------------------------------- */

/* dtype = [['float64', 'float64', 'intp'], ['float32', 'float64', 'intp']] */
NRA(nanrankdata, DTYPE0) {
    Py_ssize_t j=0, k, idx, dupcount=0, i;
    npy_DTYPE1 old, new, averank, sumranks = 0;

    PyObject *z = PyArray_ArgSort(a, axis, NPY_QUICKSORT);
    PyObject *y = PyArray_EMPTY(PyArray_NDIM(a),
                                PyArray_SHAPE(a), NPY_DTYPE1, 0);

    iter3 it;
    init_iter3(&it, a, y, z, axis);

    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        Py_ssize_t size = PyArray_SIZE((PyArrayObject *)y);
        npy_DTYPE1 *py = (npy_DTYPE1 *)PyArray_DATA(a);
        for (i = 0; i < size; i++) YPP = BN_NAN;
    } else {
        WHILE {
            idx = ZX(DTYPE2, 0);
            old = AX(DTYPE0, idx);
            sumranks = 0;
            dupcount = 0;
            for (i = 0; i < LENGTH - 1; i++) {
                sumranks += i;
                dupcount++;
                k = i + 1;
                idx = ZX(DTYPE2, k);
                new = AX(DTYPE0, idx);
                if (old != new) {
                    if (old == old) {
                        averank = sumranks / dupcount + 1;
                        for (j = k - dupcount; j < k; j++) {
                            idx = ZX(DTYPE2, j);
                            YX(DTYPE1, idx) = averank;
                        }
                    } else {
                        idx = ZX(DTYPE2, i);
                        YX(DTYPE1, idx) = BN_NAN;
                    }
                    sumranks = 0;
                    dupcount = 0;
                }
                old = new;
            }
            sumranks += (LENGTH - 1);
            dupcount++;
            averank = sumranks / dupcount + 1;
            if (old == old) {
                for (j = LENGTH - dupcount; j < LENGTH; j++) {
                    idx = ZX(DTYPE2, j);
                    YX(DTYPE1, idx) = averank;
                }
            } else {
                idx = ZX(DTYPE2, LENGTH - 1);
                YX(DTYPE1, idx) = BN_NAN;
            }
            NEXT3
        }
    }
    BN_END_ALLOW_THREADS

    Py_DECREF(z);
    return y;
}
/* dtype end */

static PyObject *
nanrankdata(PyObject *self, PyObject *args, PyObject *kwds) {
    return nonreducer_axis("nanrankdata",
                           args,
                           kwds,
                           nanrankdata_float64,
                           nanrankdata_float32,
                           rankdata_int64,
                           rankdata_int32,
                           PARSE_RANKDATA);
}


/* push ------------------------------------------------------------------ */

/* dtype = [['float64'], ['float32']] */
NRA(push, DTYPE0) {
    npy_intp index;
    npy_DTYPE0 ai, ai_last, n_float;
    PyObject *y = PyArray_Copy(a);
    iter it;
    init_iter_one(&it, (PyArrayObject *)y, axis);
    if (LENGTH == 0 || NDIM == 0) {
        return y;
    }
    n_float = n < 0 ? BN_INFINITY : (npy_DTYPE0)n;
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        index = 0;
        ai_last = BN_NAN;
        FOR {
            ai = AI(DTYPE0);
            if (ai == ai) {
                ai_last = ai;
                index = INDEX;
            } else {
                if (INDEX - index <= n_float) {
                    AI(DTYPE0) = ai_last;
                }
            }
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64'], ['int32']] */
NRA(push, DTYPE0) {
    PyObject *y = PyArray_Copy(a);
    return y;
}
/* dtype end */

NRA_MAIN(push, PARSE_PUSH)


/* python strings -------------------------------------------------------- */

PyObject *pystr_a = NULL;
PyObject *pystr_n = NULL;
PyObject *pystr_kth = NULL;
PyObject *pystr_axis = NULL;

static int
intern_strings(void) {
    pystr_a = PyString_InternFromString("a");
    pystr_n = PyString_InternFromString("n");
    pystr_kth = PyString_InternFromString("kth");
    pystr_axis = PyString_InternFromString("axis");
    return pystr_a && pystr_n && pystr_axis;
}

/* nonreducer_axis ------------------------------------------------------- */

static inline int
parse_partition(PyObject *args,
                PyObject *kwds,
                PyObject **a,
                PyObject **n,
                PyObject **axis) {
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int nkwds_found = 0;
        PyObject *tmp;
        switch (nargs) {
            case 2: *n = PyTuple_GET_ITEM(args, 1);
            case 1: *a = PyTuple_GET_ITEM(args, 0);
            case 0: break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
        switch (nargs) {
            case 0:
                *a = PyDict_GetItem(kwds, pystr_a);
                if (*a == NULL) {
                    TYPE_ERR("Cannot find `a` keyword input");
                    return 0;
                }
                nkwds_found += 1;
            case 1:
                *n = PyDict_GetItem(kwds, pystr_kth);
                if (*n == NULL) {
                    TYPE_ERR("Cannot find `kth` keyword input");
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
    } else {
        switch (nargs) {
            case 3:
                *axis = PyTuple_GET_ITEM(args, 2);
            case 2:
                *n = PyTuple_GET_ITEM(args, 1);
                *a = PyTuple_GET_ITEM(args, 0);
                break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
    }

    return 1;

}

static inline int
parse_rankdata(PyObject *args,
               PyObject *kwds,
               PyObject **a,
               PyObject **axis) {
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int nkwds_found = 0;
        PyObject *tmp;
        switch (nargs) {
            case 1: *a = PyTuple_GET_ITEM(args, 0);
            case 0: break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
        switch (nargs) {
            case 0:
                *a = PyDict_GetItem(kwds, pystr_a);
                if (*a == NULL) {
                    TYPE_ERR("Cannot find `a` keyword input");
                    return 0;
                }
                nkwds_found += 1;
            case 1:
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
        if (nargs + nkwds_found > 2) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    } else {
        switch (nargs) {
            case 2:
                *axis = PyTuple_GET_ITEM(args, 1);
            case 1:
                *a = PyTuple_GET_ITEM(args, 0);
                break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
    }

    return 1;

}

static inline int
parse_push(PyObject *args,
           PyObject *kwds,
           PyObject **a,
           PyObject **n,
           PyObject **axis) {
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int nkwds_found = 0;
        PyObject *tmp;
        switch (nargs) {
            case 2: *n = PyTuple_GET_ITEM(args, 1);
            case 1: *a = PyTuple_GET_ITEM(args, 0);
            case 0: break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
        switch (nargs) {
            case 0:
                *a = PyDict_GetItem(kwds, pystr_a);
                if (*a == NULL) {
                    TYPE_ERR("Cannot find `a` keyword input");
                    return 0;
                }
                nkwds_found += 1;
            case 1:
                tmp = PyDict_GetItem(kwds, pystr_n);
                if (tmp != NULL) {
                    *n = tmp;
                    nkwds_found++;
                }
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
    } else {
        switch (nargs) {
            case 3:
                *axis = PyTuple_GET_ITEM(args, 2);
            case 2:
                *n = PyTuple_GET_ITEM(args, 1);
            case 1:
                *a = PyTuple_GET_ITEM(args, 0);
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
                parse_type parse) {

    int n;
    int axis;
    int dtype;

    PyArrayObject *a;
    PyObject *y;

    PyObject *a_obj = NULL;
    PyObject *n_obj = NULL;
    PyObject *axis_obj = NULL;

    if (parse == PARSE_PARTITION) {
        if (!parse_partition(args, kwds, &a_obj, &n_obj, &axis_obj)) {
            return NULL;
        }
    } else if (parse == PARSE_RANKDATA) {
        if (!parse_rankdata(args, kwds, &a_obj, &axis_obj)) {
            return NULL;
        }
    } else if (parse == PARSE_PUSH) {
        if (!parse_push(args, kwds, &a_obj, &n_obj, &axis_obj)) {
            return NULL;
        }
    } else {
        RUNTIME_ERR("Unknown parse type; please report error.");
    }

    /* convert to array if necessary */
    if (PyArray_Check(a_obj)) {
        a = (PyArrayObject *)a_obj;
        Py_INCREF(a);
    } else {
        a = (PyArrayObject *)PyArray_FROM_O(a_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if (PyArray_ISBYTESWAPPED(a)) {
        Py_DECREF(a);
        return slow(name, args, kwds);
    }

    /* defend against the axis of negativity */
    if (axis_obj == NULL) {
        if (parse == PARSE_PARTITION || parse == PARSE_PUSH) {
            axis = PyArray_NDIM(a) - 1;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis);
                goto error;
            }
        } else {
            if (PyArray_NDIM(a) != 1) {
                a = (PyArrayObject *)PyArray_Ravel(a, NPY_CORDER);
            }
            axis = 0;
        }
    } else if (axis_obj == Py_None) {
        if (parse == PARSE_PUSH) {
            VALUE_ERR("`axis` cannot be None");
            goto error;
        }
        if (PyArray_NDIM(a) != 1) {
            a = (PyArrayObject *)PyArray_Ravel(a, NPY_CORDER);
        }
        axis = 0;
    } else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer");
            goto error;
        }
        if (axis < 0) {
            axis += PyArray_NDIM(a);
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis);
                goto error;
            }
        } else if (axis >= PyArray_NDIM(a)) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            goto error;
        }
    }

    /* n */
    if (n_obj == NULL) {
        n = -1;
    } else if (parse == PARSE_PUSH && n_obj == Py_None) {
        n = -1;
    } else {
        n = PyArray_PyIntAsInt(n_obj);
        if (error_converting(n)) {
            TYPE_ERR("`n` must be an integer");
            goto error;
        }
        if (n < 0 && parse == PARSE_PUSH) {
            VALUE_ERR("`n` must be nonnegative");
            goto error;
        }
    }

    dtype = PyArray_TYPE(a);
    if      (dtype == NPY_float64) y = nra_float64(a, axis, n);
    else if (dtype == NPY_float32) y = nra_float32(a, axis, n);
    else if (dtype == NPY_int64)   y = nra_int64(a, axis, n);
    else if (dtype == NPY_int32)   y = nra_int32(a, axis, n);
    else                           y = slow(name, args, kwds);

    Py_DECREF(a);

    return y;

error:
    Py_DECREF(a);
    return NULL;

}

/* docstrings ------------------------------------------------------------- */

static char nra_doc[] =
"Bottleneck non-reducing functions that operate along an axis.";

static char partition_doc[] =
/* MULTILINE STRING BEGIN
partition(a, kth, axis=-1)

Partition array elements along given axis.

A 1d array B is partitioned at array index `kth` if three conditions
are met: (1) B[kth] is in its sorted position, (2) all elements to the
left of `kth` are less than or equal to B[kth], and (3) all elements
to the right of `kth` are greater than or equal to B[kth]. Note that
the array elements in conditions (2) and (3) are in general unordered.

Shuffling the input array may change the output. The only guarantee is
given by the three conditions above.

This functions is not protected against NaN. Therefore, you may get
unexpected results if the input contains NaN.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
kth : int
    The value of the element at index `kth` will be in its sorted
    position. Smaller (larger) or equal values will be to the left
    (right) of index `kth`.
axis : {int, None}, optional
    Axis along which the partition is performed. The default
    (axis=-1) is to partition along the last axis.

Returns
-------
y : ndarray
    A partitioned copy of the input array with the same shape and
    type of `a`.

See Also
--------
bottleneck.argpartition: Indices that would partition an array

Notes
-----
Unexpected results may occur if the input array contains NaN.

Examples
--------
Create a numpy array:

>>> a = np.array([1, 0, 3, 4, 2])

Partition array so that the first 3 elements (indices 0, 1, 2) are the
smallest 3 elements (note, as in this example, that the smallest 3
elements may not be sorted):

>>> bn.partition(a, kth=2)
array([1, 0, 2, 4, 3])

Now Partition array so that the last 2 elements are the largest 2
elements:

>>> bn.partition(a, kth=3)
array([1, 0, 2, 3, 4])

MULTILINE STRING END */

static char argpartition_doc[] =
/* MULTILINE STRING BEGIN
argpartition(a, kth, axis=-1)

Return indices that would partition array along the given axis.

A 1d array B is partitioned at array index `kth` if three conditions
are met: (1) B[kth] is in its sorted position, (2) all elements to the
left of `kth` are less than or equal to B[kth], and (3) all elements
to the right of `kth` are greater than or equal to B[kth]. Note that
the array elements in conditions (2) and (3) are in general unordered.

Shuffling the input array may change the output. The only guarantee is
given by the three conditions above.

This functions is not protected against NaN. Therefore, you may get
unexpected results if the input contains NaN.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
kth : int
    The value of the element at index `kth` will be in its sorted
    position. Smaller (larger) or equal values will be to the left
    (right) of index `kth`.
axis : {int, None}, optional
    Axis along which the partition is performed. The default (axis=-1)
    is to partition along the last axis.

Returns
-------
y : ndarray
    An array the same shape as the input array containing the indices
    that partition `a`. The dtype of the indices is numpy.intp.

See Also
--------
bottleneck.partition: Partition array elements along given axis.

Notes
-----
Unexpected results may occur if the input array contains NaN.

Examples
--------
Create a numpy array:

>>> a = np.array([10, 0, 30, 40, 20])

Find the indices that partition the array so that the first 3
elements are the smallest 3 elements:

>>> index = bn.argpartition(a, kth=2)
>>> index
array([0, 1, 4, 3, 2])

Let's use the indices to partition the array (note, as in this
example, that the smallest 3 elements may not be in order):

>>> a[index]
array([10, 0, 20, 40, 30])

MULTILINE STRING END */

static char rankdata_doc[] =
/* MULTILINE STRING BEGIN
rankdata(a, axis=None)

Ranks the data, dealing with ties appropriately.

Equal values are assigned a rank that is the average of the ranks that
would have been otherwise assigned to all of the values within that set.
Ranks begin at 1, not 0.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the elements of the array are ranked. The default
    (axis=None) is to rank the elements of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`. The dtype is 'float64'.

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

MULTILINE STRING END */

static char nanrankdata_doc[] =
/* MULTILINE STRING BEGIN
nanrankdata(a, axis=None)

Ranks the data, dealing with ties and NaNs appropriately.

Equal values are assigned a rank that is the average of the ranks that
would have been otherwise assigned to all of the values within that set.
Ranks begin at 1, not 0.

NaNs in the input array are returned as NaNs.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the elements of the array are ranked. The default
    (axis=None) is to rank the elements of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`. The dtype is 'float64'.

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

MULTILINE STRING END */

static char push_doc[] =
/* MULTILINE STRING BEGIN
push(a, n=None, axis=-1)

Fill missing values (NaNs) with most recent non-missing values.

Filling proceeds along the specified axis from small index values to large
index values.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
n : {int, None}, optional
    How far to push values. If the most recent non-NaN array element is
    more than `n` index positions away, than a NaN is returned. The default
    (n = None) is to push the entire length of the slice. If `n` is an integer
    it must be nonnegative.
axis : int, optional
    Axis along which the elements of the array are pushed. The default
    (axis=-1) is to push along the last axis of the input array.

Returns
-------
y : ndarray
    An array with the same shape and dtype as `a`.

See also
--------
bottleneck.replace: Replace specified value of an array with new value.

Examples
--------
>>> a = np.array([5, np.nan, np.nan, 6, np.nan])
>>> bn.push(a)
    array([ 5.,  5.,  5.,  6.,  6.])
>>> bn.push(a, n=1)
    array([  5.,   5.,  nan,   6.,   6.])
>>> bn.push(a, n=2)
    array([ 5.,  5.,  5.,  6.,  6.])

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
nra_methods[] = {
    {"partition",    (PyCFunction)partition,    VARKEY, partition_doc},
    {"argpartition", (PyCFunction)argpartition, VARKEY, argpartition_doc},
    {"rankdata",     (PyCFunction)rankdata,     VARKEY, rankdata_doc},
    {"nanrankdata",  (PyCFunction)nanrankdata,  VARKEY, nanrankdata_doc},
    {"push",         (PyCFunction)push,         VARKEY, push_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
nra_def = {
   PyModuleDef_HEAD_INIT,
   "nonreduce_axis",
   nra_doc,
   -1,
   nra_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_nonreduce_axis(void)
#else
#define RETVAL
initnonreduce_axis(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&nra_def);
    #else
        PyObject *m = Py_InitModule3("nonreduce_axis", nra_methods, nra_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
