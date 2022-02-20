// Copyright 2010-2019 Keith Goodman
// Copyright 2019 Bottleneck Developers
#ifndef BOTTLENECK_H_
#define BOTTLENECK_H_

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <numpy/arrayobject.h>
#include <bn_config.h>

/* THREADS=1 releases the GIL but increases function call
 * overhead. THREADS=0 does not release the GIL but keeps
 * function call overhead low. Curly brackets are for C89
 * support. */
#define THREADS 1
#if THREADS
    #define BN_BEGIN_ALLOW_THREADS Py_BEGIN_ALLOW_THREADS {
    #define BN_END_ALLOW_THREADS   ;} Py_END_ALLOW_THREADS
#else
    #define BN_BEGIN_ALLOW_THREADS {
    #define BN_END_ALLOW_THREADS }
#endif

/* for ease of dtype templating */
#define NPY_float64 NPY_FLOAT64
#define NPY_float32 NPY_FLOAT32
#define NPY_int64   NPY_INT64
#define NPY_int32   NPY_INT32
#define NPY_intp    NPY_INTP
#define NPY_MAX_int64 NPY_MAX_INT64
#define NPY_MAX_int32 NPY_MAX_INT32
#define NPY_MIN_int64 NPY_MIN_INT64
#define NPY_MIN_int32 NPY_MIN_INT32

#if PY_MAJOR_VERSION >= 3
    #define PyString_FromString PyBytes_FromString
    #define PyInt_AsLong PyLong_AsLong
    #define PyString_InternFromString PyUnicode_InternFromString
#endif

#define VARKEY METH_VARARGS | METH_KEYWORDS
#define error_converting(x) (((x) == -1) && PyErr_Occurred())

#define VALUE_ERR(text)   PyErr_SetString(PyExc_ValueError, text)
#define TYPE_ERR(text)    PyErr_SetString(PyExc_TypeError, text)
#define MEMORY_ERR(text)  PyErr_SetString(PyExc_MemoryError, text)
#define RUNTIME_ERR(text) PyErr_SetString(PyExc_RuntimeError, text)

/* `inline` and `opt_3` copied from NumPy. */
#if HAVE_ATTRIBUTE_OPTIMIZE_OPT_3
    #define BN_OPT_3 __attribute__((optimize("O3")))
#else
    #define BN_OPT_3
#endif

/*
 * NAN and INFINITY like macros (same behavior as glibc for NAN, same as C99
 * for INFINITY). Copied from NumPy.
 */
static inline float __bn_inff(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x7f800000UL};
    return __bint.__f;
}

static inline float __bn_nanf(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x7fc00000UL};
    return __bint.__f;
}

#define BN_INFINITYF __bn_inff()
#define BN_NANF      __bn_nanf()
#define BN_INFINITY ((npy_double)BN_INFINITYF)
#define BN_NAN      ((npy_double)BN_NANF)

#define C_CONTIGUOUS(a) PyArray_CHKFLAGS(a, NPY_ARRAY_C_CONTIGUOUS)
#define F_CONTIGUOUS(a) PyArray_CHKFLAGS(a, NPY_ARRAY_F_CONTIGUOUS)
#define IS_CONTIGUOUS(a) (C_CONTIGUOUS(a) || F_CONTIGUOUS(a))

/* WIRTH ----------------------------------------------------------------- */

/*
 WIRTH macro based on:
   Fast median search: an ANSI C implementation
   Nicolas Devillard - ndevilla AT free DOT fr
   July 1998
 which, in turn, took the algorithm from
   Wirth, Niklaus
   Algorithms + data structures = programs, p. 366
   Englewood Cliffs: Prentice-Hall, 1976

 Adapted for Bottleneck:
 (C) 2016 Keith Goodman
*/

#define WIRTH(dtype) \
    x = B(dtype, k); \
    i = l; \
    j = r; \
    do { \
        while (B(dtype, i) < x) i++; \
        while (x < B(dtype, j)) j--; \
        if (i <= j) { \
            npy_##dtype atmp = B(dtype, i); \
            B(dtype, i) = B(dtype, j); \
            B(dtype, j) = atmp; \
            i++; \
            j--; \
        } \
    } while (i <= j); \
    if (j < k) l = i; \
    if (k < i) r = j;

/* partition ------------------------------------------------------------- */

#define PARTITION(dtype) \
    while (l < r) { \
        npy_##dtype x; \
        npy_##dtype al = B(dtype, l); \
        npy_##dtype ak = B(dtype, k); \
        npy_##dtype ar = B(dtype, r); \
        if (al > ak) { \
            if (ak < ar) { \
                if (al < ar) { \
                    B(dtype, k) = al; \
                    B(dtype, l) = ak; \
                } else { \
                    B(dtype, k) = ar; \
                    B(dtype, r) = ak; \
                } \
            } \
        } else { \
            if (ak > ar) { \
                if (al > ar) { \
                    B(dtype, k) = al; \
                    B(dtype, l) = ak; \
                } else { \
                    B(dtype, k) = ar; \
                    B(dtype, r) = ak; \
                } \
            } \
        } \
        WIRTH(dtype) \
    }

/* slow ------------------------------------------------------------------ */

static PyObject *slow_module = NULL;

static PyObject *
slow(char *name, PyObject *args, PyObject *kwds)
{
    PyObject *func = NULL;
    PyObject *out = NULL;

    if (slow_module == NULL) {
        /* bottleneck.slow has not been imported during the current
         * python session. Only import it once per session to save time */
        slow_module = PyImport_ImportModule("bottleneck.slow");
        if (slow_module == NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Cannot import bottleneck.slow");
            return NULL;
        }
    }

    func = PyObject_GetAttrString(slow_module, name);
    if (func == NULL) {
        PyErr_Format(PyExc_RuntimeError,
                     "Cannot import %s from bottleneck.slow", name);
        return NULL;
    }
    if (PyCallable_Check(func)) {
        out = PyObject_Call(func, args, kwds);
        if (out == NULL) {
            Py_XDECREF(func);
            return NULL;
        }
    } else {
        Py_XDECREF(func);
        PyErr_Format(PyExc_RuntimeError,
                     "bottleneck.slow.%s is not callable", name);
        return NULL;
    }
    Py_XDECREF(func);

    return out;
}

#endif  // BOTTLENECK_H_
