#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <numpy/arrayobject.h>
#include "slow.h"

/* THREADS=1 releases the GIL but increases function call
 * overhead. THREADS=0 does not release the GIL but keeps
 * function call overhead low. */
#define THREADS 1
#if THREADS
    #define BN_BEGIN_ALLOW_THREADS Py_BEGIN_ALLOW_THREADS
    #define BN_END_ALLOW_THREADS Py_END_ALLOW_THREADS
#else
    #define BN_BEGIN_ALLOW_THREADS
    #define BN_END_ALLOW_THREADS
#endif

#define VAKW METH_VARARGS | METH_KEYWORDS
#define error_converting(x) (((x) == -1) && PyErr_Occurred())

#if PY_MAJOR_VERSION >= 3
#define PyString_FromString PyBytes_FromString
#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong
#define PyString_InternFromString PyUnicode_InternFromString
#endif
