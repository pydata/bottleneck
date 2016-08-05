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

/* for ease of dtype templating */
#define NPY_float64 NPY_FLOAT64
#define NPY_float32 NPY_FLOAT32
#define NPY_int64 NPY_INT64
#define NPY_int32 NPY_INT32
#define NPY_intp NPY_INTP
#define NPY_MAX_int64 NPY_MAX_INT64
#define NPY_MAX_int32 NPY_MAX_INT32
#define NPY_MIN_int64 NPY_MIN_INT64
#define NPY_MIN_int32 NPY_MIN_INT32

#if PY_MAJOR_VERSION >= 3
    #define PyString_FromString PyBytes_FromString
    #define PyInt_FromLong PyLong_FromLong
    #define PyInt_AsLong PyLong_AsLong
    #define PyString_InternFromString PyUnicode_InternFromString
#endif

#define VARKEY METH_VARARGS | METH_KEYWORDS
#define error_converting(x) (((x) == -1) && PyErr_Occurred())

#define VALUE_ERR(i) PyErr_SetString(PyExc_ValueError, i)
#define TYPE_ERR(i) PyErr_SetString(PyExc_TypeError, i)

/* `inline` copied from NumPy. */
#if defined(_MSC_VER)
        #define BN_INLINE __inline
#elif defined(__GNUC__)
	#if defined(__STRICT_ANSI__)
		#define BN_INLINE __inline__
	#else
		#define BN_INLINE inline
	#endif
#else
        #define BN_INLINE
#endif

/*
 * NAN and INFINITY like macros (same behavior as glibc for NAN, same as C99
 * for INFINITY). Copied from NumPy.
 */
BN_INLINE static float __bn_inff(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x7f800000UL};
    return __bint.__f;
}

BN_INLINE static float __bn_nanf(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x7fc00000UL};
    return __bint.__f;
}

#define BN_INFINITYF __bn_inff()
#define BN_NANF __bn_nanf()
#define BN_INFINITY ((npy_double)BN_INFINITYF)
#define BN_NAN ((npy_double)BN_NANF)

/* does not check for 0d which by definition is contiguous */
#define IS_CONTIGUOUS(a) \
    (PyArray_CHKFLAGS(a, NPY_ARRAY_C_CONTIGUOUS) || \
     PyArray_CHKFLAGS(a, NPY_ARRAY_F_CONTIGUOUS))
