#include "bottleneck.h"
#include "iterators.h"

/* typedefs and prototypes ----------------------------------------------- */

/* function pointer for functions passed to nonreducer */
typedef PyObject *(*nr_t)(PyArrayObject *, double, double);

static PyObject *
nonreducer(char *name,
           PyObject *args,
           PyObject *kwds,
           nr_t,
           nr_t,
           nr_t,
           nr_t,
           int inplace);

/* replace --------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
replace_DTYPE0(PyArrayObject *a, double old, double new)
{
    npy_DTYPE0 ai;
    iter it;
    init_iter_all(&it, a, 0, 1);
    BN_BEGIN_ALLOW_THREADS
    if (old == old) {
        WHILE {
            FOR {
                if (AI(DTYPE0) == old) AI(DTYPE0) = new;
            }
            NEXT
        }
    }
    else {
        WHILE {
            FOR {
                ai = AI(DTYPE0);
                if (ai != ai) AI(DTYPE0) = new;
            }
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    Py_INCREF(a);
    return (PyObject *)a;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
replace_DTYPE0(PyArrayObject *a, double old, double new)
{
    npy_DTYPE0 oldint, newint;
    iter it;
    init_iter_all(&it, a, 0, 1);
    if (old == old) {
        oldint = (npy_DTYPE0)old;
        newint = (npy_DTYPE0)new;
        if (oldint != old) {
            VALUE_ERR("Cannot safely cast `old` to int");
            return NULL;
        }
        if (newint != new) {
            VALUE_ERR("Cannot safely cast `new` to int");
            return NULL;
        }
        BN_BEGIN_ALLOW_THREADS
        WHILE {
            FOR {
                if (AI(DTYPE0) == oldint) AI(DTYPE0) = newint;
            }
            NEXT
        }
        BN_END_ALLOW_THREADS
    }
    Py_INCREF(a);
    return (PyObject *)a;
}
/* dtype end */

static PyObject *
replace(PyObject *self, PyObject *args, PyObject *kwds)
{
    return nonreducer("replace",
                      args,
                      kwds,
                      replace_float64,
                      replace_float32,
                      replace_int64,
                      replace_int32,
                      1);
}


/* python strings -------------------------------------------------------- */

PyObject *pystr_a = NULL;
PyObject *pystr_old = NULL;
PyObject *pystr_new = NULL;

static int
intern_strings(void) {
    pystr_a = PyString_InternFromString("a");
    pystr_old = PyString_InternFromString("old");
    pystr_new = PyString_InternFromString("new");
    return pystr_a && pystr_old && pystr_new;
}

/* nonreduce ------------------------------------------------------------- */

static BN_INLINE int
parse_args(PyObject *args,
           PyObject *kwds,
           PyObject **a,
           PyObject **old,
           PyObject **new)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int nkwds_found = 0;
        switch (nargs) {
            case 2: *old = PyTuple_GET_ITEM(args, 1);
            case 1: *a = PyTuple_GET_ITEM(args, 0);
            case 0: break;
            default:
                TYPE_ERR("wrong number of arguments 1");
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
                *old = PyDict_GetItem(kwds, pystr_old);
                if (*old == NULL) {
                    TYPE_ERR("Cannot find `old` keyword input");
                    return 0;
                }
                nkwds_found += 1;
            case 2:
                *new = PyDict_GetItem(kwds, pystr_new);
                if (*new == NULL) {
                    TYPE_ERR("Cannot find `new` keyword input");
                    return 0;
                }
                nkwds_found += 1;
                break;
            default:
                TYPE_ERR("wrong number of arguments 2");
                return 0;
        }
        if (nkwds_found != nkwds) {
            TYPE_ERR("wrong number of keyword arguments 3");
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
                *a = PyTuple_GET_ITEM(args, 0);
                *old = PyTuple_GET_ITEM(args, 1);
                *new = PyTuple_GET_ITEM(args, 2);
                break;
            default:
                TYPE_ERR("wrong number of arguments 4");
                return 0;
        }
    }

    return 1;

}

static PyObject *
nonreducer(char *name,
           PyObject *args,
           PyObject *kwds,
           nr_t nr_float64,
           nr_t nr_float32,
           nr_t nr_int64,
           nr_t nr_int32,
           int inplace)
{
    int dtype;
    double old, new;
    PyArrayObject *a;

    PyObject *a_obj = NULL;
    PyObject *old_obj = NULL;
    PyObject *new_obj = NULL;

    if (!parse_args(args, kwds, &a_obj, &old_obj, &new_obj)) return NULL;

    /* convert to array if necessary */
    if PyArray_Check(a_obj) {
        a = (PyArrayObject *)a_obj;
    } else {
        if (inplace) {
            TYPE_ERR("works in place so input must be an array, "
                     "not (e.g.) a list");
            return NULL;
        }
        a = (PyArrayObject *)PyArray_FROM_O(a_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        return slow(name, args, kwds);
    }

    /* old */
    if (old_obj == NULL) {
        RUNTIME_ERR("`old_obj` should never be NULL; please report this bug.");
        return NULL;
    }
    else {
        old = PyFloat_AsDouble(old_obj);
        if (error_converting(old)) {
            TYPE_ERR("`old` must be a number");
            return NULL;
        }
    }

    /* new */
    if (new_obj == NULL) {
        RUNTIME_ERR("`new_obj` should never be NULL; please report this bug.");
        return NULL;
    }
    else {
        new = PyFloat_AsDouble(new_obj);
        if (error_converting(new)) {
            TYPE_ERR("`new` must be a number");
            return NULL;
        }
    }

    dtype = PyArray_TYPE(a);

    if      (dtype == NPY_float64) return nr_float64(a, old, new);
    else if (dtype == NPY_float32) return nr_float32(a, old, new);
    else if (dtype == NPY_int64)   return nr_int64(a, old, new);
    else if (dtype == NPY_int32)   return nr_int32(a, old, new);
    else                           return slow(name, args, kwds);

}

/* docstrings ------------------------------------------------------------- */

static char nonreduce_doc[] =
"Bottleneck nonreducing functions.";

static char replace_doc[] =
/* MULTILINE STRING BEGIN
replace(a, old, new)

Replace (inplace) given scalar values of an array with new values.

The equivalent numpy function:

    a[a==old] = new

Or in the case where old=np.nan:

    a[np.isnan(old)] = new

Parameters
----------
a : numpy.ndarray
    The input array, which is also the output array since this functions
    works inplace.
old : scalar
    All elements in `a` with this value will be replaced by `new`.
new : scalar
    All elements in `a` with a value of `old` will be replaced by `new`.

Returns
-------
Returns a view of the input array after performing the replacements,
if any.

Examples
--------
Replace zero with 3 (note that the input array is modified):

>>> a = np.array([1, 2, 0])
>>> bn.replace(a, 0, 3)
>>> a
array([1, 2, 3])

Replace np.nan with 0:

>>> a = np.array([1, 2, np.nan])
>>> bn.replace(a, np.nan, 0)
>>> a
array([ 1.,  2.,  0.])

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
nonreduce_methods[] = {
    {"replace", (PyCFunction)replace, VARKEY, replace_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
nonreduce_def = {
   PyModuleDef_HEAD_INIT,
   "nonreduce",
   nonreduce_doc,
   -1,
   nonreduce_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_nonreduce(void)
#else
#define RETVAL
initnonreduce(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&nonreduce_def);
    #else
        PyObject *m = Py_InitModule3("nonreduce", nonreduce_methods,
                                     nonreduce_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
