#include "bottleneck.h"


/* iterator --------------------------------------------------------------
 *
 * The iterator below (INIT and NEXT) is similar to the iterator in move.c.
 * See that file for a brief description.
 */

#define INIT \
    Py_ssize_t _i; \
    char *_pa = PyArray_BYTES(a); \
    const npy_intp *_astrides = PyArray_STRIDES(a); \
    const npy_intp *_ashape = PyArray_SHAPE(a); \
    npy_intp _index = 0; \
    npy_intp _size = PyArray_SIZE(a); \
    npy_intp _indices[ndim]; \
    memset(_indices, 0, ndim * sizeof(npy_intp)); \
    if (length != 0) _size /= length;

#define NEXT \
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

/* if you  exited the iterator before it was done you will need
 * to call the memset line above inorder to reset */
#define RESET \
    _index = 0; \

#define  WHILE    while (_index < _size)
#define  FOR      for (_i = 0; _i < length; _i++)
#define  AI(dt)   *(dt*)(_pa + _i * stride)

/* output array ---------------------------------------------------------- */

#define Y_INIT(dt0, dt1) \
    PyObject *y = PyArray_EMPTY(ndim - 1, yshape, dt0, 0); \
    dt1 *py = (dt1 *)PyArray_DATA((PyArrayObject *)y);

#define YI *py++

/* function pointers ----------------------------------------------------- */

typedef PyObject *(*fall_t)(PyArrayObject *, int, Py_ssize_t, Py_ssize_t, int,
                            int);
typedef PyObject *(*fone_t)(PyArrayObject *, int, Py_ssize_t, Py_ssize_t, int,
                            npy_intp*, int);

/* prototypes ------------------------------------------------------------ */

static PyObject *
reducer(char *name,
        PyObject *args,
        PyObject *kwds,
        fall_t fall_float64,
        fall_t fall_float32,
        fall_t fall_int64,
        fall_t fall_int32,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32,
        int ravel,
        int copy,
        int has_ddof);

/* nansum ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nansum_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ignore)
{
    INIT
    npy_DTYPE0 ai, asum = 0;
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai == ai) asum += ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}


static PyObject *
nansum_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ignore)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    BN_BEGIN_ALLOW_THREADS
    npy_DTYPE0 ai, asum;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = 0;
    }
    else {
        WHILE {
            asum = 0;
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai == ai) asum += ai;
            }
            YI = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nansum_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ignore)
{
    INIT
    npy_DTYPE0 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(npy_DTYPE0);
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}


static PyObject *
nansum_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ignore)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    BN_BEGIN_ALLOW_THREADS
    npy_DTYPE0 asum;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = 0;
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(npy_DTYPE0);
            YI = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
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
                   nansum_one_float64,
                   nansum_one_float32,
                   nansum_one_int64,
                   nansum_one_int32,
                   0, 0, 0);
}

/* nanmean ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmean_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                   Py_ssize_t length, int ndim, int ignore)
{
    INIT
    Py_ssize_t count = 0;
    npy_DTYPE0 ai, asum = 0;
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (count > 0) {
        return PyFloat_FromDouble(asum / count);
    } else {
        return PyFloat_FromDouble(BN_NAN);
    }
}


static PyObject *
nanmean_one_DTYPE0(PyArrayObject *a,
                   int axis,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int ndim,
                   npy_intp* yshape,
                   int ignore)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    BN_BEGIN_ALLOW_THREADS
    Py_ssize_t count;
    npy_DTYPE0 ai, asum;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = BN_NAN;
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai == ai) {
                    asum += ai;
                    count += 1;
                }
            }
            if (count > 0) {
                asum /= count;
            } else {
                asum = BN_NAN;
            }
            YI = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
nanmean_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                   Py_ssize_t length, int ndim, int ignore)
{
    INIT
    Py_ssize_t total_length = 0;
    npy_DTYPE1 asum = 0;
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(npy_DTYPE0);
        total_length += length;
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (total_length > 0) {
        return PyFloat_FromDouble(asum / total_length);
    } else {
        return PyFloat_FromDouble(BN_NAN);
    }
}


static PyObject *
nanmean_one_DTYPE0(PyArrayObject *a,
                   int axis,
                   Py_ssize_t stride,
                   Py_ssize_t length,
                   int ndim,
                   npy_intp* yshape,
                   int ignore)
{
    Y_INIT(NPY_DTYPE1, npy_DTYPE1)
    INIT
    BN_BEGIN_ALLOW_THREADS
    npy_DTYPE1 asum;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = BN_NAN;
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(npy_DTYPE0);
            if (length > 0) {
                asum /= length;
            } else {
                asum = BN_NAN;
            }
            YI = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
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
                   nanmean_one_float64,
                   nanmean_one_float32,
                   nanmean_one_int64,
                   nanmean_one_int32,
                   0, 0, 0);
}

/* nanstd ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanstd_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ddof)
{
    INIT
    Py_ssize_t count = 0;
    npy_DTYPE0 ai, amean, out, asum = 0;
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count++;
            }
        }
        NEXT
    }
    if (count > ddof) {
        amean = asum / count;
        asum = 0;
        RESET
        WHILE {
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai == ai) {
                    ai -= amean;
                    asum += ai * ai;
                }
            }
            NEXT
        }
        out = sqrt(asum / (count - ddof));
    }
    else {
        out = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}


static PyObject *
nanstd_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ddof)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    BN_BEGIN_ALLOW_THREADS
    Py_ssize_t count;
    npy_DTYPE0 ai, asum, amean;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = BN_NAN;
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai == ai) {
                    asum += ai;
                    count++;
                }
            }
            if (count > ddof) {
                amean = asum / count;
                asum = 0;
                FOR {
                    ai = AI(npy_DTYPE0);
                    if (ai == ai) {
                        ai -= amean;
                        asum += ai * ai;
                    }
                }
                asum = sqrt(asum / (count - ddof));
            }
            else {
                asum = BN_NAN;
            }
            YI = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
nanstd_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ddof)
{
    INIT
    npy_DTYPE1 out;
    BN_BEGIN_ALLOW_THREADS
    Py_ssize_t size = 0;
    npy_DTYPE1 ai, amean, asum = 0;
    WHILE {
        FOR asum += AI(npy_DTYPE0);
        size += length;
        NEXT
    }
    if (size > ddof) {
        amean = asum / size;
        asum = 0;
        RESET
        WHILE {
            FOR {
                ai = AI(npy_DTYPE0) - amean;
                asum += ai * ai;
            }
            NEXT
        }
        out = sqrt(asum / (size - ddof));
    }
    else {
        out = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}


static PyObject *
nanstd_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ddof)
{
    Y_INIT(NPY_DTYPE1, npy_DTYPE1)
    INIT
    BN_BEGIN_ALLOW_THREADS
    npy_DTYPE1 ai, asum, amean;
    npy_DTYPE1 length_inv = 1.0 / length;
    npy_DTYPE1 length_ddof_inv = 1.0 / (length - ddof);
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = BN_NAN;
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(npy_DTYPE0);
            if (length > ddof) {
                amean = asum * length_inv;
                asum = 0;
                FOR {
                    ai = AI(npy_DTYPE0) - amean;
                    asum += ai * ai;
                }
                asum = sqrt(asum * length_ddof_inv);
            }
            else {
                asum = BN_NAN;
            }
            YI = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
nanstd(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nanstd",
                   args,
                   kwds,
                   nanstd_all_float64,
                   nanstd_all_float32,
                   nanstd_all_int64,
                   nanstd_all_int32,
                   nanstd_one_float64,
                   nanstd_one_float32,
                   nanstd_one_int64,
                   nanstd_one_int32,
                   0, 0, 1);
}

/* nanvar ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanvar_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ddof)
{
    INIT
    Py_ssize_t count = 0;
    npy_DTYPE0 ai, amean, out, asum = 0;
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count++;
            }
        }
        NEXT
    }
    if (count > ddof) {
        amean = asum / count;
        asum = 0;
        RESET
        WHILE {
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai == ai) {
                    ai -= amean;
                    asum += ai * ai;
                }
            }
            NEXT
        }
        out = asum / (count - ddof);
    }
    else {
        out = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}


static PyObject *
nanvar_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ddof)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    BN_BEGIN_ALLOW_THREADS
    Py_ssize_t count;
    npy_DTYPE0 ai, asum, amean;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = BN_NAN;
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai == ai) {
                    asum += ai;
                    count++;
                }
            }
            if (count > ddof) {
                amean = asum / count;
                asum = 0;
                FOR {
                    ai = AI(npy_DTYPE0);
                    if (ai == ai) {
                        ai -= amean;
                        asum += ai * ai;
                    }
                }
                asum = asum / (count - ddof);
            }
            else {
                asum = BN_NAN;
            }
            YI = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
nanvar_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ddof)
{
    INIT
    npy_DTYPE1 out;
    BN_BEGIN_ALLOW_THREADS
    Py_ssize_t size = 0;
    npy_DTYPE1 ai, amean, asum = 0;
    WHILE {
        FOR asum += AI(npy_DTYPE0);
        size += length;
        NEXT
    }
    if (size > ddof) {
        amean = asum / size;
        asum = 0;
        RESET
        WHILE {
            FOR {
                ai = AI(npy_DTYPE0) - amean;
                asum += ai * ai;
            }
            NEXT
        }
        out = asum / (size - ddof);
    }
    else {
        out = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}


static PyObject *
nanvar_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ddof)
{
    Y_INIT(NPY_DTYPE1, npy_DTYPE1)
    INIT
    BN_BEGIN_ALLOW_THREADS
    npy_DTYPE1 ai, asum, amean;
    npy_DTYPE1 length_inv = 1.0 / length;
    npy_DTYPE1 length_ddof_inv = 1.0 / (length - ddof);
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = BN_NAN;
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(npy_DTYPE0);
            if (length > ddof) {
                amean = asum * length_inv;
                asum = 0;
                FOR {
                    ai = AI(npy_DTYPE0) - amean;
                    asum += ai * ai;
                }
                asum = asum * length_ddof_inv;
            }
            else {
                asum = BN_NAN;
            }
            YI = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
nanvar(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nanvar",
                   args,
                   kwds,
                   nanvar_all_float64,
                   nanvar_all_float32,
                   nanvar_all_int64,
                   nanvar_all_int32,
                   nanvar_one_float64,
                   nanvar_one_float32,
                   nanvar_one_int64,
                   nanvar_one_int32,
                   0, 0, 1);
}

/* nanmin ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmin_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ignore)
{
    INIT
    npy_DTYPE0 ai, amin = BN_INFINITY;
    int allnan = 1;
    if (PyArray_SIZE(a) == 0) {
        VALUE_ERR("numpy.nanmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai <= amin) {
                amin = ai;
                allnan = 0;
            }
        }
        NEXT
    }
    if (allnan) amin = BN_NAN;
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(amin);
}


static PyObject *
nanmin_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ignore)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    npy_DTYPE0 ai, amin;
    int allnan;
    if (length == 0) {
        VALUE_ERR("numpy.nanmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        amin = BN_INFINITY;
        allnan = 1;
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai <= amin) {
                amin = ai;
                allnan = 0;
            }
        }
        if (allnan) amin = BN_NAN;
        YI = amin;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nanmin_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ignore)
{
    INIT
    npy_DTYPE0 ai, amin = NPY_MAX_DTYPE0;
    if (PyArray_SIZE(a) == 0) {
        VALUE_ERR("numpy.nanmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai <= amin) amin = ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(amin);
}


static PyObject *
nanmin_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ignore)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    npy_DTYPE0 ai, amin;
    if (length == 0) {
        VALUE_ERR("numpy.nanmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        amin = NPY_MAX_DTYPE0;
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai <= amin) amin = ai;
        }
        YI = amin;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
nanmin(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nanmin",
                   args,
                   kwds,
                   nanmin_all_float64,
                   nanmin_all_float32,
                   nanmin_all_int64,
                   nanmin_all_int32,
                   nanmin_one_float64,
                   nanmin_one_float32,
                   nanmin_one_int64,
                   nanmin_one_int32,
                   0, 0, 0);
}

/* nanmax ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
nanmax_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ignore)
{
    INIT
    npy_DTYPE0 ai, amax = -BN_INFINITY;
    int allnan = 1;
    if (PyArray_SIZE(a)== 0) {
        VALUE_ERR("numpy.nanmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai >= amax) {
                amax = ai;
                allnan = 0;
            }
        }
        NEXT
    }
    if (allnan) amax = BN_NAN;
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(amax);
}


static PyObject *
nanmax_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ignore)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    npy_DTYPE0 ai, amax = 0;
    int allnan;
    if (length == 0) {
        VALUE_ERR("numpy.nanmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        amax = -BN_INFINITY;
        allnan = 1;
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai >= amax) {
                amax = ai;
                allnan = 0;
            }
        }
        if (allnan) amax = BN_NAN;
        YI = amax;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
nanmax_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ignore)
{
    INIT
    npy_DTYPE0 ai, amax = NPY_MIN_DTYPE0;
    if (PyArray_SIZE(a)== 0) {
        VALUE_ERR("numpy.nanmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai >= amax) amax = ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(amax);
}


static PyObject *
nanmax_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ignore)
{
    Y_INIT(NPY_DTYPE0, npy_DTYPE0)
    INIT
    npy_DTYPE0 ai, amax;
    if (length == 0) {
        VALUE_ERR("numpy.nanmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        amax = NPY_MIN_DTYPE0;
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai >= amax) amax = ai;
        }
        YI = amax;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
nanmax(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nanmax",
                   args,
                   kwds,
                   nanmax_all_float64,
                   nanmax_all_float32,
                   nanmax_all_int64,
                   nanmax_all_int32,
                   nanmax_one_float64,
                   nanmax_one_float32,
                   nanmax_one_int64,
                   nanmax_one_int32,
                   0, 0, 0);
}

/* anynan ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
anynan_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ignore)
{
    INIT
    int f = 0;
    npy_DTYPE0 ai;
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(npy_DTYPE0);
            if (ai != ai) {
                f = 1;
                break;
            }
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (f) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}


static PyObject *
anynan_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ignore)
{
    Y_INIT(NPY_BOOL, npy_uint8)
    INIT
    BN_BEGIN_ALLOW_THREADS
    int f;
    npy_DTYPE0 ai;
    if (length == 0) {
        Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
        FOR YI = 0;
    }
    else {
        WHILE {
            f = 0;
            FOR {
                ai = AI(npy_DTYPE0);
                if (ai != ai) {
                    f = 1;
                    break;
                }
            }
            YI = f;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64'], ['int32']] */
static PyObject *
anynan_all_DTYPE0(PyArrayObject *a, int axis, Py_ssize_t stride,
                  Py_ssize_t length, int ndim, int ignore)
{
    Py_RETURN_FALSE;
}


static PyObject *
anynan_one_DTYPE0(PyArrayObject *a,
                  int axis,
                  Py_ssize_t stride,
                  Py_ssize_t length,
                  int ndim,
                  npy_intp* yshape,
                  int ignore)
{
    Y_INIT(NPY_BOOL, npy_uint8)
    Py_ssize_t _i;
    BN_BEGIN_ALLOW_THREADS
    Py_ssize_t length = PyArray_SIZE((PyArrayObject *)y);
    FOR YI = 0;
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


static PyObject *
anynan(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("anynan",
                   args,
                   kwds,
                   anynan_all_float64,
                   anynan_all_float32,
                   anynan_all_int64,
                   anynan_all_int32,
                   anynan_one_float64,
                   anynan_one_float32,
                   anynan_one_int64,
                   anynan_one_int32,
                   0, 0, 0);
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

static BN_INLINE int
parse_args(PyObject *args,
           PyObject *kwds,
           int has_ddof,
           PyObject **arr,
           PyObject **axis,
           PyObject **ddof)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (kwds) {
        int nkwds_found = 0;
        const Py_ssize_t nkwds = PyDict_Size(kwds);
        PyObject *tmp;
        switch (nargs) {
            case 2:
                if (has_ddof) {
                    *axis = PyTuple_GET_ITEM(args, 1);
                } else {
                    TYPE_ERR("wrong number of arguments");
                    return 0;
                }
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
                tmp = PyDict_GetItem(kwds, pystr_axis);
                if (tmp != NULL) {
                    *axis = tmp;
                    nkwds_found++;
                }
            case 2:
                if (has_ddof) {
                    tmp = PyDict_GetItem(kwds, pystr_ddof);
                    if (tmp != NULL) {
                        *ddof = tmp;
                        nkwds_found++;
                    }
                    break;
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
        if (nargs + nkwds_found > 2 + has_ddof) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    }
    else {
        switch (nargs) {
            case 3:
                if (has_ddof) {
                    *ddof = PyTuple_GET_ITEM(args, 2);
                } else {
                    TYPE_ERR("wrong number of arguments");
                    return 0;
                }
            case 2:
                *axis = PyTuple_GET_ITEM(args, 1);
            case 1:
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
reducer(char *name,
        PyObject *args,
        PyObject *kwds,
        fall_t fall_float64,
        fall_t fall_float32,
        fall_t fall_int64,
        fall_t fall_int32,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32,
        int ravel,
        int copy,
        int has_ddof)
{

    int ndim;
    int reduce_all = 0;
    int axis;
    int dtype;
    int ddof;

    Py_ssize_t i;
    Py_ssize_t j = 0;
    Py_ssize_t stride;
    Py_ssize_t length;

    npy_intp *shape;
    npy_intp *strides;

    PyArrayObject *a;

    PyObject *arr_obj = NULL;
    PyObject *axis_obj = Py_None;
    PyObject *ddof_obj = NULL;

    if (!parse_args(args, kwds, has_ddof, &arr_obj, &axis_obj, &ddof_obj)) {
        return NULL;
    }

    /* convert to array if necessary */
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

    ndim = PyArray_NDIM(a);

    /* defend against 0d beings */
    if (ndim == 0) {
        if (axis_obj == Py_None ||
            axis_obj == PyInt_FromLong(0) ||
            axis_obj == PyInt_FromLong(-1))
            return slow(name, args, kwds);
        else {
            VALUE_ERR("axis out of bounds for 0d input");
            return NULL;
        }
    }

    /* does user want to reduce over all axes? */
    if (axis_obj == Py_None) {
        reduce_all = 1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer or None");
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

    /* ddof */
    if (ddof_obj == NULL) {
        ddof = 0;
    }
    else {
        ddof = PyArray_PyIntAsInt(ddof_obj);
        if (error_converting(ddof)) {
            TYPE_ERR("`ddof` must be an integer");
            return NULL;
        }
    }

    dtype = PyArray_TYPE(a);
    shape = PyArray_SHAPE(a);
    strides = PyArray_STRIDES(a);

    if (reduce_all == 1) {
        /* we are reducing the array along all axes */
        if (ravel == 0) {
            axis = 0;
            stride = strides[0];
            for (i=1; i < ndim; i++) {
                if (strides[i] < stride) {
                    axis = i;
                    stride = strides[i];
                }
            }
            length = shape[axis];
        }
        else {
            /* TODO a = PyArray_Ravel(a, NPY_ANYORDER);*/
            axis = 0; /* TODO find min stride axis */
            stride = PyArray_STRIDE(a, 0);
            length = PyArray_SIZE(a);
        }
        if (dtype == NPY_FLOAT64) {
            return fall_float64(a, axis, stride, length, ndim, ddof);
        }
        else if (dtype == NPY_FLOAT32) {
            return fall_float32(a, axis, stride, length, ndim, ddof);
        }
        else if (dtype == NPY_INT64) {
            return fall_int64(a, axis, stride, length, ndim, ddof);
        }
        else if (dtype == NPY_INT32) {
            return fall_int32(a, axis, stride, length, ndim, ddof);
        }
        else {
            return slow(name, args, kwds);
        }
    }
    else {

        /* we are reducing an array with ndim > 1 over a single axis */

        npy_intp yshape[ndim - 1];

        /* shape of output */
        for (i=0; i < ndim; i++) {
            if (i != axis) yshape[j++] = shape[i];
        }

        stride = strides[axis];
        length = shape[axis];

        if (dtype == NPY_FLOAT64) {
            return fone_float64(a, axis, stride, length, ndim, yshape, ddof);
        }
        else if (dtype == NPY_FLOAT32) {
            return fone_float32(a, axis, stride, length, ndim, yshape, ddof);
        }
        else if (dtype == NPY_INT64) {
            return fone_int64(a, axis, stride, length, ndim, yshape, ddof);
        }
        else if (dtype == NPY_INT32) {
            return fone_int32(a, axis, stride, length, ndim, yshape, ddof);
        }
        else {
            return slow(name, args, kwds);
        }
    }

}

/* docstrings ------------------------------------------------------------- */

static char reduce_doc[] =
"Bottleneck functions that reduce the input array along a specified axis.";

static char nansum_doc[] =
/* MULTILINE STRING BEGIN
nansum(arr, axis=None)

Sum of array elements along given axis treating NaNs as zero.

The data type (dtype) of the output is the same as the input. On 64-bit
operating systems, 32-bit input is NOT upcast to 64-bit accumulator and
return values.

Parameters
----------
arr : array_like
    Array containing numbers whose sum is desired. If `arr` is not an
    array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the sum is computed. The default (axis=None) is to
    compute the sum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned.

Notes
-----
No error is raised on overflow.

If positive or negative infinity are present the result is positive or
negative infinity. But if both positive and negative infinity are present,
the result is Not A Number (NaN).

Examples
--------
>>> bn.nansum(1)
1
>>> bn.nansum([1])
1
>>> bn.nansum([1, np.nan])
1.0
>>> a = np.array([[1, 1], [1, np.nan]])
>>> bn.nansum(a)
3.0
>>> bn.nansum(a, axis=0)
array([ 2.,  1.])

When positive infinity and negative infinity are present:

>>> bn.nansum([1, np.nan, np.inf])
inf
>>> bn.nansum([1, np.nan, np.NINF])
-inf
>>> bn.nansum([1, np.nan, np.inf, np.NINF])
nan

MULTILINE STRING END */

static char nanmean_doc[] =
/* MULTILINE STRING BEGIN
nanmean(arr, axis=None)

Mean of array elements along given axis ignoring NaNs.

`float64` intermediate and return values are used for integer inputs.

Parameters
----------
arr : array_like
    Array containing numbers whose mean is desired. If `arr` is not an
    array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the means are computed. The default (axis=None) is to
    compute the mean of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned.
    `float64` intermediate and return values are used for integer inputs.

See also
--------
bottleneck.nanmedian: Median along specified axis, ignoring NaNs.

Notes
-----
No error is raised on overflow. (The sum is computed and then the result
is divided by the number of non-NaN elements.)

If positive or negative infinity are present the result is positive or
negative infinity. But if both positive and negative infinity are present,
the result is Not A Number (NaN).

Examples
--------
>>> bn.nanmean(1)
1.0
>>> bn.nanmean([1])
1.0
>>> bn.nanmean([1, np.nan])
1.0
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.nanmean(a)
2.0
>>> bn.nanmean(a, axis=0)
array([ 1.,  4.])

When positive infinity and negative infinity are present:

>>> bn.nanmean([1, np.nan, np.inf])
inf
>>> bn.nanmean([1, np.nan, np.NINF])
-inf
>>> bn.nanmean([1, np.nan, np.inf, np.NINF])
nan

MULTILINE STRING END */

static char nanstd_doc[] =
/* MULTILINE STRING BEGIN
nanstd(arr, axis=None, ddof=0)

Standard deviation along the specified axis, ignoring NaNs.

`float64` intermediate and return values are used for integer inputs.

Instead of a faster one-pass algorithm, a more stable two-pass algorithm
is used.

An example of a one-pass algorithm:

    >>> np.sqrt((arr*arr).mean() - arr.mean()**2)

An example of a two-pass algorithm:

    >>> np.sqrt(((arr - arr.mean())**2).mean())

Note in the two-pass algorithm the mean must be found (first pass) before
the squared deviation (second pass) can be found.

Parameters
----------
arr : array_like
    Input array. If `arr` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the standard deviation is computed. The default
    (axis=None) is to compute the standard deviation of the flattened
    array.
ddof : int, optional
    Means Delta Degrees of Freedom. The divisor used in calculations
    is ``N - ddof``, where ``N`` represents the number of non-NaN elements.
    By default `ddof` is zero.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned.
    `float64` intermediate and return values are used for integer inputs.
    If ddof is >= the number of non-NaN elements in a slice or the slice
    contains only NaNs, then the result for that slice is NaN.

See also
--------
bottleneck.nanvar: Variance along specified axis ignoring NaNs

Notes
-----
If positive or negative infinity are present the result is Not A Number
(NaN).

Examples
--------
>>> bn.nanstd(1)
0.0
>>> bn.nanstd([1])
0.0
>>> bn.nanstd([1, np.nan])
0.0
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.nanstd(a)
1.4142135623730951
>>> bn.nanstd(a, axis=0)
array([ 0.,  0.])

When positive infinity or negative infinity are present NaN is returned:

>>> bn.nanstd([1, np.nan, np.inf])
nan

MULTILINE STRING END */

static char nanvar_doc[] =
/* MULTILINE STRING BEGIN
nanvar(arr, axis=None, ddof=0)

Variance along the specified axis, ignoring NaNs.

`float64` intermediate and return values are used for integer inputs.

Instead of a faster one-pass algorithm, a more stable two-pass algorithm
is used.

An example of a one-pass algorithm:

    >>> (arr*arr).mean() - arr.mean()**2

An example of a two-pass algorithm:

    >>> ((arr - arr.mean())**2).mean()

Note in the two-pass algorithm the mean must be found (first pass) before
the squared deviation (second pass) can be found.

Parameters
----------
arr : array_like
    Input array. If `arr` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the variance is computed. The default (axis=None) is
    to compute the variance of the flattened array.
ddof : int, optional
    Means Delta Degrees of Freedom. The divisor used in calculations
    is ``N - ddof``, where ``N`` represents the number of non_NaN elements.
    By default `ddof` is zero.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis
    removed. If `arr` is a 0-d array, or if axis is None, a scalar is
    returned. `float64` intermediate and return values are used for
    integer inputs. If ddof is >= the number of non-NaN elements in a
    slice or the slice contains only NaNs, then the result for that slice
    is NaN.

See also
--------
bottleneck.nanstd: Standard deviation along specified axis ignoring NaNs.

Notes
-----
If positive or negative infinity are present the result is Not A Number
(NaN).

Examples
--------
>>> bn.nanvar(1)
0.0
>>> bn.nanvar([1])
0.0
>>> bn.nanvar([1, np.nan])
0.0
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.nanvar(a)
2.0
>>> bn.nanvar(a, axis=0)
array([ 0.,  0.])

When positive infinity or negative infinity are present NaN is returned:

>>> bn.nanvar([1, np.nan, np.inf])
nan

MULTILINE STRING END */

static char nanmin_doc[] =
/* MULTILINE STRING BEGIN
nanmin(arr, axis=None)

Minimum values along specified axis, ignoring NaNs.

When all-NaN slices are encountered, NaN is returned for that slice.

Parameters
----------
arr : array_like
    Input array. If `arr` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the minimum is computed. The default (axis=None) is
    to compute the minimum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned. The
    same dtype as `arr` is returned.

See also
--------
bottleneck.nanmax: Maximum along specified axis, ignoring NaNs.
bottleneck.nanargmin: Indices of minimum values along axis, ignoring NaNs.

Examples
--------
>>> bn.nanmin(1)
1
>>> bn.nanmin([1])
1
>>> bn.nanmin([1, np.nan])
1.0
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.nanmin(a)
1.0
>>> bn.nanmin(a, axis=0)
array([ 1.,  4.])

MULTILINE STRING END */

static char nanmax_doc[] =
/* MULTILINE STRING BEGIN
nanmax(arr, axis=None)

Maximum values along specified axis, ignoring NaNs.

When all-NaN slices are encountered, NaN is returned for that slice.

Parameters
----------
arr : array_like
    Input array. If `arr` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the maximum is computed. The default (axis=None) is
    to compute the maximum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `arr`, with the specified axis removed.
    If `arr` is a 0-d array, or if axis is None, a scalar is returned. The
    same dtype as `arr` is returned.

See also
--------
bottleneck.nanmin: Minimum along specified axis, ignoring NaNs.
bottleneck.nanargmax: Indices of maximum values along axis, ignoring NaNs.

Examples
--------
>>> bn.nanmax(1)
1
>>> bn.nanmax([1])
1
>>> bn.nanmax([1, np.nan])
1.0
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.nanmax(a)
4.0
>>> bn.nanmax(a, axis=0)
array([ 1.,  4.])

MULTILINE STRING END */

static char anynan_doc[] =
/* MULTILINE STRING BEGIN
anynan(arr, axis=None)

Test whether any array element along a given axis is NaN.

Returns the same output as np.isnan(arr).any(axis)

Parameters
----------
arr : array_like
    Input array. If `arr` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which NaNs are searched. The default (`axis` = ``None``)
    is to search for NaNs over a flattened input array.

Returns
-------
y : bool or ndarray
    A boolean or new `ndarray` is returned.

See also
--------
bottleneck.allnan: Test if all array elements along given axis are NaN

Examples
--------
>>> bn.anynan(1)
False
>>> bn.anynan(np.nan)
True
>>> bn.anynan([1, np.nan])
True
>>> a = np.array([[1, 4], [1, np.nan]])
>>> bn.anynan(a)
True
>>> bn.anynan(a, axis=0)
array([False,  True], dtype=bool)

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
reduce_methods[] = {
    {"nansum", (PyCFunction)nansum, VARKEY, nansum_doc},
    {"nanmean", (PyCFunction)nanmean, VARKEY, nanmean_doc},
    {"nanstd", (PyCFunction)nanstd, VARKEY, nanstd_doc},
    {"nanvar", (PyCFunction)nanvar, VARKEY, nanvar_doc},
    {"nanmin", (PyCFunction)nanmin, VARKEY, nanmin_doc},
    {"nanmax", (PyCFunction)nanmax, VARKEY, nanmax_doc},
    {"anynan", (PyCFunction)anynan, VARKEY, anynan_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
reduce_def = {
   PyModuleDef_HEAD_INIT,
   "reduce2",
   reduce_doc,
   -1,
   reduce_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_reduce2(void)
#else
#define RETVAL
initreduce2(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&reduce_def);
    #else
        PyObject *m = Py_InitModule3("reduce2", reduce_methods, reduce_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
