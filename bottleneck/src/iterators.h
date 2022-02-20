// Copyright 2010-2019 Keith Goodman
// Copyright 2019 Bottleneck Developers
#ifndef ITERATORS_H_
#define ITERATORS_H_

#include <numpy/arrayobject.h>

/*
   Bottleneck iterators are based on ideas from NumPy's PyArray_IterAllButAxis
   and PyArray_ITER_NEXT.
*/

/* one input array ------------------------------------------------------- */

/* these iterators are used mainly by reduce functions such as nansum */

struct _iter {
    int        ndim_m2; /* ndim - 2 */
    int        axis;    /* axis to not iterate over */
    Py_ssize_t length;  /* a.shape[axis] */
    Py_ssize_t astride; /* a.strides[axis] */
    npy_intp   stride;  /* element-level stride to take in the array */
    npy_intp   i;       /* integer used by some macros */
    npy_intp   its;     /* number of iterations completed */
    npy_intp   nits;    /* number of iterations iterator plans to make */
    npy_intp   indices[NPY_MAXDIMS];  /* current location of iterator */
    npy_intp   astrides[NPY_MAXDIMS]; /* a.strides, a.strides[axis] removed */
    npy_intp   shape[NPY_MAXDIMS];    /* a.shape, a.shape[axis] removed */
    char       *pa;     /* pointer to data corresponding to indices */
    PyArrayObject *a_ravel; /* NULL or pointer to ravelled input array */
};
typedef struct _iter iter;

static inline void
init_iter_one(iter *it, PyArrayObject *a, int axis)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *strides = PyArray_STRIDES(a);
    const npy_intp item_size = PyArray_ITEMSIZE(a);

    it->axis = axis;
    it->its = 0;
    it->nits = 1;
    it->pa = PyArray_BYTES(a);

    it->ndim_m2 = -1;
    it->length = 1;
    it->astride = 0;

    if (ndim != 0) {
        it->ndim_m2 = ndim - 2;
        for (i = 0; i < ndim; i++) {
            if (i == axis) {
                it->astride = strides[i];
                it->length = shape[i];
            } else {
                it->indices[j] = 0;
                it->astrides[j] = strides[i];
                it->shape[j] = shape[i];
                it->nits *= shape[i];
                j++;
            }
        }
    }
    it->stride = it->astride / item_size;
}

/*
 * If both ravel != 0 and it.a_ravel != NULL then you are responsible for
 * calling Py_DECREF(it.a_ravel) after you are done with the iterator.
 * See nanargmin for an example.
 */
static inline void
init_iter_all(iter *it, PyArrayObject *a, int ravel, int anyorder)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *strides = PyArray_STRIDES(a);
    const npy_intp item_size = PyArray_ITEMSIZE(a);

    it->axis = 0;
    it->its = 0;
    it->nits = 1;
    it->a_ravel = NULL;

    /* The fix for relaxed strides checking in numpy and the fix for
     * issue #183 has left this if..else tree in need of a refactor from the
     * the ground up */
    if (ndim == 1) {
        it->ndim_m2 = -1;
        it->length = shape[0];
        it->astride = strides[0];
    } else if (ndim == 0) {
        it->ndim_m2 = -1;
        it->length = 1;
        it->astride = 0;
    } else if (C_CONTIGUOUS(a) && !F_CONTIGUOUS(a)) {
        /* The &&! in the next two else ifs is to deal with relaxed
         * stride checking introduced in numpy 1.12.0; see gh #161 */
        it->ndim_m2 = -1;
        it->axis = ndim - 1;
        it->length = PyArray_SIZE(a);
        it->astride = 0;
        for (i=ndim-1; i > -1; i--) {
            /* protect against length zero  strides such as in
             * np.ones((2, 2))[..., np.newaxis] */
            if (strides[i] == 0) {
                continue;
            }
            it->astride = strides[i];
            break;
       }
    } else if (F_CONTIGUOUS(a) && !C_CONTIGUOUS(a)) {
        if (anyorder || !ravel) {
            it->ndim_m2 = -1;
            it->length = PyArray_SIZE(a);
            it->astride = 0;
            for (i=0; i < ndim; i++) {
                /* protect against length zero  strides such as in
                 * np.ones((2, 2), order='F')[np.newaxis, ...] */
                if (strides[i] == 0) {
                    continue;
                }
                it->astride = strides[i];
                break;
           }
        } else {
            it->ndim_m2 = -1;
            if (anyorder) {
                a = (PyArrayObject *)PyArray_Ravel(a, NPY_ANYORDER);
            } else {
                a = (PyArrayObject *)PyArray_Ravel(a, NPY_CORDER);
            }
            it->a_ravel = a;
            it->length = PyArray_DIM(a, 0);
            it->astride = PyArray_STRIDE(a, 0);
        }
    } else if (ravel) {
        it->ndim_m2 = -1;
        if (anyorder) {
            a = (PyArrayObject *)PyArray_Ravel(a, NPY_ANYORDER);
        } else {
            a = (PyArrayObject *)PyArray_Ravel(a, NPY_CORDER);
        }
        it->a_ravel = a;
        it->length = PyArray_DIM(a, 0);
        it->astride = PyArray_STRIDE(a, 0);
    } else {
        it->ndim_m2 = ndim - 2;
        it->astride = strides[0];
        for (i = 1; i < ndim; i++) {
            if (strides[i] < it->astride) {
                it->astride = strides[i];
                it->axis = i;
            }
        }
        it->length = shape[it->axis];
        for (i = 0; i < ndim; i++) {
            if (i != it->axis) {
                it->indices[j] = 0;
                it->astrides[j] = strides[i];
                it->shape[j] = shape[i];
                it->nits *= shape[i];
                j++;
            }
        }
    }

    it->stride = it->astride / item_size;
    it->pa = PyArray_BYTES(a);
}

#define NEXT \
    for (it.i = it.ndim_m2; it.i > -1; it.i--) { \
        if (it.indices[it.i] < it.shape[it.i] - 1) { \
            it.pa += it.astrides[it.i]; \
            it.indices[it.i]++; \
            break; \
        } \
        it.pa -= it.indices[it.i] * it.astrides[it.i]; \
        it.indices[it.i] = 0; \
    } \
    it.its++;

/* two input arrays ------------------------------------------------------ */

/* this iterator is used mainly by moving window functions such as move_sum */

struct _iter2 {
    int        ndim_m2;
    int        axis;
    Py_ssize_t length;
    Py_ssize_t astride;
    Py_ssize_t ystride;
    npy_intp   i;
    npy_intp   its;
    npy_intp   nits;
    npy_intp   indices[NPY_MAXDIMS];
    npy_intp   astrides[NPY_MAXDIMS];
    npy_intp   ystrides[NPY_MAXDIMS];
    npy_intp   shape[NPY_MAXDIMS];
    char       *pa;
    char       *py;
};
typedef struct _iter2 iter2;

static inline void
init_iter2(iter2 *it, PyArrayObject *a, PyObject *y, int axis)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *astrides = PyArray_STRIDES(a);
    const npy_intp *ystrides = PyArray_STRIDES((PyArrayObject *)y);

    /* to avoid compiler warning of uninitialized variables */
    it->length = 0;
    it->astride = 0;
    it->ystride = 0;

    it->ndim_m2 = ndim - 2;
    it->axis = axis;
    it->its = 0;
    it->nits = 1;
    it->pa = PyArray_BYTES(a);
    it->py = PyArray_BYTES((PyArrayObject *)y);

    for (i = 0; i < ndim; i++) {
        if (i == axis) {
            it->astride = astrides[i];
            it->ystride = ystrides[i];
            it->length = shape[i];
        } else {
            it->indices[j] = 0;
            it->astrides[j] = astrides[i];
            it->ystrides[j] = ystrides[i];
            it->shape[j] = shape[i];
            it->nits *= shape[i];
            j++;
        }
    }
}

#define NEXT2 \
    for (it.i = it.ndim_m2; it.i > -1; it.i--) { \
        if (it.indices[it.i] < it.shape[it.i] - 1) { \
            it.pa += it.astrides[it.i]; \
            it.py += it.ystrides[it.i]; \
            it.indices[it.i]++; \
            break; \
        } \
        it.pa -= it.indices[it.i] * it.astrides[it.i]; \
        it.py -= it.indices[it.i] * it.ystrides[it.i]; \
        it.indices[it.i] = 0; \
    } \
    it.its++;

/* three input arrays ---------------------------------------------------- */

/* this iterator is used mainly by rankdata and nanrankdata */

struct _iter3 {
    int        ndim_m2;
    int        axis;
    Py_ssize_t length;
    Py_ssize_t astride;
    Py_ssize_t ystride;
    Py_ssize_t zstride;
    npy_intp   i;
    npy_intp   its;
    npy_intp   nits;
    npy_intp   indices[NPY_MAXDIMS];
    npy_intp   astrides[NPY_MAXDIMS];
    npy_intp   ystrides[NPY_MAXDIMS];
    npy_intp   zstrides[NPY_MAXDIMS];
    npy_intp   shape[NPY_MAXDIMS];
    char       *pa;
    char       *py;
    char       *pz;
};
typedef struct _iter3 iter3;

static inline void
init_iter3(iter3 *it, PyArrayObject *a, PyObject *y, PyObject *z, int axis)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *astrides = PyArray_STRIDES(a);
    const npy_intp *ystrides = PyArray_STRIDES((PyArrayObject *)y);
    const npy_intp *zstrides = PyArray_STRIDES((PyArrayObject *)z);

    /* to avoid compiler warning of uninitialized variables */
    it->length = 0;
    it->astride = 0;
    it->ystride = 0;
    it->zstride = 0;

    it->ndim_m2 = ndim - 2;
    it->axis = axis;
    it->its = 0;
    it->nits = 1;
    it->pa = PyArray_BYTES(a);
    it->py = PyArray_BYTES((PyArrayObject *)y);
    it->pz = PyArray_BYTES((PyArrayObject *)z);

    for (i = 0; i < ndim; i++) {
        if (i == axis) {
            it->astride = astrides[i];
            it->ystride = ystrides[i];
            it->zstride = zstrides[i];
            it->length = shape[i];
        } else {
            it->indices[j] = 0;
            it->astrides[j] = astrides[i];
            it->ystrides[j] = ystrides[i];
            it->zstrides[j] = zstrides[i];
            it->shape[j] = shape[i];
            it->nits *= shape[i];
            j++;
        }
    }
}

#define NEXT3 \
    for (it.i = it.ndim_m2; it.i > -1; it.i--) { \
        if (it.indices[it.i] < it.shape[it.i] - 1) { \
            it.pa += it.astrides[it.i]; \
            it.py += it.ystrides[it.i]; \
            it.pz += it.zstrides[it.i]; \
            it.indices[it.i]++; \
            break; \
        } \
        it.pa -= it.indices[it.i] * it.astrides[it.i]; \
        it.py -= it.indices[it.i] * it.ystrides[it.i]; \
        it.pz -= it.indices[it.i] * it.zstrides[it.i]; \
        it.indices[it.i] = 0; \
    } \
    it.its++;

/* macros used with iterators -------------------------------------------- */

/* most of these macros assume iterator is named `it` */

#define  NDIM           it.ndim_m2 + 2
#define  SHAPE          it.shape
#define  SIZE           it.nits * it.length
#define  LENGTH         it.length
#define  INDEX          it.i

#define  WHILE          while (it.its < it.nits)
#define  WHILE0         it.i = 0; while (it.i < min_count - 1)
#define  WHILE1         while (it.i < window)
#define  WHILE2         while (it.i < it.length)

#define  FOR            for (it.i = 0; it.i < it.length; it.i++)
#define  FOR_REVERSE    for (it.i = it.length - 1; it.i > -1; it.i--)

#define  RESET          it.its = 0;

#define  PA(dtype)      (npy_##dtype *)(it.pa)

#define  A0(dtype)      *(npy_##dtype *)(it.pa)
#define  AI(dtype)      *(npy_##dtype *)(it.pa + it.i * it.astride)
#define  AX(dtype, x)   *(npy_##dtype *)(it.pa + (x) * it.astride)
#define  AOLD(dtype)    *(npy_##dtype *)(it.pa + (it.i - window) * it.astride)

#define  SI(pa)         pa[it.i * it.stride]    

#define  YPP            *py++
#define  YI(dtype)      *(npy_##dtype *)(it.py + it.i++ * it.ystride)
#define  YX(dtype, x)   *(npy_##dtype *)(it.py + (x) * it.ystride)

#define  ZX(dtype, x)   *(npy_##dtype *)(it.pz + (x) * it.zstride)

#define FILL_Y(value) \
    npy_intp _i; \
    npy_intp size = PyArray_SIZE((PyArrayObject *)y); \
    for (_i = 0; _i < size; _i++) { \
        YPP = value; \
    }

#endif  // ITERATORS_H_
