#include "bottleneck.h"

struct _iter {
    int        ndim;
    int        axis;
    Py_ssize_t length;
    Py_ssize_t stride;
    npy_intp   i;
    npy_intp   its;
    npy_intp   nits;
    npy_intp   indices[NPY_MAXDIMS];
    npy_intp   strides[NPY_MAXDIMS];
    npy_intp   shape_m1[NPY_MAXDIMS];
    char      *p;
};
typedef struct _iter iter;

static BN_INLINE iter *
iter_reduce_one(PyArrayObject *a, int axis)
{
    int i, j = 0;
    npy_intp dim;
    iter *it = malloc(sizeof(iter));

    it->ndim = PyArray_NDIM(a);
    it->axis = axis;
    it->its = 0;
    it->nits = 1;
    it->p = PyArray_DATA(a);

    for (i = 0; i < it->ndim; i++) {
        if (i == axis) {
            it->stride = PyArray_STRIDE(a, i);
            it->length = PyArray_DIM(a, i);
        }
        else {
            it->indices[j] = 0;
            it->strides[j] = PyArray_STRIDE(a, i);
            dim = PyArray_DIM(a, i);
            it->shape_m1[j] = dim - 1;
            it->nits *= dim;
            j++;
        }
    }

    return it;
}

#define NEXT99(it) \
    for (it->i = it->ndim - 2; it->i > -1; it->i--) { \
        if (it->indices[it->i] < it->shape_m1[it->i]) { \
            it->p += it->strides[it->i]; \
            it->indices[it->i]++; \
            break; \
        } \
        it->p -= it->indices[it->i] * it->strides[it->i]; \
        it->indices[it->i] = 0; \
    } \
    it->its++;

#define  WHILE99(it)     while (it->its < it->nits)
#define  FOR99(it)       for (it->i = 0; it->i < it->length; it->i++)
#define  AI99(it, dt)    *(dt*)(it->p + it->i * it->stride)
