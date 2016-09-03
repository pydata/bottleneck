#include "bottleneck.h"

struct _iter {
    Py_ssize_t length;
    int        ndim_m2;
    int        axis;
    npy_intp   i;
    npy_intp   index;
    npy_intp   size;
    npy_intp   indices[NPY_MAXDIMS];
    npy_intp   strides[NPY_MAXDIMS];
    npy_intp   shape_m1[NPY_MAXDIMS];
    char      *p;
};
typedef struct _iter iter;

static BN_INLINE iter *
new_iter(PyArrayObject *a, int axis, int ndim, Py_ssize_t length)
{
    int i, j = 0;
    iter *it = malloc(sizeof(iter));

    it->length = length;
    it->ndim_m2 = ndim - 2;
    it->axis = axis;
    it->index = 0;
    it->size = PyArray_SIZE(a);
    it->p = PyArray_DATA(a);

    if (length != 0) it->size /= length;

    memset(it->indices, 0, (ndim - 1) * sizeof(npy_intp));
    for (i = 0; i < ndim; i++) {
        if (i == axis) continue;
        it->strides[j] = PyArray_STRIDE(a, i);
        it->shape_m1[j] = PyArray_DIM(a, i) - 1;
        j++;
    }

    return it;
}

#define NEXT99 \
    for (it->i = it->ndim_m2; it->i > -1; it->i--) { \
        if (it->indices[it->i] < it->shape_m1[it->i]) { \
            it->p += it->strides[it->i]; \
            it->indices[it->i]++; \
            break; \
        } \
        it->p -= it->indices[it->i] * it->strides[it->i]; \
        it->indices[it->i] = 0; \
    } \
    it->index++;

#define  WHILE99          while (it->index < it->size)
#define  FOR99            for (it->i = 0; it->i < it->length; it->i++)
#define  AI99(dt)         *(dt*)(it->p + it->i * stride)
