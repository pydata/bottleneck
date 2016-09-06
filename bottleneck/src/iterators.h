#include <numpy/arrayobject.h>

struct _iter {
    int        ndim_m2;
    int        axis;
    Py_ssize_t length;
    Py_ssize_t stride;
    npy_intp   i;
    npy_intp   its;
    npy_intp   nits;
    npy_intp   indices[NPY_MAXDIMS];
    npy_intp   strides[NPY_MAXDIMS];
    npy_intp   shape[NPY_MAXDIMS];
    char      *p;
};
typedef struct _iter iter;

static BN_INLINE iter *
new_iter(PyArrayObject *a, int axis)
{
    int i, j = 0;
    iter *it = malloc(sizeof(iter));
    int ndim = PyArray_NDIM(a);
    npy_intp *shape = PyArray_SHAPE(a);
    npy_intp *strides = PyArray_STRIDES(a);

    it->ndim_m2 = ndim - 2;
    it->axis = axis;
    it->its = 0;
    it->nits = 1;
    it->p = PyArray_DATA(a);

    for (i = 0; i < ndim; i++) {
        if (i == axis) {
            it->stride = strides[i];
            it->length = shape[i];
        }
        else {
            it->indices[j] = 0;
            it->strides[j] = strides[i];
            it->shape[j] = shape[i];
            it->nits *= shape[i];
            j++;
        }
    }

    return it;
}

#define NEXT99(it) \
    for (it->i = it->ndim_m2; it->i > -1; it->i--) { \
        if (it->indices[it->i] < it->shape[it->i] - 1) { \
            it->p += it->strides[it->i]; \
            it->indices[it->i]++; \
            break; \
        } \
        it->p -= it->indices[it->i] * it->strides[it->i]; \
        it->indices[it->i] = 0; \
    } \
    it->its++;

#define  WHILE99(it)       while (it->its < it->nits)
#define  FOR99(it)         for (it->i = 0; it->i < it->length; it->i++)
#define  FOR_REVERSE99(it) for (it->i = it->length - 1; it->i > -1; it->i--)
#define  AIT(dt, it)      *(dt*)(it->p + it->i * it->stride)
#define  AX99(dt, x)      *(dt*)(it->p + x * it->stride)

#define  ITER_LENGTH(it)   it->length
#define  ITER_I(it)        it->i

#define Y_INIT99(dt0, dt1) \
    PyObject *y = PyArray_EMPTY(it->ndim_m2 + 1, it->shape, dt0, 0); \
    dt1 *py = (dt1 *)PyArray_DATA((PyArrayObject *)y);
