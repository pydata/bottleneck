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

void
print_array(npy_intp *array, npy_intp length)
{
    int i;
    for (i = 0; i < length; i++) {
        printf("%zd ", array[i]);
    }
    printf("\n");
}

void
print_iter(iter *it)
{
    npy_intp length = it->ndim_m2 + 1;
    printf("-------------------------\n");
    printf("ndim_m2  %d\n", it->ndim_m2);
    printf("axis     %d\n", it->axis);
    printf("length   %zd\n", it->length);
    printf("stride   %zd\n", it->stride);
    printf("i        %zd\n", it->i);
    printf("its      %zd\n", it->its);
    printf("nits     %zd\n", it->nits);
    printf("indices  ");
    print_array(it->indices, length);
    printf("strides  ");
    print_array(it->strides, length);
    printf("shape    ");
    print_array(it->shape, length);
    printf("-------------------------\n");
}

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

static BN_INLINE iter *
new_iter_all(PyArrayObject *a, int ravel)
{
    int i, j = 0;
    int ndim = PyArray_NDIM(a);
    iter *it = malloc(sizeof(iter));
    npy_intp *shape;
    npy_intp *strides;

    it->axis = 0;
    it->its = 0;
    it->nits = 1;

    if (ndim == 1) {
        it->ndim_m2 = -1;
        it->length = PyArray_DIM(a, 0);
        it->stride = PyArray_STRIDE(a, 0);
    }
    else if (C_CONTIGUOUS(a)) {
        it->ndim_m2 = -1;
        it->axis = ndim - 1;
        it->length = PyArray_SIZE(a);
        it->stride = PyArray_STRIDE(a, ndim - 1);
    }
    else if (F_CONTIGUOUS(a)) {
        it->ndim_m2 = -1;
        it->length = PyArray_SIZE(a);
        it->stride = PyArray_STRIDE(a, 0);
    }
    else if (ravel) {
        it->ndim_m2 = -1;
        a = (PyArrayObject *)PyArray_Ravel(a, NPY_ANYORDER);
        Py_DECREF(a);
        it->length = PyArray_DIM(a, 0);
        it->stride = PyArray_STRIDE(a, 0);
    }
    else {
        it->ndim_m2 = ndim - 2;
        shape = PyArray_SHAPE(a);
        strides = PyArray_STRIDES(a);
        it->stride = strides[0];
        for (i = 1; i < ndim; i++) {
            if (strides[i] < it->stride) {
                it->stride = strides[i];
                it->axis = i;
            }
        }
        it->length = shape[it->axis];
        for (i = 0; i < ndim; i++) {
            if (i != it->axis) {
                it->indices[j] = 0;
                it->strides[j] = strides[i];
                it->shape[j] = shape[i];
                it->nits *= shape[i];
                j++;
            }
        }
    }

    it->p = PyArray_DATA(a);

    return it;
}

#define NEXT99 \
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

#define  WHILE99        while (it->its < it->nits)
#define  FOR99          for (it->i = 0; it->i < it->length; it->i++)
#define  FOR_REVERSE99  for (it->i = it->length - 1; it->i > -1; it->i--)
#define  AI99(dt)      *(dt*)(it->p + it->i * it->stride)
#define  AX99(dt, x)   *(dt*)(it->p + x * it->stride)
#define  RESET99        it->its = 0;

#define  NDIM           it->ndim_m2 + 2
#define  SHAPE          it->shape
#define  SIZE           it->nits * it->length
#define  LENGTH         it->length
#define  ITER_I         it->i

#define Y_INIT99(dt0, dt1) \
    PyObject *y = PyArray_EMPTY(it->ndim_m2 + 1, it->shape, dt0, 0); \
    dt1 *py = (dt1 *)PyArray_DATA((PyArrayObject *)y);
