#include <numpy/arrayobject.h>

/*
   Bottleneck iterators are based on ideas from NumPy's PyArray_IterAllButAxis
   and PyArray_ITER_NEXT.
*/

/* one input array ------------------------------------------------------- */

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
    char       *p;
};
typedef struct _iter iter;

static BN_INLINE void
init_iter_one(iter *it, PyArrayObject *a, int axis)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *strides = PyArray_STRIDES(a);

    it->ndim_m2 = ndim - 2;
    it->axis = axis;
    it->its = 0;
    it->nits = 1;
    it->p = PyArray_BYTES(a);

    if (ndim == 0) {
        it->ndim_m2 = -1;
        it->length = 1;
        it->stride = 0;
    }
    else {
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
    }
}

static BN_INLINE void
init_iter_all(iter *it, PyArrayObject *a, int ravel)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *strides = PyArray_STRIDES(a);

    it->axis = 0;
    it->its = 0;
    it->nits = 1;

    if (ndim == 1) {
        it->ndim_m2 = -1;
        it->length = shape[0];
        it->stride = strides[0];
    }
    else if (ndim == 0) {
        it->ndim_m2 = -1;
        it->length = 1;
        it->stride = 0;
    }
    else if (C_CONTIGUOUS(a)) {
        it->ndim_m2 = -1;
        it->axis = ndim - 1;
        it->length = PyArray_SIZE(a);
        it->stride = strides[ndim - 1];
    }
    else if (F_CONTIGUOUS(a)) {
        it->ndim_m2 = -1;
        it->length = PyArray_SIZE(a);
        it->stride = strides[0];
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

    it->p = PyArray_BYTES(a);
}

#define NEXT \
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

/* two input arrays ------------------------------------------------------ */

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

static BN_INLINE void
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
        }
        else {
            it->indices[j] = 0;
            it->astrides[j] = astrides[i];
            it->ystrides[j] = ystrides[i];
            it->shape[j] = shape[i];
            it->nits *= shape[i];
            j++;
        }
    }
}

#define NEXT99 \
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

/* macros used with iterators -------------------------------------------- */

#define  NDIM           it->ndim_m2 + 2
#define  SHAPE          it->shape
#define  SIZE           it->nits * it->length
#define  LENGTH         it->length
#define  INDEX          it->i

#define  WHILE          while (it->its < it->nits)
#define  FOR            for (it->i = 0; it->i < it->length; it->i++)
#define  FOR_REVERSE    for (it->i = it->length - 1; it->i > -1; it->i--)
#define  RESET          it->its = 0;

#define  AI(dtype)      *(dtype *)(it->p + it->i * it->stride)
#define  AX(dtype, x)   *(dtype *)(it->p + x * it->stride)

#define Y_INIT(dt0, dt1) \
    PyObject *y = PyArray_EMPTY(it->ndim_m2 + 1, it->shape, dt0, 0); \
    dt1 *py = (dt1 *)PyArray_DATA((PyArrayObject *)y);

#define YI *py++

#define FILL_Y(value) \
    int i; \
    Py_ssize_t size = PyArray_SIZE((PyArrayObject *)y); \
    for (i = 0; i < size; i++) YI = value;

#define  WHILE99   while (it.its < it.nits)
#define  WHILE099  it.i = 0; while (it.i < min_count - 1)
#define  WHILE199  while (it.i < window)
#define  WHILE299  while (it.i < it.length)

#define  A099(dt)    *(dt *)(it.pa)
#define  AI99(dt)    *(dt *)(it.pa + it.i * it.astride)
#define  AOLD99(dt)  *(dt *)(it.pa + (it.i - window) * it.astride)
#define  AX99(dt, x) *(dt *)(it.pa + x * it.astride)
#define  YI99(dt)    *(dt *)(it.py + it.i++ * it.ystride)
#define  INDEX99     it.i

/* debug stuff ----------------------------------------------------------- */

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
