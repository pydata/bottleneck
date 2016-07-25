#include "bottleneck.h"

/*
 * The moving window functions can handle input arrays of arbitrary dimension
 * (ndim > 0). To accomplish that Bottleneck borrows ideas from NumPy.
 *
 * Loosely speaking, Bottleneck's INIT macro is PyArray_IterAllButAxis and
 * NEXT is PyArray_ITER_NEXT. If you strip out everything that is not needed
 * for moving window functions from those NumPy functions and apply a few
 * optimizations such as iterating over the input and output arrays in the
 * same loop (as opposed to using two iterators), then you pretty much end
 * up with INIT and NEXT.
 *
 * Being much less general than NumPy's iterators, Bottleneck's moving window
 * iterators have less overhead and are faster.
 *
 * If you are not familiar with the internal workings of
 * PyArray_IterAllButAxis and PyArray_ITER_NEXT then Bottleneck's iterators
 * may at first be difficult to understand. NumPy gives a nice description
 * of how their N-dimensional iterators work:
 *
 * http://docs.scipy.org/doc/numpy/reference/internals.code-explanations.html
 */

/* all moving window functions use the following three macros */
#define INIT(dt) \
    PyObject *y = PyArray_EMPTY(ndim, shape, dt, 0); \
    BN_BEGIN_ALLOW_THREADS \
    Py_ssize_t i; \
    char *py = PyArray_BYTES((PyArrayObject *)y); \
    char *pa = PyArray_BYTES((PyArrayObject *)a); \
    const npy_intp *ystrides = PyArray_STRIDES((PyArrayObject *)y); \
    const npy_intp *astrides = PyArray_STRIDES((PyArrayObject *)a); \
    const npy_intp ystride = ystrides[axis]; \
    npy_intp index = 0; \
    npy_intp size = PyArray_SIZE((PyArrayObject *)a); \
    npy_intp indices[ndim]; \
    memset(indices, 0, ndim * sizeof(npy_intp)); \
    if (length != 0) size /= length;

#define NEXT \
    for (i=ndim - 1; i >= 0; i--) { \
        if ((indices[i] < shape[i] - 1) && (i != axis)) { \
            pa += astrides[i]; \
            py += ystrides[i]; \
            indices[i]++; \
            break; \
        } \
        pa -= indices[i] * astrides[i]; \
        py -= indices[i] * ystrides[i]; \
        indices[i] = 0; \
    } \
    index++;

#define RETURN \
    BN_END_ALLOW_THREADS \
    return y;

#define NOT_DONE index < size
#define AI(dt) *(dt*)(pa + i * stride)
#define YI(dt) *(dt*)(py + i++ * ystride)

/* function pointers ----------------------------------------------------- */

typedef PyObject *(*move_t)(PyObject *, int, int, int, Py_ssize_t,
                            Py_ssize_t, int, npy_intp*);
typedef PyObject *(*move_ddof_t)(PyObject *, int, int, int,
                                 Py_ssize_t, Py_ssize_t, int, npy_intp*, int);

/* prototypes ------------------------------------------------------------ */

static PyObject *
mover(char *name,
      PyObject *args,
      PyObject *kwds,
      move_t,
      move_t,
      move_t,
      move_t);

static PyObject *
mover_ddof(char *name,
           PyObject *args,
           PyObject *kwds,
           move_ddof_t,
           move_ddof_t,
           move_ddof_t,
           move_ddof_t);

/* move_sum -------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
move_sum_DTYPE0(PyObject *a, int window, int min_count, int axis,
                Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape)
{
    INIT(NPY_DTYPE0)
    Py_ssize_t count;
    npy_DTYPE0 asum, ai, aold, yi;
    while (NOT_DONE) {
        asum = 0;
        count = 0;
        i = 0;
        while (i < min_count - 1) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            YI(npy_DTYPE0) = BN_NAN;
        }
        while (i < window) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            if (count >= min_count) {
                yi = asum;
            }
            else {
                yi = BN_NAN;
            }
            YI(npy_DTYPE0) = yi;
        }
        while (i < length) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            aold = *(npy_DTYPE0*)(pa + (i-window)*stride);
            if (aold == aold) {
                asum -= aold;
                count -= 1;
            }
            if (count >= min_count) {
                yi = asum;
            }
            else {
                yi = BN_NAN;
            }
            YI(npy_DTYPE0) = yi;
        }
        NEXT
    }
    RETURN
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
move_sum_DTYPE0(PyObject *a, int window, int min_count, int axis,
                Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape)
{
    INIT(NPY_DTYPE1)
    npy_DTYPE1 asum;
    while (NOT_DONE) {
        asum = 0;
        i = 0;
        while (i < min_count - 1) {
            asum += AI(npy_DTYPE0);
            YI(npy_DTYPE1) = BN_NAN;
        }
        while (i < window) {
            asum += AI(npy_DTYPE0);
            YI(npy_DTYPE1) = asum;
        }
        while (i < length) {
            asum += AI(npy_DTYPE0);
            asum -= *(npy_DTYPE0*)(pa + (i-window)*stride);
            YI(npy_DTYPE1) = asum;
        }
        NEXT
    }
    RETURN
}
/* dtype end */


static PyObject *
move_sum(PyObject *self, PyObject *args, PyObject *kwds)
{
    return mover("move_sum",
                 args,
                 kwds,
                 move_sum_float64,
                 move_sum_float32,
                 move_sum_int64,
                 move_sum_int32);
}

/* move_mean -------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
move_mean_DTYPE0(PyObject *a, int window, int min_count, int axis,
                Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape)
{
    INIT(NPY_DTYPE0)
    Py_ssize_t count;
    npy_DTYPE0 asum, ai, aold, yi;
    while (NOT_DONE) {
        asum = 0;
        count = 0;
        i = 0;
        while (i < min_count - 1) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            YI(npy_DTYPE0) = BN_NAN;
        }
        while (i < window) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            if (count >= min_count) {
                yi = asum / count;
            }
            else {
                yi = BN_NAN;
            }
            YI(npy_DTYPE0) = yi;
        }
        while (i < length) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            aold = *(npy_DTYPE0*)(pa + (i-window)*stride);
            if (aold == aold) {
                asum -= aold;
                count -= 1;
            }
            if (count >= min_count) {
                yi = asum / count;
            }
            else {
                yi = BN_NAN;
            }
            YI(npy_DTYPE0) = yi;
        }
        NEXT
    }
    RETURN
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
move_mean_DTYPE0(PyObject *a, int window, int min_count, int axis,
                Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape)
{
    INIT(NPY_DTYPE1)
    npy_DTYPE1 asum;
    while (NOT_DONE) {
        asum = 0;
        i = 0;
        while (i < min_count - 1) {
            asum += AI(npy_DTYPE0);
            YI(npy_DTYPE1) = BN_NAN;
        }
        while (i < window) {
            asum += AI(npy_DTYPE0);
            *(npy_DTYPE1*)(py + i * ystride) = (npy_DTYPE1)asum / (i + 1);
            i++;
        }
        while (i < length) {
            asum += AI(npy_DTYPE0);
            asum -= *(npy_DTYPE0*)(pa + (i-window)*stride);
            YI(npy_DTYPE1) = (npy_DTYPE1)asum / window;
        }
        NEXT
    }
    RETURN
}
/* dtype end */


static PyObject *
move_mean(PyObject *self, PyObject *args, PyObject *kwds)
{
    return mover("move_mean",
                 args,
                 kwds,
                 move_mean_float64,
                 move_mean_float32,
                 move_mean_int64,
                 move_mean_int32);
}

/* move_std -------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
move_std_DTYPE0(PyObject *a, int window, int min_count, int axis,
                Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape, int ddof)
{
    INIT(NPY_DTYPE0)
    Py_ssize_t count;
    npy_DTYPE0 delta, amean, assqdm, ai, aold, yi;
    while (NOT_DONE) {
        amean = 0;
        assqdm = 0;
        count = 0;
        i = 0;
        while (i < min_count - 1) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                count += 1;
                delta = ai - amean;
                amean += delta / count;
                assqdm += delta * (ai - amean);
            }
            YI(npy_DTYPE0) = BN_NAN;
        }
        while (i <  window) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                count += 1;
                delta = ai - amean;
                amean += delta / count;
                assqdm += delta * (ai - amean);
            }
            if (count >= min_count) {
                if (assqdm < 0) {
                    assqdm = 0;
                }
                yi = sqrt(assqdm / (count - ddof));
            }
            else {
                yi = BN_NAN;
            }
            YI(npy_DTYPE0) = yi;
        }
        while (i < length) {
            ai = AI(npy_DTYPE0);
            aold = *(npy_DTYPE0*)(pa + (i-window)*stride);
            if (ai == ai) {
                if (aold == aold) {
                    delta = ai - aold;
                    aold -= amean;
                    amean += delta / count;
                    ai -= amean;
                    assqdm += (ai + aold) * delta;
                }
                else {
                    count += 1;
                    delta = ai - amean;
                    amean += delta / count;
                    assqdm += delta * (ai - amean);
                }
            }
            else {
                if (aold == aold) {
                    count -= 1;
                    if (count > 0) {
                        delta = aold - amean;
                        amean -= delta / count;
                        assqdm -= delta * (aold - amean);
                    }
                    else {
                        amean = 0;
                        assqdm = 0;
                    }
                }
            }
            if (count >= min_count) {
                if (assqdm < 0) {
                    assqdm = 0;
                }
                yi = sqrt(assqdm / (count - ddof));
            }
            else {
                yi = BN_NAN;
            }
            YI(npy_DTYPE0) = yi;
        }
        NEXT
    }
    RETURN
}
/* dtype end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
move_std_DTYPE0(PyObject *a, int window, int min_count, int axis,
                Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape, int ddof)
{
    INIT(NPY_DTYPE1)
    int winddof = window - ddof;
    npy_DTYPE1 delta, amean, assqdm, yi, ai, aold;
    while (NOT_DONE) {
        amean = 0;
        assqdm = 0;
        i = 0;
        while (i < min_count - 1) {
            ai = AI(npy_DTYPE0);
            delta = ai - amean;
            amean += delta / (i + 1);
            assqdm += delta * (ai - amean);
            YI(npy_DTYPE1) = BN_NAN;
        }
        while (i <  window) {
            ai = AI(npy_DTYPE0);
            delta = ai - amean;
            amean += delta / (i + 1);
            assqdm += delta * (ai - amean);
            yi = sqrt(assqdm / (i + 1 - ddof));
            YI(npy_DTYPE1) = yi;
        }
        while (i < length) {
            ai = AI(npy_DTYPE0);
            aold = *(npy_DTYPE0*)(pa + (i-window)*stride);
            delta = ai - aold;
            aold -= amean;
            amean += delta / window;
            ai -= amean;
            assqdm += (ai + aold) * delta;
            if (assqdm < 0) {
                assqdm = 0;
            }
            yi = sqrt(assqdm / winddof);
            YI(npy_DTYPE1) = yi;
        }
        NEXT
    }
    RETURN
}
/* dtype end */

static PyObject *
move_std(PyObject *self, PyObject *args, PyObject *kwds)
{
    return mover_ddof("move_std",
                      args,
                      kwds,
                      move_std_float64,
                      move_std_float32,
                      move_std_int64,
                      move_std_int32);
}

/* move_var -------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
static PyObject *
move_var_DTYPE0(PyObject *a, int window, int min_count, int axis,
                Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape, int ddof)
{
    INIT(NPY_DTYPE0)
    Py_ssize_t count;
    npy_DTYPE0 delta, amean, assqdm, ai, aold, yi;
    while (NOT_DONE) {
        amean = 0;
        assqdm = 0;
        count = 0;
        i = 0;
        while (i < min_count - 1) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                count += 1;
                delta = ai - amean;
                amean += delta / count;
                assqdm += delta * (ai - amean);
            }
            YI(npy_DTYPE0) = BN_NAN;
        }
        while (i <  window) {
            ai = AI(npy_DTYPE0);
            if (ai == ai) {
                count += 1;
                delta = ai - amean;
                amean += delta / count;
                assqdm += delta * (ai - amean);
            }
            if (count >= min_count) {
                if (assqdm < 0) {
                    assqdm = 0;
                }
                yi = assqdm / (count - ddof);
            }
            else {
                yi = BN_NAN;
            }
            YI(npy_DTYPE0) = yi;
        }
        while (i < length) {
            ai = AI(npy_DTYPE0);
            aold = *(npy_DTYPE0*)(pa + (i-window)*stride);
            if (ai == ai) {
                if (aold == aold) {
                    delta = ai - aold;
                    aold -= amean;
                    amean += delta / count;
                    ai -= amean;
                    assqdm += (ai + aold) * delta;
                }
                else {
                    count += 1;
                    delta = ai - amean;
                    amean += delta / count;
                    assqdm += delta * (ai - amean);
                }
            }
            else {
                if (aold == aold) {
                    count -= 1;
                    if (count > 0) {
                        delta = aold - amean;
                        amean -= delta / count;
                        assqdm -= delta * (aold - amean);
                    }
                    else {
                        amean = 0;
                        assqdm = 0;
                    }
                }
            }
            if (count >= min_count) {
                if (assqdm < 0) {
                    assqdm = 0;
                }
                yi = assqdm / (count - ddof);
            }
            else {
                yi = BN_NAN;
            }
            YI(npy_DTYPE0) = yi;
        }
        NEXT
    }
    RETURN
}
/* dtype end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
static PyObject *
move_var_DTYPE0(PyObject *a, int window, int min_count, int axis,
                Py_ssize_t stride, Py_ssize_t length,
                int ndim, npy_intp* shape, int ddof)
{
    INIT(NPY_DTYPE1)
    int winddof = window - ddof;
    npy_DTYPE1 delta, amean, assqdm, yi, ai, aold;
    while (NOT_DONE) {
        amean = 0;
        assqdm = 0;
        i = 0;
        while (i < min_count - 1) {
            ai = AI(npy_DTYPE0);
            delta = ai - amean;
            amean += delta / (i + 1);
            assqdm += delta * (ai - amean);
            YI(npy_DTYPE1) = BN_NAN;
        }
        while (i <  window) {
            ai = AI(npy_DTYPE0);
            delta = ai - amean;
            amean += delta / (i + 1);
            assqdm += delta * (ai - amean);
            yi = assqdm / (i + 1 - ddof);
            YI(npy_DTYPE1) = yi;
        }
        while (i < length) {
            ai = AI(npy_DTYPE0);
            aold = *(npy_DTYPE0*)(pa + (i-window)*stride);
            delta = ai - aold;
            aold -= amean;
            amean += delta / window;
            ai -= amean;
            assqdm += (ai + aold) * delta;
            if (assqdm < 0) {
                assqdm = 0;
            }
            yi = assqdm / winddof;
            YI(npy_DTYPE1) = yi;
        }
        NEXT
    }
    RETURN
}
/* dtype end */

static PyObject *
move_var(PyObject *self, PyObject *args, PyObject *kwds)
{
    return mover_ddof("move_var",
                      args,
                      kwds,
                      move_var_float64,
                      move_var_float32,
                      move_var_int64,
                      move_var_int32);
}

/* python strings -------------------------------------------------------- */

PyObject *pystr_arr = NULL;
PyObject *pystr_window = NULL;
PyObject *pystr_min_count = NULL;
PyObject *pystr_axis = NULL;
PyObject *pystr_ddof = NULL;

static int
intern_strings(void) {
    pystr_arr = PyString_InternFromString("arr");
    pystr_window = PyString_InternFromString("window");
    pystr_min_count = PyString_InternFromString("min_count");
    pystr_axis = PyString_InternFromString("axis");
    pystr_ddof = PyString_InternFromString("ddof");
    return pystr_arr && pystr_window && pystr_min_count &&
           pystr_axis && pystr_ddof;
}

/* mover ----------------------------------------------------------------- */

static BN_INLINE int
parse_args(PyObject *args,
           PyObject *kwds,
           PyObject **arr,
           PyObject **window,
           PyObject **min_count,
           PyObject **axis)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (kwds) {
        int nkwds_found = 0;
        const Py_ssize_t nkwds = PyDict_Size(kwds);
        PyObject *tmp;
        switch (nargs) {
            case 3: *min_count = PyTuple_GET_ITEM(args, 2);
            case 2: *window = PyTuple_GET_ITEM(args, 1);
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
                *window = PyDict_GetItem(kwds, pystr_window);
                if (*window == NULL) {
                    TYPE_ERR("Cannot find `window` keyword input");
                    return 0;
                }
                nkwds_found++;
            case 2:
                tmp = PyDict_GetItem(kwds, pystr_min_count);
                if (tmp != NULL) {
                    *min_count = tmp;
                    nkwds_found++;
                }
            case 3:
                tmp = PyDict_GetItem(kwds, pystr_axis);
                if (tmp != NULL) {
                    *axis = tmp;
                    nkwds_found++;
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
        if (nargs + nkwds_found > 4) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    }
    else {
        switch (nargs) {
            case 4:
                *axis = PyTuple_GET_ITEM(args, 3);
            case 3:
                *min_count = PyTuple_GET_ITEM(args, 2);
            case 2:
                *window = PyTuple_GET_ITEM(args, 1);
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
mover(char *name,
      PyObject *args,
      PyObject *kwds,
      move_t move_float64,
      move_t move_float32,
      move_t move_int64,
      move_t move_int32)
{

    int mc;
    int window;
    int axis;
    int dtype;
    int ndim;

    Py_ssize_t stride;
    Py_ssize_t length;

    PyArrayObject *a;

    PyObject *y;
    npy_intp *shape;

    PyObject *arr_obj = NULL;
    PyObject *window_obj = NULL;
    PyObject *min_count_obj = Py_None;
    PyObject *axis_obj = NULL;

    if (!parse_args(args, kwds,
                    &arr_obj, &window_obj, &min_count_obj, &axis_obj)) {
        return NULL;
    }

    /* convert to array if necessary */
    if PyArray_Check(arr_obj) {
        a = (PyArrayObject *)arr_obj;
    }
    else {
        a = (PyArrayObject *)PyArray_FROM_O(arr_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        return slow(name, args, kwds);
    }

    /* window */
    window = PyArray_PyIntAsInt(window_obj);
    if (error_converting(window)) {
        TYPE_ERR("`window` must be an integer");
        return NULL;
    }

    /* min_count */
    if (min_count_obj == Py_None) {
        mc = window;
    }
    else {
        mc = PyArray_PyIntAsInt(min_count_obj);
        if (error_converting(mc)) {
            TYPE_ERR("`min_count` must be an integer or None");
            return NULL;
        }
        if (mc > window) {
            PyErr_Format(PyExc_ValueError,
                         "min_count (%d) cannot be greater than window (%d)",
                         mc, window);
            return NULL;
        }
        else if (mc <= 0) {
            VALUE_ERR("`min_count` must be greater than zero.");
            return NULL;
        }
    }

    /* input array */
    dtype = PyArray_TYPE(a);
    ndim = PyArray_NDIM(a);

    /* defend against 0d beings */
    if (ndim == 0) {
        VALUE_ERR("moving window functions require ndim > 0");
        return NULL;
    }

    /* defend against the axis of negativity */
    if (axis_obj == NULL) {
        axis = ndim - 1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer");
            return NULL;
        }
        if (axis < 0) {
            axis += ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
                return NULL;
            }
        }
        else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            return NULL;
        }
    }

    stride = PyArray_STRIDE(a, axis);
    shape = PyArray_SHAPE(a);
    length = shape[axis];

    if ((window < 1) || (window > length)) {
        PyErr_Format(PyExc_ValueError,
                     "Moving window (=%d) must between 1 and %zu, inclusive",
                     window, length);
        return NULL;
    }

    if (dtype == NPY_float64) {
        y = move_float64((PyObject *)a, window, mc, axis, stride, length,
                         ndim, shape);
    }
    else if (dtype == NPY_float32) {
        y = move_float32((PyObject *)a, window, mc, axis, stride, length,
                         ndim, shape);
    }
    else if (dtype == NPY_int64) {
        y = move_int64((PyObject *)a, window, mc, axis, stride, length,
                       ndim, shape);
    }
    else if (dtype == NPY_int32) {
        y = move_int32((PyObject *)a, window, mc, axis, stride, length,
                       ndim, shape);
    }
    else {
        y = slow(name, args, kwds);
    }

    return y;
}

/* mover_ddof ------------------------------------------------------------ */

static BN_INLINE int
parse_args_ddof(PyObject *args,
                PyObject *kwds,
                PyObject **arr,
                PyObject **window,
                PyObject **min_count,
                PyObject **axis,
                PyObject **ddof)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (kwds) {
        int nkwds_found = 0;
        const Py_ssize_t nkwds = PyDict_Size(kwds);
        PyObject *tmp;
        switch (nargs) {
            case 4: *ddof = PyTuple_GET_ITEM(args, 3);
            case 3: *min_count = PyTuple_GET_ITEM(args, 2);
            case 2: *window = PyTuple_GET_ITEM(args, 1);
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
                *window = PyDict_GetItem(kwds, pystr_window);
                if (*window == NULL) {
                    TYPE_ERR("Cannot find `window` keyword input");
                    return 0;
                }
                nkwds_found++;
            case 2:
                tmp = PyDict_GetItem(kwds, pystr_min_count);
                if (tmp != NULL) {
                    *min_count = tmp;
                    nkwds_found++;
                }
            case 3:
                tmp = PyDict_GetItem(kwds, pystr_axis);
                if (tmp != NULL) {
                    *axis = tmp;
                    nkwds_found++;
                }
            case 4:
                tmp = PyDict_GetItem(kwds, pystr_ddof);
                if (tmp != NULL) {
                    *ddof = tmp;
                    nkwds_found++;
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
        if (nargs + nkwds_found > 5) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    }
    else {
        switch (nargs) {
            case 5:
                *ddof = PyTuple_GET_ITEM(args, 4);
            case 4:
                *axis = PyTuple_GET_ITEM(args, 3);
            case 3:
                *min_count = PyTuple_GET_ITEM(args, 2);
            case 2:
                *window = PyTuple_GET_ITEM(args, 1);
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
mover_ddof(char *name,
           PyObject *args,
           PyObject *kwds,
           move_ddof_t move_float64,
           move_ddof_t move_float32,
           move_ddof_t move_int64,
           move_ddof_t move_int32)
{

    int mc;
    int window;
    int axis;
    int ddof;
    int dtype;
    int ndim;

    Py_ssize_t stride;
    Py_ssize_t length;

    PyArrayObject *a;

    PyObject *y;
    npy_intp *shape;

    PyObject *arr_obj = NULL;
    PyObject *window_obj = NULL;
    PyObject *min_count_obj = Py_None;
    PyObject *axis_obj = NULL;
    PyObject *ddof_obj = NULL;

    if (!parse_args_ddof(args, kwds, &arr_obj, &window_obj, &min_count_obj,
                         &axis_obj, &ddof_obj)) {
        return NULL;
    }

    /* convert to array if necessary */
    if PyArray_Check(arr_obj) {
        a = (PyArrayObject *)arr_obj;
    }
    else {
        a = (PyArrayObject *)PyArray_FROM_O(arr_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        return slow(name, args, kwds);
    }

    /* window */
    window = PyArray_PyIntAsInt(window_obj);
    if (error_converting(window)) {
        TYPE_ERR("`window` must be an integer");
        return NULL;
    }

    /* min_count */
    if (min_count_obj == Py_None) {
        mc = window;
    }
    else {
        mc = PyArray_PyIntAsInt(min_count_obj);
        if (error_converting(mc)) {
            TYPE_ERR("`min_count` must be an integer or None");
            return NULL;
        }
        if (mc > window) {
            PyErr_Format(PyExc_ValueError,
                         "min_count (%d) cannot be greater than window (%d)",
                         mc, window);
            return NULL;
        }
        else if (mc <= 0) {
            VALUE_ERR("`min_count` must be greater than zero.");
            return NULL;
        }
    }

    /* input array */
    dtype = PyArray_TYPE(a);
    ndim = PyArray_NDIM(a);

    /* defend against 0d beings */
    if (ndim == 0) {
        VALUE_ERR("moving window functions require ndim > 0");
        return NULL;
    }

    /* defend against the axis of negativity */
    if (axis_obj == NULL) {
        axis = ndim - 1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer");
            return NULL;
        }
        if (axis < 0) {
            axis += ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
                return NULL;
            }
        }
        else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            return NULL;
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

    stride = PyArray_STRIDE(a, axis);
    shape = PyArray_SHAPE(a);
    length = shape[axis];

    if ((window < 1) || (window > length)) {
        PyErr_Format(PyExc_ValueError,
                     "Moving window (=%d) must between 1 and %zu, inclusive",
                     window, length);
        return NULL;
    }

    if (dtype == NPY_float64) {
        y = move_float64((PyObject *)a, window, mc, axis, stride, length,
                         ndim, shape, ddof);
    }
    else if (dtype == NPY_float32) {
        y = move_float32((PyObject *)a, window, mc, axis, stride, length,
                         ndim, shape, ddof);
    }
    else if (dtype == NPY_int64) {
        y = move_int64((PyObject *)a, window, mc, axis, stride, length,
                       ndim, shape, ddof);
    }
    else if (dtype == NPY_int32) {
        y = move_int32((PyObject *)a, window, mc, axis, stride, length,
                       ndim, shape, ddof);
    }
    else {
        y = slow(name, args, kwds);
    }

    return y;
}

/* docstrings ------------------------------------------------------------- */

static char move_doc[] =
"Bottleneck moving window functions.";

static char move_sum_doc[] =
/* MULTILINE STRING BEGIN
move_sum(arr, window, min_count=None, axis=-1)

Moving window sum along the specified axis, optionally ignoring NaNs.

This function cannot handle input arrays that contain Inf. When the
window contains Inf, the output will correctly be Inf. However, when Inf
moves out of the window, the remaining output values in the slice will
incorrectly be NaN.

Parameters
----------
arr : ndarray
    Input array. If `arr` is not an array, a conversion is attempted.
window : int
    The number of elements in the moving window.
min_count: {int, None}, optional
    If the number of non-NaN values in a window is less than `min_count`,
    then a value of NaN is assigned to the window. By default `min_count`
    is None, which is equivalent to setting `min_count` equal to `window`.
axis : int, optional
    The axis over which the window is moved. By default the last axis
    (axis=-1) is used. An axis of None is not allowed.

Returns
-------
y : ndarray
    The moving sum of the input array along the specified axis. The output
    has the same shape as the input.

Examples
--------
>>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_sum(arr, window=2)
array([ nan,   3.,   5.,  nan,  nan])
>>> bn.move_sum(arr, window=2, min_count=1)
array([ 1.,  3.,  5.,  3.,  5.])

MULTILINE STRING END */

static char move_mean_doc[] =
/* MULTILINE STRING BEGIN
move_mean(arr, window, min_count=None, axis=-1)

Moving window mean along the specified axis, optionally ignoring NaNs.

This function cannot handle input arrays that contain Inf. When the
window contains Inf, the output will correctly be Inf. However, when Inf
moves out of the window, the remaining output values in the slice will
incorrectly be NaN.

Parameters
----------
arr : ndarray
    Input array. If `arr` is not an array, a conversion is attempted.
window : int
    The number of elements in the moving window.
min_count: {int, None}, optional
    If the number of non-NaN values in a window is less than `min_count`,
    then a value of NaN is assigned to the window. By default `min_count`
    is None, which is equivalent to setting `min_count` equal to `window`.
axis : int, optional
    The axis over which the window is moved. By default the last axis
    (axis=-1) is used. An axis of None is not allowed.

Returns
-------
y : ndarray
    The moving mean of the input array along the specified axis. The output
    has the same shape as the input.

Examples
--------
>>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_mean(arr, window=2)
array([ nan,  1.5,  2.5,  nan,  nan])
>>> bn.move_mean(arr, window=2, min_count=1)
array([ 1. ,  1.5,  2.5,  3. ,  5. ])

MULTILINE STRING END */

static char move_std_doc[] =
/* MULTILINE STRING BEGIN
move_std(arr, window, min_count=None, axis=-1, ddof=0)

Moving window standard deviation along the specified axis, optionally
ignoring NaNs.

This function cannot handle input arrays that contain Inf. When Inf
enters the moving window, the outout becomes NaN and will continue to
be NaN for the remainer of the slice.

Unlike bn.nanstd, which uses a two-pass algorithm, move_nanstd uses a
one-pass algorithm called Welford's method. The algorithm is slow but
numerically stable for cases where the mean is large compared to the
standard deviation.

Parameters
----------
arr : ndarray
    Input array. If `arr` is not an array, a conversion is attempted.
window : int
    The number of elements in the moving window.
min_count: {int, None}, optional
    If the number of non-NaN values in a window is less than `min_count`,
    then a value of NaN is assigned to the window. By default `min_count`
    is None, which is equivalent to setting `min_count` equal to `window`.
axis : int, optional
    The axis over which the window is moved. By default the last axis
    (axis=-1) is used. An axis of None is not allowed.
ddof : int, optional
    Means Delta Degrees of Freedom. The divisor used in calculations
    is ``N - ddof``, where ``N`` represents the number of elements.
    By default `ddof` is zero.

Returns
-------
y : ndarray
    The moving standard deviation of the input array along the specified
    axis. The output has the same shape as the input.

Examples
--------
>>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_std(arr, window=2)
array([ nan,  0.5,  0.5,  nan,  nan])
>>> bn.move_std(arr, window=2, min_count=1)
array([ 0. ,  0.5,  0.5,  0. ,  0. ])

MULTILINE STRING END */

static char move_var_doc[] =
/* MULTILINE STRING BEGIN
move_var(arr, window, min_count=None, axis=-1, ddof=0)

Moving window variance along the specified axis, optionally ignoring NaNs.

This function cannot handle input arrays that contain Inf. When Inf
enters the moving window, the outout becomes NaN and will continue to
be NaN for the remainer of the slice.

Unlike bn.nanvar, which uses a two-pass algorithm, move_nanvar uses a
one-pass algorithm called Welford's method. The algorithm is slow but
numerically stable for cases where the mean is large compared to the
standard deviation.

Parameters
----------
arr : ndarray
    Input array. If `arr` is not an array, a conversion is attempted.
window : int
    The number of elements in the moving window.
min_count: {int, None}, optional
    If the number of non-NaN values in a window is less than `min_count`,
    then a value of NaN is assigned to the window. By default `min_count`
    is None, which is equivalent to setting `min_count` equal to `window`.
axis : int, optional
    The axis over which the window is moved. By default the last axis
    (axis=-1) is used. An axis of None is not allowed.
ddof : int, optional
    Means Delta Degrees of Freedom. The divisor used in calculations
    is ``N - ddof``, where ``N`` represents the number of elements.
    By default `ddof` is zero.

Returns
-------
y : ndarray
    The moving variance of the input array along the specified axis. The
    output has the same shape as the input.

Examples
--------
>>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_var(arr, window=2)
array([ nan,  0.25,  0.25,  nan,  nan])
>>> bn.move_var(arr, window=2, min_count=1)
array([ 0. ,  0.25,  0.25,  0. ,  0. ])

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
move_methods[] = {
    {"move_sum", (PyCFunction)move_sum, VARKEY, move_sum_doc},
    {"move_mean", (PyCFunction)move_mean, VARKEY, move_mean_doc},
    {"move_std", (PyCFunction)move_std, VARKEY, move_std_doc},
    {"move_var", (PyCFunction)move_var, VARKEY, move_var_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
move_def = {
   PyModuleDef_HEAD_INIT,
   "move2",
   move_doc,
   -1,
   move_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_move2(void)
#else
#define RETVAL
initmove2(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&move_def);
    #else
        PyObject *m = Py_InitModule3("move2", move_methods, move_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
