#include "bottleneck.h"
#include "iterators.h"

/* init macros ----------------------------------------------------------- */

#define INIT_ALL \
    iter it; \
    init_iter_all(&it, a, 0, 1);

#define INIT_ALL_RAVEL \
    iter it; \
    init_iter_all(&it, a, 1, 0);

/* used with INIT_ALL_RAVEL */
#define DECREF_INIT_ALL_RAVEL \
    if (it.a_ravel != NULL) { \
        Py_DECREF(it.a_ravel); \
    }

#define INIT_ONE(dtype0, dtype1) \
    iter it; \
    PyObject *y; \
    npy_##dtype1 *py; \
    init_iter_one(&it, a, axis); \
    y = PyArray_EMPTY(NDIM - 1, SHAPE, NPY_##dtype0, 0); \
    py = (npy_##dtype1 *)PyArray_DATA((PyArrayObject *)y);

/* function signatures --------------------------------------------------- */

/* low-level functions such as nansum_all_float64 */
#define REDUCE_ALL(name, dtype) \
    static PyObject * \
    name##_all_##dtype(PyArrayObject *a, int ddof)

/* low-level functions such as nansum_one_float64 */
#define REDUCE_ONE(name, dtype) \
    static PyObject * \
    name##_one_##dtype(PyArrayObject *a, int axis, int ddof)

/* top-level functions such as nansum */
#define REDUCE_MAIN(name, has_ddof) \
    static PyObject * \
    name(PyObject *self, PyObject *args, PyObject *kwds) \
    { \
        return reducer(#name, \
                       args, \
                       kwds, \
                       name##_all_float64, \
                       name##_all_float32, \
                       name##_all_int64, \
                       name##_all_int32, \
                       name##_one_float64, \
                       name##_one_float32, \
                       name##_one_int64, \
                       name##_one_int32, \
                       has_ddof); \
    }

/* typedefs and prototypes ----------------------------------------------- */

typedef PyObject *(*fall_t)(PyArrayObject *a, int ddof);
typedef PyObject *(*fone_t)(PyArrayObject *a, int axis, int ddof);

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
        int has_ddof);

/* nansum ---------------------------------------------------------------- */

REDUCE_ALL(nansum, float64)
{
    npy_float64 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
            if (ai == ai) asum += ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}

REDUCE_ONE(nansum, float64)
{
    npy_float64 ai, asum;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR {
                ai = AI(float64);
                if (ai == ai) asum += ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nansum, float32)
{
    npy_float32 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
            if (ai == ai) asum += ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}

REDUCE_ONE(nansum, float32)
{
    npy_float32 ai, asum;
    INIT_ONE(float32, float32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR {
                ai = AI(float32);
                if (ai == ai) asum += ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nansum, int64)
{
    npy_int64 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(int64);
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}

REDUCE_ONE(nansum, int64)
{
    npy_int64 asum;
    INIT_ONE(int64, int64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(int64);
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nansum, int32)
{
    npy_int32 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(int32);
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}

REDUCE_ONE(nansum, int32)
{
    npy_int32 asum;
    INIT_ONE(int32, int32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(int32);
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(nansum, 0)

/* nanmean ---------------------------------------------------------------- */

REDUCE_ALL(nanmean, float64)
{
    Py_ssize_t count = 0;
    npy_float64 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
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

REDUCE_ONE(nanmean, float64)
{
    Py_ssize_t count;
    npy_float64 ai, asum;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(float64);
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
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmean, float32)
{
    Py_ssize_t count = 0;
    npy_float32 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
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

REDUCE_ONE(nanmean, float32)
{
    Py_ssize_t count;
    npy_float32 ai, asum;
    INIT_ONE(float32, float32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(float32);
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
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmean, int64)
{
    Py_ssize_t total_length = 0;
    npy_float64 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(int64);
        total_length += LENGTH;
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (total_length > 0) {
        return PyFloat_FromDouble(asum / total_length);
    } else {
        return PyFloat_FromDouble(BN_NAN);
    }
}

REDUCE_ONE(nanmean, int64)
{
    npy_float64 asum;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(int64);
            if (LENGTH > 0) {
                asum /= LENGTH;
            } else {
                asum = BN_NAN;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmean, int32)
{
    Py_ssize_t total_length = 0;
    npy_float64 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(int32);
        total_length += LENGTH;
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (total_length > 0) {
        return PyFloat_FromDouble(asum / total_length);
    } else {
        return PyFloat_FromDouble(BN_NAN);
    }
}

REDUCE_ONE(nanmean, int32)
{
    npy_float64 asum;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(int32);
            if (LENGTH > 0) {
                asum /= LENGTH;
            } else {
                asum = BN_NAN;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(nanmean, 0)

/* nanstd, nanvar- ------------------------------------------------------- */

REDUCE_ALL(nanstd, float64)
{
    Py_ssize_t count = 0;
    npy_float64 ai, amean, out, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
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
                ai = AI(float64);
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

REDUCE_ONE(nanstd, float64)
{
    Py_ssize_t count;
    npy_float64 ai, asum, amean;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(float64);
                if (ai == ai) {
                    asum += ai;
                    count++;
                }
            }
            if (count > ddof) {
                amean = asum / count;
                asum = 0;
                FOR {
                    ai = AI(float64);
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
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanstd, float32)
{
    Py_ssize_t count = 0;
    npy_float32 ai, amean, out, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
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
                ai = AI(float32);
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

REDUCE_ONE(nanstd, float32)
{
    Py_ssize_t count;
    npy_float32 ai, asum, amean;
    INIT_ONE(float32, float32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(float32);
                if (ai == ai) {
                    asum += ai;
                    count++;
                }
            }
            if (count > ddof) {
                amean = asum / count;
                asum = 0;
                FOR {
                    ai = AI(float32);
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
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanstd, int64)
{
    npy_float64 out;
    Py_ssize_t size = 0;
    npy_float64 ai, amean, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(int64);
        size += LENGTH;
        NEXT
    }
    if (size > ddof) {
        amean = asum / size;
        asum = 0;
        RESET
        WHILE {
            FOR {
                ai = AI(int64) - amean;
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

REDUCE_ONE(nanstd, int64)
{
    npy_float64 ai, asum, amean, length_inv, length_ddof_inv;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    length_inv = 1.0 / LENGTH;
    length_ddof_inv = 1.0 / (LENGTH - ddof);
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(int64);
            if (LENGTH > ddof) {
                amean = asum * length_inv;
                asum = 0;
                FOR {
                    ai = AI(int64) - amean;
                    asum += ai * ai;
                }
                asum = sqrt(asum * length_ddof_inv);
            }
            else {
                asum = BN_NAN;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanstd, int32)
{
    npy_float64 out;
    Py_ssize_t size = 0;
    npy_float64 ai, amean, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(int32);
        size += LENGTH;
        NEXT
    }
    if (size > ddof) {
        amean = asum / size;
        asum = 0;
        RESET
        WHILE {
            FOR {
                ai = AI(int32) - amean;
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

REDUCE_ONE(nanstd, int32)
{
    npy_float64 ai, asum, amean, length_inv, length_ddof_inv;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    length_inv = 1.0 / LENGTH;
    length_ddof_inv = 1.0 / (LENGTH - ddof);
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(int32);
            if (LENGTH > ddof) {
                amean = asum * length_inv;
                asum = 0;
                FOR {
                    ai = AI(int32) - amean;
                    asum += ai * ai;
                }
                asum = sqrt(asum * length_ddof_inv);
            }
            else {
                asum = BN_NAN;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(nanstd, 1)

REDUCE_ALL(nanvar, float64)
{
    Py_ssize_t count = 0;
    npy_float64 ai, amean, out, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
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
                ai = AI(float64);
                if (ai == ai) {
                    ai -= amean;
                    asum += ai * ai;
                }
            }
            NEXT
        }
        out = (asum / (count - ddof));
    }
    else {
        out = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}

REDUCE_ONE(nanvar, float64)
{
    Py_ssize_t count;
    npy_float64 ai, asum, amean;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(float64);
                if (ai == ai) {
                    asum += ai;
                    count++;
                }
            }
            if (count > ddof) {
                amean = asum / count;
                asum = 0;
                FOR {
                    ai = AI(float64);
                    if (ai == ai) {
                        ai -= amean;
                        asum += ai * ai;
                    }
                }
                asum = (asum / (count - ddof));
            }
            else {
                asum = BN_NAN;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanvar, float32)
{
    Py_ssize_t count = 0;
    npy_float32 ai, amean, out, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
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
                ai = AI(float32);
                if (ai == ai) {
                    ai -= amean;
                    asum += ai * ai;
                }
            }
            NEXT
        }
        out = (asum / (count - ddof));
    }
    else {
        out = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}

REDUCE_ONE(nanvar, float32)
{
    Py_ssize_t count;
    npy_float32 ai, asum, amean;
    INIT_ONE(float32, float32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = count = 0;
            FOR {
                ai = AI(float32);
                if (ai == ai) {
                    asum += ai;
                    count++;
                }
            }
            if (count > ddof) {
                amean = asum / count;
                asum = 0;
                FOR {
                    ai = AI(float32);
                    if (ai == ai) {
                        ai -= amean;
                        asum += ai * ai;
                    }
                }
                asum = (asum / (count - ddof));
            }
            else {
                asum = BN_NAN;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanvar, int64)
{
    npy_float64 out;
    Py_ssize_t size = 0;
    npy_float64 ai, amean, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(int64);
        size += LENGTH;
        NEXT
    }
    if (size > ddof) {
        amean = asum / size;
        asum = 0;
        RESET
        WHILE {
            FOR {
                ai = AI(int64) - amean;
                asum += ai * ai;
            }
            NEXT
        }
        out = (asum / (size - ddof));
    }
    else {
        out = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}

REDUCE_ONE(nanvar, int64)
{
    npy_float64 ai, asum, amean, length_inv, length_ddof_inv;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    length_inv = 1.0 / LENGTH;
    length_ddof_inv = 1.0 / (LENGTH - ddof);
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(int64);
            if (LENGTH > ddof) {
                amean = asum * length_inv;
                asum = 0;
                FOR {
                    ai = AI(int64) - amean;
                    asum += ai * ai;
                }
                asum = (asum * length_ddof_inv);
            }
            else {
                asum = BN_NAN;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanvar, int32)
{
    npy_float64 out;
    Py_ssize_t size = 0;
    npy_float64 ai, amean, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(int32);
        size += LENGTH;
        NEXT
    }
    if (size > ddof) {
        amean = asum / size;
        asum = 0;
        RESET
        WHILE {
            FOR {
                ai = AI(int32) - amean;
                asum += ai * ai;
            }
            NEXT
        }
        out = (asum / (size - ddof));
    }
    else {
        out = BN_NAN;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}

REDUCE_ONE(nanvar, int32)
{
    npy_float64 ai, asum, amean, length_inv, length_ddof_inv;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    length_inv = 1.0 / LENGTH;
    length_ddof_inv = 1.0 / (LENGTH - ddof);
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(int32);
            if (LENGTH > ddof) {
                amean = asum * length_inv;
                asum = 0;
                FOR {
                    ai = AI(int32) - amean;
                    asum += ai * ai;
                }
                asum = (asum * length_ddof_inv);
            }
            else {
                asum = BN_NAN;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(nanvar, 1)

/* nanmin, nanmax -------------------------------------------------------- */

REDUCE_ALL(nanmin, float64)
{
    npy_float64 ai, extreme = BN_INFINITY;
    int allnan = 1;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.nanmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
            if (ai <= extreme) {
                extreme = ai;
                allnan = 0;
            }
        }
        NEXT
    }
    if (allnan) extreme = BN_NAN;
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(extreme);
}

REDUCE_ONE(nanmin, float64)
{
    npy_float64 ai, extreme;
    int allnan;
    INIT_ONE(float64, float64)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = BN_INFINITY;
        allnan = 1;
        FOR {
            ai = AI(float64);
            if (ai <= extreme) {
                extreme = ai;
                allnan = 0;
            }
        }
        if (allnan) extreme = BN_NAN;
        YPP = extreme;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmin, float32)
{
    npy_float32 ai, extreme = BN_INFINITY;
    int allnan = 1;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.nanmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
            if (ai <= extreme) {
                extreme = ai;
                allnan = 0;
            }
        }
        NEXT
    }
    if (allnan) extreme = BN_NAN;
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(extreme);
}

REDUCE_ONE(nanmin, float32)
{
    npy_float32 ai, extreme;
    int allnan;
    INIT_ONE(float32, float32)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = BN_INFINITY;
        allnan = 1;
        FOR {
            ai = AI(float32);
            if (ai <= extreme) {
                extreme = ai;
                allnan = 0;
            }
        }
        if (allnan) extreme = BN_NAN;
        YPP = extreme;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmin, int64)
{
    npy_int64 ai, extreme = NPY_MAX_int64;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.nanmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(int64);
            if (ai <= extreme) extreme = ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(extreme);
}

REDUCE_ONE(nanmin, int64)
{
    npy_int64 ai, extreme;
    INIT_ONE(int64, int64)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = NPY_MAX_int64;
        FOR {
            ai = AI(int64);
            if (ai <= extreme) extreme = ai;
        }
        YPP = extreme;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmin, int32)
{
    npy_int32 ai, extreme = NPY_MAX_int32;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.nanmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(int32);
            if (ai <= extreme) extreme = ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(extreme);
}

REDUCE_ONE(nanmin, int32)
{
    npy_int32 ai, extreme;
    INIT_ONE(int32, int32)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = NPY_MAX_int32;
        FOR {
            ai = AI(int32);
            if (ai <= extreme) extreme = ai;
        }
        YPP = extreme;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(nanmin, 0)

REDUCE_ALL(nanmax, float64)
{
    npy_float64 ai, extreme = -BN_INFINITY;
    int allnan = 1;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.nanmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
            if (ai >= extreme) {
                extreme = ai;
                allnan = 0;
            }
        }
        NEXT
    }
    if (allnan) extreme = BN_NAN;
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(extreme);
}

REDUCE_ONE(nanmax, float64)
{
    npy_float64 ai, extreme;
    int allnan;
    INIT_ONE(float64, float64)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = -BN_INFINITY;
        allnan = 1;
        FOR {
            ai = AI(float64);
            if (ai >= extreme) {
                extreme = ai;
                allnan = 0;
            }
        }
        if (allnan) extreme = BN_NAN;
        YPP = extreme;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmax, float32)
{
    npy_float32 ai, extreme = -BN_INFINITY;
    int allnan = 1;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.nanmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
            if (ai >= extreme) {
                extreme = ai;
                allnan = 0;
            }
        }
        NEXT
    }
    if (allnan) extreme = BN_NAN;
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(extreme);
}

REDUCE_ONE(nanmax, float32)
{
    npy_float32 ai, extreme;
    int allnan;
    INIT_ONE(float32, float32)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = -BN_INFINITY;
        allnan = 1;
        FOR {
            ai = AI(float32);
            if (ai >= extreme) {
                extreme = ai;
                allnan = 0;
            }
        }
        if (allnan) extreme = BN_NAN;
        YPP = extreme;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmax, int64)
{
    npy_int64 ai, extreme = NPY_MIN_int64;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.nanmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(int64);
            if (ai >= extreme) extreme = ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(extreme);
}

REDUCE_ONE(nanmax, int64)
{
    npy_int64 ai, extreme;
    INIT_ONE(int64, int64)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = NPY_MIN_int64;
        FOR {
            ai = AI(int64);
            if (ai >= extreme) extreme = ai;
        }
        YPP = extreme;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmax, int32)
{
    npy_int32 ai, extreme = NPY_MIN_int32;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.nanmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(int32);
            if (ai >= extreme) extreme = ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(extreme);
}

REDUCE_ONE(nanmax, int32)
{
    npy_int32 ai, extreme;
    INIT_ONE(int32, int32)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = NPY_MIN_int32;
        FOR {
            ai = AI(int32);
            if (ai >= extreme) extreme = ai;
        }
        YPP = extreme;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(nanmax, 0)

/* nanargmin, nanargmax -------------------------------------------------- */

REDUCE_ALL(nanargmin, float64)
{
    npy_float64 ai, extreme = BN_INFINITY;
    int allnan = 1;
    Py_ssize_t idx = 0;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.nanargmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        ai = AI(float64);
        if (ai <= extreme) {
            extreme = ai;
            allnan = 0;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    if (allnan) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    } else {
        return PyInt_FromLong(idx);
    }
}

REDUCE_ONE(nanargmin, float64)
{
    int allnan, err_code = 0;
    Py_ssize_t idx = 0;
    npy_float64 ai, extreme;
    INIT_ONE(INTP, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanargmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = BN_INFINITY;
        allnan = 1;
        FOR_REVERSE {
            ai = AI(float64);
            if (ai <= extreme) {
                extreme = ai;
                allnan = 0;
                idx = INDEX;
            }
        }
        if (allnan == 0) {
            YPP = idx;
        } else {
            err_code = 1;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (err_code) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    }
    return y;
}

REDUCE_ALL(nanargmin, float32)
{
    npy_float32 ai, extreme = BN_INFINITY;
    int allnan = 1;
    Py_ssize_t idx = 0;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.nanargmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        ai = AI(float32);
        if (ai <= extreme) {
            extreme = ai;
            allnan = 0;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    if (allnan) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    } else {
        return PyInt_FromLong(idx);
    }
}

REDUCE_ONE(nanargmin, float32)
{
    int allnan, err_code = 0;
    Py_ssize_t idx = 0;
    npy_float32 ai, extreme;
    INIT_ONE(INTP, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanargmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = BN_INFINITY;
        allnan = 1;
        FOR_REVERSE {
            ai = AI(float32);
            if (ai <= extreme) {
                extreme = ai;
                allnan = 0;
                idx = INDEX;
            }
        }
        if (allnan == 0) {
            YPP = idx;
        } else {
            err_code = 1;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (err_code) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    }
    return y;
}

REDUCE_ALL(nanargmin, int64)
{
    npy_intp idx = 0;
    npy_int64 ai, extreme = NPY_MAX_int64;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.nanargmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        ai = AI(int64);
        if (ai <= extreme) {
            extreme = ai;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyInt_FromLong(idx);
}

REDUCE_ONE(nanargmin, int64)
{
    npy_intp idx = 0;
    npy_int64 ai, extreme;
    INIT_ONE(intp, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanargmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = NPY_MAX_int64;
        FOR_REVERSE{
            ai = AI(int64);
            if (ai <= extreme) {
                extreme = ai;
                idx = INDEX;
            }
        }
        YPP = idx;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanargmin, int32)
{
    npy_intp idx = 0;
    npy_int32 ai, extreme = NPY_MAX_int32;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.nanargmin raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        ai = AI(int32);
        if (ai <= extreme) {
            extreme = ai;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyInt_FromLong(idx);
}

REDUCE_ONE(nanargmin, int32)
{
    npy_intp idx = 0;
    npy_int32 ai, extreme;
    INIT_ONE(intp, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanargmin raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = NPY_MAX_int32;
        FOR_REVERSE{
            ai = AI(int32);
            if (ai <= extreme) {
                extreme = ai;
                idx = INDEX;
            }
        }
        YPP = idx;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(nanargmin, 0)

REDUCE_ALL(nanargmax, float64)
{
    npy_float64 ai, extreme = -BN_INFINITY;
    int allnan = 1;
    Py_ssize_t idx = 0;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.nanargmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        ai = AI(float64);
        if (ai >= extreme) {
            extreme = ai;
            allnan = 0;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    if (allnan) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    } else {
        return PyInt_FromLong(idx);
    }
}

REDUCE_ONE(nanargmax, float64)
{
    int allnan, err_code = 0;
    Py_ssize_t idx = 0;
    npy_float64 ai, extreme;
    INIT_ONE(INTP, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanargmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = -BN_INFINITY;
        allnan = 1;
        FOR_REVERSE {
            ai = AI(float64);
            if (ai >= extreme) {
                extreme = ai;
                allnan = 0;
                idx = INDEX;
            }
        }
        if (allnan == 0) {
            YPP = idx;
        } else {
            err_code = 1;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (err_code) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    }
    return y;
}

REDUCE_ALL(nanargmax, float32)
{
    npy_float32 ai, extreme = -BN_INFINITY;
    int allnan = 1;
    Py_ssize_t idx = 0;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.nanargmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        ai = AI(float32);
        if (ai >= extreme) {
            extreme = ai;
            allnan = 0;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    if (allnan) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    } else {
        return PyInt_FromLong(idx);
    }
}

REDUCE_ONE(nanargmax, float32)
{
    int allnan, err_code = 0;
    Py_ssize_t idx = 0;
    npy_float32 ai, extreme;
    INIT_ONE(INTP, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanargmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = -BN_INFINITY;
        allnan = 1;
        FOR_REVERSE {
            ai = AI(float32);
            if (ai >= extreme) {
                extreme = ai;
                allnan = 0;
                idx = INDEX;
            }
        }
        if (allnan == 0) {
            YPP = idx;
        } else {
            err_code = 1;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (err_code) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    }
    return y;
}

REDUCE_ALL(nanargmax, int64)
{
    npy_intp idx = 0;
    npy_int64 ai, extreme = NPY_MIN_int64;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.nanargmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        ai = AI(int64);
        if (ai >= extreme) {
            extreme = ai;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyInt_FromLong(idx);
}

REDUCE_ONE(nanargmax, int64)
{
    npy_intp idx = 0;
    npy_int64 ai, extreme;
    INIT_ONE(intp, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanargmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = NPY_MIN_int64;
        FOR_REVERSE{
            ai = AI(int64);
            if (ai >= extreme) {
                extreme = ai;
                idx = INDEX;
            }
        }
        YPP = idx;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanargmax, int32)
{
    npy_intp idx = 0;
    npy_int32 ai, extreme = NPY_MIN_int32;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.nanargmax raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        ai = AI(int32);
        if (ai >= extreme) {
            extreme = ai;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyInt_FromLong(idx);
}

REDUCE_ONE(nanargmax, int32)
{
    npy_intp idx = 0;
    npy_int32 ai, extreme;
    INIT_ONE(intp, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.nanargmax raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        extreme = NPY_MIN_int32;
        FOR_REVERSE{
            ai = AI(int32);
            if (ai >= extreme) {
                extreme = ai;
                idx = INDEX;
            }
        }
        YPP = idx;
        NEXT
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(nanargmax, 0)

/* ss ---------------------------------------------------------------- */

REDUCE_ALL(ss, float64)
{
    npy_float64 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
            asum += ai * ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}

REDUCE_ONE(ss, float64)
{
    npy_float64 ai, asum;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR{
                ai = AI(float64);
                asum += ai * ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(ss, float32)
{
    npy_float32 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
            asum += ai * ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}

REDUCE_ONE(ss, float32)
{
    npy_float32 ai, asum;
    INIT_ONE(float32, float32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR{
                ai = AI(float32);
                asum += ai * ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(ss, int64)
{
    npy_int64 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(int64);
            asum += ai * ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}

REDUCE_ONE(ss, int64)
{
    npy_int64 ai, asum;
    INIT_ONE(int64, int64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR{
                ai = AI(int64);
                asum += ai * ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(ss, int32)
{
    npy_int32 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(int32);
            asum += ai * ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}

REDUCE_ONE(ss, int32)
{
    npy_int32 ai, asum;
    INIT_ONE(int32, int32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR{
                ai = AI(int32);
                asum += ai * ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(ss, 0)

/* median, nanmedian MACROS ---------------------------------------------- */

#define B(dtype, i) buffer[i] /* used by PARTITION */

#define EVEN_ODD(dtype, N) \
    if (N % 2 == 0) { \
        npy_##dtype amax = B(dtype, 0); \
        for (i = 1; i < k; i++) { \
            ai = B(dtype, i); \
            if (ai > amax) amax = ai; \
        } \
        med = 0.5 * (B(dtype, k) + amax); \
    } \
    else { \
        med =  B(dtype, k); \
    } \

#define MEDIAN(dtype) \
    npy_intp j, l, r, k; \
    npy_##dtype ai; \
    l = 0; \
    for (i = 0; i < LENGTH; i++) { \
        ai = AX(dtype, i); \
        if (ai == ai) { \
            B(dtype, l++) = ai; \
        } \
    } \
    if (l != LENGTH) { \
        med = BN_NAN; \
        goto done; \
    } \
    k = LENGTH >> 1; \
    l = 0; \
    r = LENGTH - 1; \
    PARTITION(dtype) \
    EVEN_ODD(dtype, LENGTH)

#define MEDIAN_INT(dtype) \
    npy_intp j, l, r, k; \
    npy_##dtype ai; \
    for (i = 0; i < LENGTH; i++) { \
        B(dtype, i) = AX(dtype, i); \
    } \
    k = LENGTH >> 1; \
    l = 0; \
    r = LENGTH - 1; \
    PARTITION(dtype) \
    EVEN_ODD(dtype, LENGTH)

#define NANMEDIAN(dtype) \
    npy_intp j, l, r, k, n; \
    npy_##dtype ai; \
    l = 0; \
    for (i = 0; i < LENGTH; i++) { \
        ai = AX(dtype, i); \
        if (ai == ai) { \
            B(dtype, l++) = ai; \
        } \
    } \
    n = l; \
    k = n >> 1; \
    l = 0; \
    r = n - 1; \
    if (n == 0) { \
        med = BN_NAN; \
        goto done; \
    } \
    PARTITION(dtype) \
    EVEN_ODD(dtype, n)

#define BUFFER_NEW(dtype, length) \
        npy_##dtype *buffer = malloc(length * sizeof(npy_##dtype));

#define BUFFER_DELETE free(buffer);

/* median, nanmedian ----------------------------------------------------- */

REDUCE_ALL(median, float64)
{
    npy_intp i;
    npy_float64 med;
    INIT_ALL_RAVEL
    BN_BEGIN_ALLOW_THREADS
    BUFFER_NEW(float64, LENGTH)
    if (LENGTH == 0) {
        med = BN_NAN;
    }
    else {
        MEDIAN(float64)
    }
    done:
    BUFFER_DELETE
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyFloat_FromDouble(med);
}

REDUCE_ONE(median, float64)
{
    npy_intp i;
    npy_float64 med;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        BUFFER_NEW(float64, LENGTH)
        WHILE {
            MEDIAN(float64)
            done:
            YPP = med;
            NEXT
        }
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(median, float32)
{
    npy_intp i;
    npy_float32 med;
    INIT_ALL_RAVEL
    BN_BEGIN_ALLOW_THREADS
    BUFFER_NEW(float32, LENGTH)
    if (LENGTH == 0) {
        med = BN_NAN;
    }
    else {
        MEDIAN(float32)
    }
    done:
    BUFFER_DELETE
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyFloat_FromDouble(med);
}

REDUCE_ONE(median, float32)
{
    npy_intp i;
    npy_float32 med;
    INIT_ONE(float32, float32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        BUFFER_NEW(float32, LENGTH)
        WHILE {
            MEDIAN(float32)
            done:
            YPP = med;
            NEXT
        }
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmedian, float64)
{
    npy_intp i;
    npy_float64 med;
    INIT_ALL_RAVEL
    BN_BEGIN_ALLOW_THREADS
    BUFFER_NEW(float64, LENGTH)
    if (LENGTH == 0) {
        med = BN_NAN;
    }
    else {
        NANMEDIAN(float64)
    }
    done:
    BUFFER_DELETE
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyFloat_FromDouble(med);
}

REDUCE_ONE(nanmedian, float64)
{
    npy_intp i;
    npy_float64 med;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        BUFFER_NEW(float64, LENGTH)
        WHILE {
            NANMEDIAN(float64)
            done:
            YPP = med;
            NEXT
        }
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(nanmedian, float32)
{
    npy_intp i;
    npy_float32 med;
    INIT_ALL_RAVEL
    BN_BEGIN_ALLOW_THREADS
    BUFFER_NEW(float32, LENGTH)
    if (LENGTH == 0) {
        med = BN_NAN;
    }
    else {
        NANMEDIAN(float32)
    }
    done:
    BUFFER_DELETE
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyFloat_FromDouble(med);
}

REDUCE_ONE(nanmedian, float32)
{
    npy_intp i;
    npy_float32 med;
    INIT_ONE(float32, float32)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        BUFFER_NEW(float32, LENGTH)
        WHILE {
            NANMEDIAN(float32)
            done:
            YPP = med;
            NEXT
        }
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(median, int64)
{
    npy_intp i;
    npy_float64 med;
    INIT_ALL_RAVEL
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        med = BN_NAN;
    }
    else {
        BUFFER_NEW(int64, LENGTH)
        MEDIAN_INT(int64)
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyFloat_FromDouble(med);
}

REDUCE_ONE(median, int64)
{
    npy_intp i;
    npy_float64 med;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        BUFFER_NEW(int64, LENGTH)
        WHILE {
            MEDIAN_INT(int64)
            YPP = med;
            NEXT
        }
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(median, int32)
{
    npy_intp i;
    npy_float64 med;
    INIT_ALL_RAVEL
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        med = BN_NAN;
    }
    else {
        BUFFER_NEW(int32, LENGTH)
        MEDIAN_INT(int32)
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyFloat_FromDouble(med);
}

REDUCE_ONE(median, int32)
{
    npy_intp i;
    npy_float64 med;
    INIT_ONE(float64, float64)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN)
    }
    else {
        BUFFER_NEW(int32, LENGTH)
        WHILE {
            MEDIAN_INT(int32)
            YPP = med;
            NEXT
        }
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(median, 0)

static PyObject *
nanmedian(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer("nanmedian",
                   args,
                   kwds,
                   nanmedian_all_float64,
                   nanmedian_all_float32,
                   median_all_int64,
                   median_all_int32,
                   nanmedian_one_float64,
                   nanmedian_one_float32,
                   median_one_int64,
                   median_one_int32,
                   0);
}

/* anynan ---------------------------------------------------------------- */

REDUCE_ALL(anynan, float64)
{
    int f = 0;
    npy_float64 ai;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
            if (ai != ai) {
                f = 1;
                goto done;
            }
        }
        NEXT
    }
    done:
    BN_END_ALLOW_THREADS
    if (f) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

REDUCE_ONE(anynan, float64)
{
    int f;
    npy_float64 ai;
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            f = 0;
            FOR {
                ai = AI(float64);
                if (ai != ai) {
                    f = 1;
                    break;
                }
            }
            YPP = f;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(anynan, float32)
{
    int f = 0;
    npy_float32 ai;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
            if (ai != ai) {
                f = 1;
                goto done;
            }
        }
        NEXT
    }
    done:
    BN_END_ALLOW_THREADS
    if (f) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

REDUCE_ONE(anynan, float32)
{
    int f;
    npy_float32 ai;
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            f = 0;
            FOR {
                ai = AI(float32);
                if (ai != ai) {
                    f = 1;
                    break;
                }
            }
            YPP = f;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(anynan, int64)
{
    Py_RETURN_FALSE;
}

REDUCE_ONE(anynan, int64)
{
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    FILL_Y(0);
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(anynan, int32)
{
    Py_RETURN_FALSE;
}

REDUCE_ONE(anynan, int32)
{
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    FILL_Y(0);
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(anynan, 0)

/* allnan ---------------------------------------------------------------- */

REDUCE_ALL(allnan, float64)
{
    int f = 0;
    npy_float64 ai;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float64);
            if (ai == ai) {
                f = 1;
                goto done;
            }
        }
        NEXT
    }
    done:
    BN_END_ALLOW_THREADS
    if (f) Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

REDUCE_ONE(allnan, float64)
{
    int f;
    npy_float64 ai;
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(1)
    }
    else {
        WHILE {
            f = 1;
            FOR {
                ai = AI(float64);
                if (ai == ai) {
                    f = 0;
                    break;
                }
            }
            YPP = f;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(allnan, float32)
{
    int f = 0;
    npy_float32 ai;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(float32);
            if (ai == ai) {
                f = 1;
                goto done;
            }
        }
        NEXT
    }
    done:
    BN_END_ALLOW_THREADS
    if (f) Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

REDUCE_ONE(allnan, float32)
{
    int f;
    npy_float32 ai;
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(1)
    }
    else {
        WHILE {
            f = 1;
            FOR {
                ai = AI(float32);
                if (ai == ai) {
                    f = 0;
                    break;
                }
            }
            YPP = f;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(allnan, int64)
{
    if (PyArray_SIZE(a) == 0) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

REDUCE_ONE(allnan, int64)
{
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (SIZE == 0) {
        FILL_Y(1);
    }
    else {
        FILL_Y(0);
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_ALL(allnan, int32)
{
    if (PyArray_SIZE(a) == 0) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

REDUCE_ONE(allnan, int32)
{
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (SIZE == 0) {
        FILL_Y(1);
    }
    else {
        FILL_Y(0);
    }
    BN_END_ALLOW_THREADS
    return y;
}

REDUCE_MAIN(allnan, 0)

/* python strings -------------------------------------------------------- */

PyObject *pystr_a = NULL;
PyObject *pystr_axis = NULL;
PyObject *pystr_ddof = NULL;

static int
intern_strings(void) {
    pystr_a = PyString_InternFromString("a");
    pystr_axis = PyString_InternFromString("axis");
    pystr_ddof = PyString_InternFromString("ddof");
    return pystr_a && pystr_axis && pystr_ddof;
}

/* reducer --------------------------------------------------------------- */

static BN_INLINE int
parse_args(PyObject *args,
           PyObject *kwds,
           int has_ddof,
           PyObject **a,
           PyObject **axis,
           PyObject **ddof)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int nkwds_found = 0;
        PyObject *tmp;
        switch (nargs) {
            case 2:
                if (has_ddof) {
                    *axis = PyTuple_GET_ITEM(args, 1);
                } else {
                    TYPE_ERR("wrong number of arguments");
                    return 0;
                }
            case 1: *a = PyTuple_GET_ITEM(args, 0);
            case 0: break;
            default:
                TYPE_ERR("wrong number of arguments");
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
                *a = PyTuple_GET_ITEM(args, 0);
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
        int has_ddof)
{

    int ndim;
    int axis = 0; /* initialize to avoid compiler error */
    int dtype;
    int ddof;
    int reduce_all = 0;

    PyArrayObject *a;

    PyObject *a_obj = NULL;
    PyObject *axis_obj = Py_None;
    PyObject *ddof_obj = NULL;

    if (!parse_args(args, kwds, has_ddof, &a_obj, &axis_obj, &ddof_obj)) {
        return NULL;
    }

    /* convert to array if necessary */
    if PyArray_Check(a_obj) {
        a = (PyArrayObject *)a_obj;
    } else {
        a = (PyArrayObject *)PyArray_FROM_O(a_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        return slow(name, args, kwds);
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
        ndim = PyArray_NDIM(a);
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
        if (ndim == 1) {
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

    if (reduce_all == 1) {
        /* we are reducing the array along all axes */
        if (dtype == NPY_FLOAT64) {
            return fall_float64(a, ddof);
        }
        else if (dtype == NPY_FLOAT32) {
            return fall_float32(a, ddof);
        }
        else if (dtype == NPY_INT64) {
            return fall_int64(a, ddof);
        }
        else if (dtype == NPY_INT32) {
            return fall_int32(a, ddof);
        }
        else {
            return slow(name, args, kwds);
        }
    }
    else {
        /* we are reducing an array with ndim > 1 over a single axis */
        if (dtype == NPY_FLOAT64) {
            return fone_float64(a, axis, ddof);
        }
        else if (dtype == NPY_FLOAT32) {
            return fone_float32(a, axis, ddof);
        }
        else if (dtype == NPY_INT64) {
            return fone_int64(a, axis, ddof);
        }
        else if (dtype == NPY_INT32) {
            return fone_int32(a, axis, ddof);
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
"nansum(a, axis=None)\n"
"\n"
"Sum of array elements along given axis treating NaNs as zero.\n"
"\n"
"The data type (dtype) of the output is the same as the input. On 64-bit\n"
"operating systems, 32-bit input is NOT upcast to 64-bit accumulator and\n"
"return values.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Array containing numbers whose sum is desired. If `a` is not an\n"
"    array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the sum is computed. The default (axis=None) is to\n"
"    compute the sum of the flattened array.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    An array with the same shape as `a`, with the specified axis removed.\n"
"    If `a` is a 0-d array, or if axis is None, a scalar is returned.\n"
"\n"
"Notes\n"
"-----\n"
"No error is raised on overflow.\n"
"\n"
"If positive or negative infinity are present the result is positive or\n"
"negative infinity. But if both positive and negative infinity are present,\n"
"the result is Not A Number (NaN).\n"
"\n"
"Examples\n"
"--------\n"
">>> bn.nansum(1)\n"
"1\n"
">>> bn.nansum([1])\n"
"1\n"
">>> bn.nansum([1, np.nan])\n"
"1.0\n"
">>> a = np.array([[1, 1], [1, np.nan]])\n"
">>> bn.nansum(a)\n"
"3.0\n"
">>> bn.nansum(a, axis=0)\n"
"array([ 2.,  1.])\n"
"\n"
"When positive infinity and negative infinity are present:\n"
"\n"
">>> bn.nansum([1, np.nan, np.inf])\n"
"inf\n"
">>> bn.nansum([1, np.nan, np.NINF])\n"
"-inf\n"
">>> bn.nansum([1, np.nan, np.inf, np.NINF])\n"
"nan\n"
"\n";

static char nanmean_doc[] =
"nanmean(a, axis=None)\n"
"\n"
"Mean of array elements along given axis ignoring NaNs.\n"
"\n"
"`float64` intermediate and return values are used for integer inputs.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Array containing numbers whose mean is desired. If `a` is not an\n"
"    array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the means are computed. The default (axis=None) is to\n"
"    compute the mean of the flattened array.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    An array with the same shape as `a`, with the specified axis removed.\n"
"    If `a` is a 0-d array, or if axis is None, a scalar is returned.\n"
"    `float64` intermediate and return values are used for integer inputs.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.nanmedian: Median along specified axis, ignoring NaNs.\n"
"\n"
"Notes\n"
"-----\n"
"No error is raised on overflow. (The sum is computed and then the result\n"
"is divided by the number of non-NaN elements.)\n"
"\n"
"If positive or negative infinity are present the result is positive or\n"
"negative infinity. But if both positive and negative infinity are present,\n"
"the result is Not A Number (NaN).\n"
"\n"
"Examples\n"
"--------\n"
">>> bn.nanmean(1)\n"
"1.0\n"
">>> bn.nanmean([1])\n"
"1.0\n"
">>> bn.nanmean([1, np.nan])\n"
"1.0\n"
">>> a = np.array([[1, 4], [1, np.nan]])\n"
">>> bn.nanmean(a)\n"
"2.0\n"
">>> bn.nanmean(a, axis=0)\n"
"array([ 1.,  4.])\n"
"\n"
"When positive infinity and negative infinity are present:\n"
"\n"
">>> bn.nanmean([1, np.nan, np.inf])\n"
"inf\n"
">>> bn.nanmean([1, np.nan, np.NINF])\n"
"-inf\n"
">>> bn.nanmean([1, np.nan, np.inf, np.NINF])\n"
"nan\n"
"\n";

static char nanstd_doc[] =
"nanstd(a, axis=None, ddof=0)\n"
"\n"
"Standard deviation along the specified axis, ignoring NaNs.\n"
"\n"
"`float64` intermediate and return values are used for integer inputs.\n"
"\n"
"Instead of a faster one-pass algorithm, a more stable two-pass algorithm\n"
"is used.\n"
"\n"
"An example of a one-pass algorithm:\n"
"\n"
"    >>> np.sqrt((a*a).mean() - a.mean()**2)\n"
"\n"
"An example of a two-pass algorithm:\n"
"\n"
"    >>> np.sqrt(((a - a.mean())**2).mean())\n"
"\n"
"Note in the two-pass algorithm the mean must be found (first pass) before\n"
"the squared deviation (second pass) can be found.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the standard deviation is computed. The default\n"
"    (axis=None) is to compute the standard deviation of the flattened\n"
"    array.\n"
"ddof : int, optional\n"
"    Means Delta Degrees of Freedom. The divisor used in calculations\n"
"    is ``N - ddof``, where ``N`` represents the number of non-NaN elements.\n"
"    By default `ddof` is zero.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    An array with the same shape as `a`, with the specified axis removed.\n"
"    If `a` is a 0-d array, or if axis is None, a scalar is returned.\n"
"    `float64` intermediate and return values are used for integer inputs.\n"
"    If ddof is >= the number of non-NaN elements in a slice or the slice\n"
"    contains only NaNs, then the result for that slice is NaN.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.nanvar: Variance along specified axis ignoring NaNs\n"
"\n"
"Notes\n"
"-----\n"
"If positive or negative infinity are present the result is Not A Number\n"
"(NaN).\n"
"\n"
"Examples\n"
"--------\n"
">>> bn.nanstd(1)\n"
"0.0\n"
">>> bn.nanstd([1])\n"
"0.0\n"
">>> bn.nanstd([1, np.nan])\n"
"0.0\n"
">>> a = np.array([[1, 4], [1, np.nan]])\n"
">>> bn.nanstd(a)\n"
"1.4142135623730951\n"
">>> bn.nanstd(a, axis=0)\n"
"array([ 0.,  0.])\n"
"\n"
"When positive infinity or negative infinity are present NaN is returned:\n"
"\n"
">>> bn.nanstd([1, np.nan, np.inf])\n"
"nan\n"
"\n";

static char nanvar_doc[] =
"nanvar(a, axis=None, ddof=0)\n"
"\n"
"Variance along the specified axis, ignoring NaNs.\n"
"\n"
"`float64` intermediate and return values are used for integer inputs.\n"
"\n"
"Instead of a faster one-pass algorithm, a more stable two-pass algorithm\n"
"is used.\n"
"\n"
"An example of a one-pass algorithm:\n"
"\n"
"    >>> (a*a).mean() - a.mean()**2\n"
"\n"
"An example of a two-pass algorithm:\n"
"\n"
"    >>> ((a - a.mean())**2).mean()\n"
"\n"
"Note in the two-pass algorithm the mean must be found (first pass) before\n"
"the squared deviation (second pass) can be found.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the variance is computed. The default (axis=None) is\n"
"    to compute the variance of the flattened array.\n"
"ddof : int, optional\n"
"    Means Delta Degrees of Freedom. The divisor used in calculations\n"
"    is ``N - ddof``, where ``N`` represents the number of non_NaN elements.\n"
"    By default `ddof` is zero.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    An array with the same shape as `a`, with the specified axis\n"
"    removed. If `a` is a 0-d array, or if axis is None, a scalar is\n"
"    returned. `float64` intermediate and return values are used for\n"
"    integer inputs. If ddof is >= the number of non-NaN elements in a\n"
"    slice or the slice contains only NaNs, then the result for that slice\n"
"    is NaN.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.nanstd: Standard deviation along specified axis ignoring NaNs.\n"
"\n"
"Notes\n"
"-----\n"
"If positive or negative infinity are present the result is Not A Number\n"
"(NaN).\n"
"\n"
"Examples\n"
"--------\n"
">>> bn.nanvar(1)\n"
"0.0\n"
">>> bn.nanvar([1])\n"
"0.0\n"
">>> bn.nanvar([1, np.nan])\n"
"0.0\n"
">>> a = np.array([[1, 4], [1, np.nan]])\n"
">>> bn.nanvar(a)\n"
"2.0\n"
">>> bn.nanvar(a, axis=0)\n"
"array([ 0.,  0.])\n"
"\n"
"When positive infinity or negative infinity are present NaN is returned:\n"
"\n"
">>> bn.nanvar([1, np.nan, np.inf])\n"
"nan\n"
"\n";

static char nanmin_doc[] =
"nanmin(a, axis=None)\n"
"\n"
"Minimum values along specified axis, ignoring NaNs.\n"
"\n"
"When all-NaN slices are encountered, NaN is returned for that slice.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the minimum is computed. The default (axis=None) is\n"
"    to compute the minimum of the flattened array.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    An array with the same shape as `a`, with the specified axis removed.\n"
"    If `a` is a 0-d array, or if axis is None, a scalar is returned. The\n"
"    same dtype as `a` is returned.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.nanmax: Maximum along specified axis, ignoring NaNs.\n"
"bottleneck.nanargmin: Indices of minimum values along axis, ignoring NaNs.\n"
"\n"
"Examples\n"
"--------\n"
">>> bn.nanmin(1)\n"
"1\n"
">>> bn.nanmin([1])\n"
"1\n"
">>> bn.nanmin([1, np.nan])\n"
"1.0\n"
">>> a = np.array([[1, 4], [1, np.nan]])\n"
">>> bn.nanmin(a)\n"
"1.0\n"
">>> bn.nanmin(a, axis=0)\n"
"array([ 1.,  4.])\n"
"\n";

static char nanmax_doc[] =
"nanmax(a, axis=None)\n"
"\n"
"Maximum values along specified axis, ignoring NaNs.\n"
"\n"
"When all-NaN slices are encountered, NaN is returned for that slice.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the maximum is computed. The default (axis=None) is\n"
"    to compute the maximum of the flattened array.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    An array with the same shape as `a`, with the specified axis removed.\n"
"    If `a` is a 0-d array, or if axis is None, a scalar is returned. The\n"
"    same dtype as `a` is returned.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.nanmin: Minimum along specified axis, ignoring NaNs.\n"
"bottleneck.nanargmax: Indices of maximum values along axis, ignoring NaNs.\n"
"\n"
"Examples\n"
"--------\n"
">>> bn.nanmax(1)\n"
"1\n"
">>> bn.nanmax([1])\n"
"1\n"
">>> bn.nanmax([1, np.nan])\n"
"1.0\n"
">>> a = np.array([[1, 4], [1, np.nan]])\n"
">>> bn.nanmax(a)\n"
"4.0\n"
">>> bn.nanmax(a, axis=0)\n"
"array([ 1.,  4.])\n"
"\n";

static char nanargmin_doc[] =
"nanargmin(a, axis=None)\n"
"\n"
"Indices of the minimum values along an axis, ignoring NaNs.\n"
"\n"
"For all-NaN slices ``ValueError`` is raised. Unlike NumPy, the results\n"
"can be trusted if a slice contains only NaNs and Infs.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which to operate. By default (axis=None) flattened input\n"
"    is used.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.nanargmax: Indices of the maximum values along an axis.\n"
"bottleneck.nanmin: Minimum values along specified axis, ignoring NaNs.\n"
"\n"
"Returns\n"
"-------\n"
"index_array : ndarray\n"
"    An array of indices or a single index value.\n"
"\n"
"Examples\n"
"--------\n"
">>> a = np.array([[np.nan, 4], [2, 3]])\n"
">>> bn.nanargmin(a)\n"
"2\n"
">>> a.flat[2]\n"
"2.0\n"
">>> bn.nanargmin(a, axis=0)\n"
"array([1, 1])\n"
">>> bn.nanargmin(a, axis=1)\n"
"array([1, 0])\n"
"\n";

static char nanargmax_doc[] =
"nanargmax(a, axis=None)\n"
"\n"
"Indices of the maximum values along an axis, ignoring NaNs.\n"
"\n"
"For all-NaN slices ``ValueError`` is raised. Unlike NumPy, the results\n"
"can be trusted if a slice contains only NaNs and Infs.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which to operate. By default (axis=None) flattened input\n"
"    is used.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.nanargmin: Indices of the minimum values along an axis.\n"
"bottleneck.nanmax: Maximum values along specified axis, ignoring NaNs.\n"
"\n"
"Returns\n"
"-------\n"
"index_array : ndarray\n"
"    An array of indices or a single index value.\n"
"\n"
"Examples\n"
"--------\n"
">>> a = np.array([[np.nan, 4], [2, 3]])\n"
">>> bn.nanargmax(a)\n"
"1\n"
">>> a.flat[1]\n"
"4.0\n"
">>> bn.nanargmax(a, axis=0)\n"
"array([1, 0])\n"
">>> bn.nanargmax(a, axis=1)\n"
"array([1, 1])\n"
"\n";

static char ss_doc[] =
"ss(a, axis=None)\n"
"\n"
"Sum of the square of each element along the specified axis.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Array whose sum of squares is desired. If `a` is not an array, a\n"
"    conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the sum of squares is computed. The default\n"
"    (axis=None) is to sum the squares of the flattened array.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    The sum of a**2 along the given axis.\n"
"\n"
"Examples\n"
"--------\n"
">>> a = np.array([1., 2., 5.])\n"
">>> bn.ss(a)\n"
"30.0\n"
"\n"
"And calculating along an axis:\n"
"\n"
">>> b = np.array([[1., 2., 5.], [2., 5., 6.]])\n"
">>> bn.ss(b, axis=1)\n"
"array([ 30., 65.])\n"
"\n";

static char median_doc[] =
"median(a, axis=None)\n"
"\n"
"Median of array elements along given axis.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the median is computed. The default (axis=None) is to\n"
"    compute the median of the flattened array.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    An array with the same shape as `a`, except that the specified axis\n"
"    has been removed. If `a` is a 0d array, or if axis is None, a scalar\n"
"    is returned. `float64` return values are used for integer inputs. NaN\n"
"    is returned for a slice that contains one or more NaNs.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.nanmedian: Median along specified axis ignoring NaNs.\n"
"\n"
"Examples\n"
"--------\n"
">>> a = np.array([[10, 7, 4], [3, 2, 1]])\n"
">>> bn.median(a)\n"
"    3.5\n"
">>> bn.median(a, axis=0)\n"
"    array([ 6.5,  4.5,  2.5])\n"
">>> bn.median(a, axis=1)\n"
"    array([ 7.,  2.])\n"
"\n";

static char nanmedian_doc[] =
"nanmedian(a, axis=None)\n"
"\n"
"Median of array elements along given axis ignoring NaNs.\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which the median is computed. The default (axis=None) is to\n"
"    compute the median of the flattened array.\n"
"\n"
"Returns\n"
"-------\n"
"y : ndarray\n"
"    An array with the same shape as `a`, except that the specified axis\n"
"    has been removed. If `a` is a 0d array, or if axis is None, a scalar\n"
"    is returned. `float64` return values are used for integer inputs.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.median: Median along specified axis.\n"
"\n"
"Examples\n"
"--------\n"
">>> a = np.array([[np.nan, 7, 4], [3, 2, 1]])\n"
">>> a\n"
"array([[ nan,   7.,   4.],\n"
"       [  3.,   2.,   1.]])\n"
">>> bn.nanmedian(a)\n"
"3.0\n"
">> bn.nanmedian(a, axis=0)\n"
"array([ 3. ,  4.5,  2.5])\n"
">> bn.nanmedian(a, axis=1)\n"
"array([ 5.5,  2. ])\n"
"\n";

static char anynan_doc[] =
"anynan(a, axis=None)\n"
"\n"
"Test whether any array element along a given axis is NaN.\n"
"\n"
"Returns the same output as np.isnan(a).any(axis)\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which NaNs are searched. The default (`axis` = ``None``)\n"
"    is to search for NaNs over a flattened input array.\n"
"\n"
"Returns\n"
"-------\n"
"y : bool or ndarray\n"
"    A boolean or new `ndarray` is returned.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.allnan: Test if all array elements along given axis are NaN\n"
"\n"
"Examples\n"
"--------\n"
">>> bn.anynan(1)\n"
"False\n"
">>> bn.anynan(np.nan)\n"
"True\n"
">>> bn.anynan([1, np.nan])\n"
"True\n"
">>> a = np.array([[1, 4], [1, np.nan]])\n"
">>> bn.anynan(a)\n"
"True\n"
">>> bn.anynan(a, axis=0)\n"
"array([False,  True], dtype=bool)\n"
"\n";

static char allnan_doc[] =
"allnan(a, axis=None)\n"
"\n"
"Test whether all array elements along a given axis are NaN.\n"
"\n"
"Returns the same output as np.isnan(a).all(axis)\n"
"\n"
"Note that allnan([]) is True to match np.isnan([]).all() and all([])\n"
"\n"
"Parameters\n"
"----------\n"
"a : array_like\n"
"    Input array. If `a` is not an array, a conversion is attempted.\n"
"axis : {int, None}, optional\n"
"    Axis along which NaNs are searched. The default (`axis` = ``None``)\n"
"    is to search for NaNs over a flattened input array.\n"
"\n"
"Returns\n"
"-------\n"
"y : bool or ndarray\n"
"    A boolean or new `ndarray` is returned.\n"
"\n"
"See also\n"
"--------\n"
"bottleneck.anynan: Test if any array element along given axis is NaN\n"
"\n"
"Examples\n"
"--------\n"
">>> bn.allnan(1)\n"
"False\n"
">>> bn.allnan(np.nan)\n"
"True\n"
">>> bn.allnan([1, np.nan])\n"
"False\n"
">>> a = np.array([[1, np.nan], [1, np.nan]])\n"
">>> bn.allnan(a)\n"
"False\n"
">>> bn.allnan(a, axis=0)\n"
"array([False,  True], dtype=bool)\n"
"\n"
"An empty array returns True:\n"
"\n"
">>> bn.allnan([])\n"
"True\n"
"\n"
"which is similar to:\n"
"\n"
">>> all([])\n"
"True\n"
">>> np.isnan([]).all()\n"
"True\n"
"\n";

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
reduce_methods[] = {
    {"nansum",    (PyCFunction)nansum,    VARKEY, nansum_doc},
    {"nanmean",   (PyCFunction)nanmean,   VARKEY, nanmean_doc},
    {"nanstd",    (PyCFunction)nanstd,    VARKEY, nanstd_doc},
    {"nanvar",    (PyCFunction)nanvar,    VARKEY, nanvar_doc},
    {"nanmin",    (PyCFunction)nanmin,    VARKEY, nanmin_doc},
    {"nanmax",    (PyCFunction)nanmax,    VARKEY, nanmax_doc},
    {"nanargmin", (PyCFunction)nanargmin, VARKEY, nanargmin_doc},
    {"nanargmax", (PyCFunction)nanargmax, VARKEY, nanargmax_doc},
    {"ss",        (PyCFunction)ss,        VARKEY, ss_doc},
    {"median",    (PyCFunction)median,    VARKEY, median_doc},
    {"nanmedian", (PyCFunction)nanmedian, VARKEY, nanmedian_doc},
    {"anynan",    (PyCFunction)anynan,    VARKEY, anynan_doc},
    {"allnan",    (PyCFunction)allnan,    VARKEY, allnan_doc},
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