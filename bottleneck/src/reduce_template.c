// Copyright 2010-2019 Keith Goodman
// Copyright 2019 Bottleneck Developers
#include "bottleneck.h"
#include "iterators.h"

#ifdef _MSC_VER
    #include "intrin.h"
#elif HAVE_SSE2
    #include "x86intrin.h"
#endif

/* init macros ----------------------------------------------------------- */

#define INIT_ALL \
    iter it;     \
    init_iter_all(&it, a, 0, 1);

#define INIT_ALL_RAVEL \
    iter it;           \
    init_iter_all(&it, a, 1, 0);

#define INIT_ALL_RAVEL_ANY_ORDER \
    iter it;                     \
    init_iter_all(&it, a, 1, 1);

/* used with INIT_ALL_RAVEL */
#define DECREF_INIT_ALL_RAVEL  \
    if (it.a_ravel != NULL) {  \
        Py_DECREF(it.a_ravel); \
    }

#define INIT_ONE(dtype0, dtype1)                         \
    iter          it;                                    \
    PyObject *    y;                                     \
    npy_##dtype1 *py;                                    \
    init_iter_one(&it, a, axis);                         \
    y = PyArray_EMPTY(NDIM - 1, SHAPE, NPY_##dtype0, 0); \
    py = (npy_##dtype1 *)PyArray_DATA((PyArrayObject *)y);

/* function signatures --------------------------------------------------- */

/* low-level functions such as nansum_all_float64 */
#define REDUCE_ALL(name, dtype) \
    static PyObject *           \
        name##_all_##dtype(PyArrayObject *a, int ddof)

/* low-level functions such as nansum_one_float64 */
#define REDUCE_ONE(name, dtype) \
    static PyObject *           \
        name##_one_##dtype(PyArrayObject *a, int axis, int ddof)

/* top-level functions such as nansum */
#define REDUCE_MAIN(name, has_ddof)                        \
    static PyObject *                                      \
    name(PyObject *self, PyObject *args, PyObject *kwds) { \
        return reducer(#name,                              \
                       args,                               \
                       kwds,                               \
                       name##_all_float64,                 \
                       name##_all_float32,                 \
                       name##_all_int64,                   \
                       name##_all_int32,                   \
                       name##_one_float64,                 \
                       name##_one_float32,                 \
                       name##_one_int64,                   \
                       name##_one_int32,                   \
                       has_ddof);                          \
    }

/* typedefs and prototypes ----------------------------------------------- */

typedef PyObject *(*fall_t)(PyArrayObject *a, int ddof);
typedef PyObject *(*fone_t)(PyArrayObject *a, int axis, int ddof);

static PyObject *
reducer(char *    name,
        PyObject *args,
        PyObject *kwds,
        fall_t    fall_float64,
        fall_t    fall_float32,
        fall_t    fall_int64,
        fall_t    fall_int32,
        fone_t    fone_float64,
        fone_t    fone_float32,
        fone_t    fone_int64,
        fone_t    fone_int32,
        int       has_ddof);

/* nansum ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
REDUCE_ALL(nansum, DTYPE0) {
    npy_DTYPE0 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            const npy_DTYPE0 ai = AI(DTYPE0);
            if (!bn_isnan(ai)) {
                asum += ai;
            }
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}

REDUCE_ONE(nansum, DTYPE0) {
    INIT_ONE(DTYPE0, DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    } else {
        WHILE {
            npy_DTYPE0 asum = 0;
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                if (!bn_isnan(ai)) {
                    asum += ai;
                }
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64', 'int64'], ['int32', 'long']] */
BN_OPT_3
REDUCE_ALL(nansum, DTYPE0) {
    npy_DTYPE1 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    if (REDUCE_CONTIGUOUS) {
        npy_DTYPE0 *   pa = PA(DTYPE0);
        const npy_intp nits = it.nits;
        const npy_intp length = it.length;
        for (npy_intp i = 0; i < nits; i++) {
            for (npy_intp j = 0; j < length; j++) {
                asum += pa[i * length + j];
            }
        }
    } else {
        WHILE {
            npy_DTYPE0 *pa = PA(DTYPE0);
            FOR {
                asum += pa[it.i * it.stride];
            }
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return PyLong_FromLongLong(asum);
}

BN_OPT_3
REDUCE_ONE(nansum, DTYPE0) {
    INIT_ONE(DTYPE1, DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    } else {
        if (ONE_CONTIGUOUS) {
            WHILE {
                npy_DTYPE1        asum = 0;
                const npy_DTYPE0 *pa = PA(DTYPE0);
                for (npy_intp i = 0; i < LENGTH; i++) {
                    asum += pa[i];
                }
                YPP = asum;
                NEXT
            }
        } else if (ONE_TRANSPOSE(npy_DTYPE0)) {
            const npy_intp    LOOP_SIZE = it.nits;
            const npy_DTYPE0 *pa = PA(DTYPE0);

            for (npy_intp i = 0; i < LOOP_SIZE; i++) {
                py[i] = 0;
            }

            for (npy_intp i = 0; i < LENGTH; i++) {
                for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                    py[j] += pa[i * LOOP_SIZE + j];
                }
            }
        } else {
            WHILE {
                npy_DTYPE1        asum = 0;
                const npy_DTYPE0 *pa = PA(DTYPE0);
                FOR {
                    asum += SI(pa);
                }
                YPP = asum;
                NEXT
            }
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

REDUCE_MAIN(nansum, 0)

/* nanmean ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
REDUCE_ALL(nanmean, DTYPE0) {
    Py_ssize_t count = 0;
    npy_DTYPE0 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            const npy_DTYPE0 ai = AI(DTYPE0);
            if (!bn_isnan(ai)) {
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
        return PyFloat_FromDouble(BN_NAN_DTYPE0);
    }
}

REDUCE_ONE(nanmean, DTYPE0) {
    INIT_ONE(DTYPE0, DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN_DTYPE0)
    } else {
        WHILE {
            Py_ssize_t count = 0;
            npy_DTYPE0 asum = 0;
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                if (!bn_isnan(ai)) {
                    asum += ai;
                    count += 1;
                }
            }
            if (count > 0) {
                asum /= count;
            } else {
                asum = BN_NAN_DTYPE0;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
REDUCE_ALL(nanmean, DTYPE0) {
    Py_ssize_t total_length = 0;
    npy_DTYPE1 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            asum += AI(DTYPE0);
        }
        total_length += LENGTH;
        NEXT
    }
    BN_END_ALLOW_THREADS
    if (total_length > 0) {
        return PyFloat_FromDouble(asum / total_length);
    } else {
        return PyFloat_FromDouble(BN_NAN_DTYPE1);
    }
}

REDUCE_ONE(nanmean, DTYPE0) {
    INIT_ONE(DTYPE1, DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN_DTYPE1)
    } else {
        WHILE {
            npy_DTYPE1 asum = 0;
            FOR {
                asum += AI(DTYPE0);
            }
            if (LENGTH > 0) {
                asum /= LENGTH;
            } else {
                asum = BN_NAN_DTYPE1;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

REDUCE_MAIN(nanmean, 0)

/* nanstd, nanvar- ------------------------------------------------------- */

/* repeat = {'NAME': ['nanstd', 'nanvar'],
             'FUNC': ['sqrt',   '']} */
/* dtype = [['float64'], ['float32']] */
REDUCE_ALL(NAME, DTYPE0) {
    Py_ssize_t count = 0;
    npy_DTYPE0 out, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            const npy_DTYPE0 ai = AI(DTYPE0);
            if (!bn_isnan(ai)) {
                asum += ai;
                count++;
            }
        }
        NEXT
    }
    if (count > ddof) {
        const npy_DTYPE0 amean = asum / count;
        asum = 0;
        RESET
        WHILE {
            FOR {
                npy_DTYPE0 ai = AI(DTYPE0);
                if (!bn_isnan(ai)) {
                    ai -= amean;
                    asum += ai * ai;
                }
            }
            NEXT
        }
        out = FUNC(asum / (count - ddof));
    } else {
        out = BN_NAN_DTYPE0;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}

REDUCE_ONE(NAME, DTYPE0) {
    INIT_ONE(DTYPE0, DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN_DTYPE0)
    } else {
        WHILE {
            Py_ssize_t count = 0;
            npy_DTYPE0 asum = 0;
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                if (!bn_isnan(ai)) {
                    asum += ai;
                    count++;
                }
            }
            if (count > ddof) {
                const npy_DTYPE0 amean = asum / count;
                asum = 0;
                FOR {
                    npy_DTYPE0 ai = AI(DTYPE0);
                    if (!bn_isnan(ai)) {
                        ai -= amean;
                        asum += ai * ai;
                    }
                }
                asum = FUNC(asum / (count - ddof));
            } else {
                asum = BN_NAN_DTYPE0;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
REDUCE_ALL(NAME, DTYPE0) {
    npy_DTYPE1 out;
    Py_ssize_t size = 0;
    npy_DTYPE1 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            asum += AI(DTYPE0);
        }
        size += LENGTH;
        NEXT
    }
    if (size > ddof) {
        const npy_DTYPE1 amean = asum / size;
        asum = 0;
        RESET
        WHILE {
            FOR {
                const npy_DTYPE1 ai = AI(DTYPE0) - amean;
                asum += ai * ai;
            }
            NEXT
        }
        out = FUNC(asum / (size - ddof));
    } else {
        out = BN_NAN_DTYPE1;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(out);
}

REDUCE_ONE(NAME, DTYPE0) {
    INIT_ONE(DTYPE1, DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    const npy_DTYPE1 length_inv = 1.0 / LENGTH;
    const npy_DTYPE1 length_ddof_inv = 1.0 / (LENGTH - ddof);
    if (LENGTH == 0) {
        FILL_Y(BN_NAN_DTYPE1)
    } else {
        WHILE {
            npy_DTYPE1 asum = 0;
            FOR {
                asum += AI(DTYPE0);
            }
            if (LENGTH > ddof) {
                const npy_DTYPE1 amean = asum * length_inv;
                asum = 0;
                FOR {
                    const npy_DTYPE1 ai = AI(DTYPE0) - amean;
                    asum += ai * ai;
                }
                asum = FUNC(asum * length_ddof_inv);
            } else {
                asum = BN_NAN_DTYPE1;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

REDUCE_MAIN(NAME, 1)
/* repeat end */

/* nanmin, nanmax -------------------------------------------------------- */

/* repeat = {'NAME':      ['nanmin',         'nanmax'],
             'INTCOMP':   ['<',              '>'],
             'COMPARE':   ['<=',             '>='],
             'BIG_FLOAT': ['BN_INFINITY',    '-BN_INFINITY'],
             'BIG_INT':   ['NPY_MAX_DTYPE0', 'NPY_MIN_DTYPE0']} */
/* dtype = [['float64', '16'], ['float32', '32']] */
BN_OPT_3
REDUCE_ALL(NAME, DTYPE0) {
    npy_DTYPE0 extreme = BIG_FLOAT;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.NAME raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    npy_bool is_allnan = 1;
    if (REDUCE_CONTIGUOUS) {
        const npy_intp LOOP_SIZE = DTYPE1;
        // Allow vectorization by keeping an array of extremums
        npy_DTYPE0 extremes[DTYPE1];
        for (npy_intp i = 0; i < LOOP_SIZE; i++) {
            extremes[i] = BIG_FLOAT;
        }
        const npy_intp loops = LENGTH / LOOP_SIZE;
        const npy_intp residual = LENGTH % LOOP_SIZE;
        WHILE {
            const npy_DTYPE0 *pa = PA(DTYPE0);
            npy_intp          i = 0;
            for (; i < loops && is_allnan; i++) {
                for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                    const npy_DTYPE0 ai = pa[i * LOOP_SIZE + j];
                    // We only need <=/>= until we identify our first non-nan value
                    if (ai COMPARE extremes[j]) {
                        is_allnan = 0;
                        extremes[j] = ai;
                    }
                }
            }
            for (; i < loops; i++) {
                // Fully vectorize since the result cannot be NaN now
                for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                    const npy_DTYPE0         ai = pa[i * LOOP_SIZE + j];
                    extremes[j] = ai INTCOMP extremes[j] ? ai : extremes[j];
                }
            }
            for (npy_intp j = 0; j < residual; j++) {
                const npy_DTYPE0 ai = pa[loops * LOOP_SIZE + j];
                if (ai COMPARE extremes[j]) {
                    is_allnan = 0;
                    extremes[j] = ai;
                }
            }
            NEXT
        }
        // Reduce our array to find the global max/min
        for (npy_intp j = 0; j < LOOP_SIZE; j++) {
            if (extremes[j] COMPARE extreme) {
                extreme = extremes[j];
            }
        }
    } else {
        WHILE {
            const npy_DTYPE0 *pa = PA(DTYPE0);
            FOR {
                const npy_DTYPE0 ai = pa[it.i * it.stride];
                if (ai COMPARE extreme) {
                    extreme = ai;
                    is_allnan = 0;
                }
            }
            NEXT
        }
    }
    if (is_allnan) {
        extreme = BN_NAN_DTYPE0;
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(extreme);
}

BN_OPT_3
REDUCE_ONE(NAME, DTYPE0) {
    npy_DTYPE0 extreme;
    npy_bool   is_allnan;
    INIT_ONE(DTYPE0, DTYPE0)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.NAME raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    if (ONE_TRANSPOSE(npy_DTYPE0)) {
        npy_intp    LOOP_SIZE = it.nits;
        npy_DTYPE0 *extremes = malloc(LOOP_SIZE * sizeof(npy_DTYPE0));
        npy_bool *  allnans = malloc(LOOP_SIZE * sizeof(npy_bool));
        for (npy_intp i = 0; i < LOOP_SIZE; i++) {
            extremes[i] = BIG_FLOAT;
            allnans[i] = 1;
        }

        const npy_DTYPE0 *pa = PA(DTYPE0);
        npy_intp          i = 0;
        for (; i < LENGTH; i++) {
            for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                const npy_DTYPE0 ai = pa[i * LOOP_SIZE + j];
                if (ai COMPARE extremes[j]) {
                    allnans[j] = 0;
                    extremes[j] = ai;
                }
            }
            if (i % 8 == 0) {
                npy_intp nan_count = 0;
                for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                    nan_count += allnans[j];
                }
                if (nan_count == 0) {
                    break;
                }
            }
        }
        for (; i < LENGTH; i++) {
            for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                const npy_DTYPE0         ai = pa[i * LOOP_SIZE + j];
                extremes[j] = ai INTCOMP extremes[j] ? ai : extremes[j];
            }
        }
        for (npy_intp k = 0; k < LOOP_SIZE; k++) {
            if (allnans[k]) {
                py[k] = BN_NAN_DTYPE0;
            } else {
                py[k] = extremes[i];
            }
        }
        free(extremes);
        free(allnans);
    } else if (ONE_CONTIGUOUS) {
        const npy_intp LOOP_SIZE = DTYPE1;
        npy_DTYPE0     extremes[DTYPE1];
        const npy_intp loops = LENGTH / LOOP_SIZE;
        const npy_intp residual = LENGTH % LOOP_SIZE;
        WHILE {
            const npy_DTYPE0 *pa = PA(DTYPE0);

            for (npy_intp i = 0; i < LOOP_SIZE; i++) {
                extremes[i] = BIG_FLOAT;
            }
            is_allnan = 1;
            extreme = BIG_FLOAT;

            npy_intp i = 0;
            for (; i < loops && is_allnan; i++) {
                for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                    const npy_DTYPE0 ai = pa[i * LOOP_SIZE + j];
                    if (ai COMPARE extremes[j]) {
                        is_allnan = 0;
                        extremes[j] = ai;
                    }
                }
            }
            for (; i < loops; i++) {
                for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                    const npy_DTYPE0         ai = pa[i * LOOP_SIZE + j];
                    extremes[j] = ai INTCOMP extremes[j] ? ai : extremes[j];
                }
            }
            for (npy_intp j = 0; j < residual; j++) {
                const npy_DTYPE0 ai = pa[i * LOOP_SIZE + j];
                if (ai COMPARE extremes[j]) {
                    is_allnan = 0;
                    extremes[j] = ai;
                }
            }
            for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                if (extremes[j] COMPARE extreme) {
                    extreme = extremes[j];
                }
            }

            if (is_allnan) {
                extreme = BN_NAN_DTYPE0;
            }
            YPP = extreme;
            NEXT
        }
    } else {
        WHILE {
            extreme = BIG_FLOAT;
            is_allnan = 1;
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                if (ai COMPARE extreme) {
                    extreme = ai;
                    is_allnan = 0;
                }
            }
            if (is_allnan) {
                extreme = BN_NAN_DTYPE0;
            }
            YPP = extreme;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64'], ['int32']] */
BN_OPT_3
REDUCE_ALL(NAME, DTYPE0) {
    npy_DTYPE0 extreme = BIG_INT;
    INIT_ALL
    if (SIZE == 0) {
        VALUE_ERR("numpy.NAME raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    if (REDUCE_CONTIGUOUS) {
        WHILE {
            const npy_DTYPE0 *pa = PA(DTYPE0);
            FOR {
                const npy_DTYPE0     ai = pa[it.i];
                extreme = ai INTCOMP extreme ? ai : extreme;
            }
            NEXT
        }
    } else {
        WHILE {
            const npy_DTYPE0 *pa = PA(DTYPE0);
            FOR {
                const npy_DTYPE0     ai = pa[it.i * it.stride];
                extreme = ai INTCOMP extreme ? ai : extreme;
            }
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return PyLong_FromLongLong(extreme);
}

BN_OPT_3
REDUCE_ONE(NAME, DTYPE0) {
    npy_DTYPE0 extreme;
    INIT_ONE(DTYPE0, DTYPE0)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.NAME raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    if (ONE_TRANSPOSE(npy_DTYPE0)) {
        npy_intp    LOOP_SIZE = it.nits;
        npy_DTYPE0 *extremes = malloc(LOOP_SIZE * sizeof(npy_DTYPE0));
        for (npy_intp i = 0; i < LOOP_SIZE; i++) {
            extremes[i] = BIG_INT;
        }

        const npy_DTYPE0 *pa = PA(DTYPE0);
        for (npy_intp i = 0; i < LENGTH; i++) {
            for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                const npy_DTYPE0         ai = pa[i * LOOP_SIZE + j];
                extremes[j] = ai INTCOMP extremes[j] ? ai : extremes[j];
            }
        }
        for (npy_intp i = 0; i < LOOP_SIZE; i++) {
            py[i] = extremes[i];
        }
        free(extremes);
    } else if (ONE_CONTIGUOUS) {
        WHILE {
            extreme = BIG_INT;
            const npy_DTYPE0 *pa = PA(DTYPE0);
            FOR {
                const npy_DTYPE0     ai = pa[it.i];
                extreme = ai INTCOMP extreme ? ai : extreme;
            }
            YPP = extreme;
            NEXT
        }
    } else {
        WHILE {
            extreme = BIG_INT;
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                if (ai INTCOMP extreme) {
                    extreme = ai;
                }
            }
            YPP = extreme;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

REDUCE_MAIN(NAME, 0)
/* repeat end */

/* nanargmin, nanargmax -------------------------------------------------- */

/* repeat = {'NAME':      ['nanargmin',      'nanargmax'],
             'COMPARE':   ['<=',             '>='],
             'INTCOMP':   ['<',              '>'],
             'BIG_FLOAT': ['BN_INFINITY',    '-BN_INFINITY'],
             'BIG_INT':   ['NPY_MAX_DTYPE0', 'NPY_MIN_DTYPE0']} */
/* dtype = [['float64'], ['float32']] */
REDUCE_ALL(NAME, DTYPE0) {
    npy_DTYPE0 extreme = BIG_FLOAT;
    npy_bool   is_allnan = 1;
    Py_ssize_t idx = 0;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.NAME raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    FOR_REVERSE {
        const npy_DTYPE0 ai = AI(DTYPE0);
        if (ai COMPARE extreme) {
            extreme = ai;
            is_allnan = 0;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    if (is_allnan) {
        VALUE_ERR("All-NaN slice encountered");
        return NULL;
    } else {
        return PyLong_FromLongLong(idx);
    }
}

REDUCE_ONE(NAME, DTYPE0) {
    int err_code = 0;
    INIT_ONE(INTP, intp)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.NAME raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        npy_DTYPE0 extreme = BIG_FLOAT;
        npy_bool   is_allnan = 1;
        npy_intp   idx = 0;
        FOR_REVERSE {
            const npy_DTYPE0 ai = AI(DTYPE0);
            if (ai COMPARE extreme) {
                extreme = ai;
                is_allnan = 0;
                idx = INDEX;
            }
        }
        if (is_allnan == 0) {
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
/* dtype end */

/* dtype = [['int64', 'intp'], ['int32', 'intp']] */
BN_OPT_3
REDUCE_ALL(NAME, DTYPE0) {
    npy_DTYPE1 idx = 0;
    npy_DTYPE0 extreme = BIG_INT;
    INIT_ALL_RAVEL
    if (SIZE == 0) {
        DECREF_INIT_ALL_RAVEL
        VALUE_ERR("numpy.NAME raises on a.size==0 and axis=None; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    const npy_DTYPE0 *pa = PA(DTYPE0);
    FOR {
        const npy_DTYPE0 ai = SI(pa);
        if (ai INTCOMP extreme) {
            extreme = ai;
            idx = INDEX;
        }
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyLong_FromLongLong(idx);
}

BN_OPT_3
REDUCE_ONE(NAME, DTYPE0) {
    INIT_ONE(DTYPE1, DTYPE1)
    if (LENGTH == 0) {
        VALUE_ERR("numpy.NAME raises on a.shape[axis]==0; "
                  "So Bottleneck too.");
        return NULL;
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        npy_DTYPE0        extreme = BIG_INT;
        npy_DTYPE1        idx = 0;
        const npy_DTYPE0 *pa = PA(DTYPE0);
        FOR {
            const npy_DTYPE0 ai = SI(pa);
            if (ai INTCOMP extreme) {
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
/* dtype end */

REDUCE_MAIN(NAME, 0)
/* repeat end */

/* ss ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
REDUCE_ALL(ss, DTYPE0) {
    npy_DTYPE0 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            const npy_DTYPE0 ai = AI(DTYPE0);
            asum += ai * ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}

REDUCE_ONE(ss, DTYPE0) {
    INIT_ONE(DTYPE0, DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    } else {
        WHILE {
            npy_DTYPE0 asum = 0;
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                asum += ai * ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64', 'int64'], ['int32', 'long']] */
BN_OPT_3
REDUCE_ALL(ss, DTYPE0) {
    npy_DTYPE1 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    if (REDUCE_CONTIGUOUS) {
        const npy_intp    count = it.nits * it.length;
        const npy_DTYPE0 *pa = PA(DTYPE0);
        for (npy_intp i = 0; i < count; i++) {
            const npy_DTYPE0 ai = pa[i];
            asum += ai * ai;
        }
    } else {
        WHILE {
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                asum += ai * ai;
            }
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return PyLong_FromLongLong(asum);
}

BN_OPT_3
REDUCE_ONE(ss, DTYPE0) {
    INIT_ONE(DTYPE1, DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    } else {
        WHILE {
            npy_DTYPE1 asum = 0;
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                asum += ai * ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

REDUCE_MAIN(ss, 0)

/* median, nanmedian MACROS ---------------------------------------------- */

#define B(dtype, i) buffer[i] /* used by PARTITION */

#define EVEN_ODD(dtype, N)                 \
    if (N % 2 == 0) {                      \
        npy_##dtype amax = B(dtype, 0);    \
        for (npy_intp i = 1; i < k; i++) { \
            ai = B(dtype, i);              \
            if (ai > amax) amax = ai;      \
        }                                  \
        med = 0.5 * (B(dtype, k) + amax);  \
    } else {                               \
        med = B(dtype, k);                 \
    }

#define MEDIAN(dtype)                       \
    npy_intp    j, l, r, k;                 \
    npy_##dtype ai;                         \
    l = 0;                                  \
    for (npy_intp i = 0; i < LENGTH; i++) { \
        ai = AX(dtype, i);                  \
        if (!bn_isnan(ai)) {                \
            B(dtype, l++) = ai;             \
        } else {                            \
            break;                          \
        }                                   \
    }                                       \
    if (l != LENGTH) {                      \
        med = BN_NAN_##dtype;               \
        goto done;                          \
    }                                       \
    k = LENGTH >> 1;                        \
    l = 0;                                  \
    r = LENGTH - 1;                         \
    PARTITION(dtype)                        \
    EVEN_ODD(dtype, LENGTH)

#define MEDIAN_INT(dtype)                       \
    npy_intp           j, l, r, k;              \
    npy_##dtype        ai;                      \
    const npy_##dtype *pa = PA(dtype);          \
    if (it.stride == 1) {                       \
        for (npy_intp i = 0; i < LENGTH; i++) { \
            B(dtype, i) = pa[i];                \
        }                                       \
    } else {                                    \
        for (npy_intp i = 0; i < LENGTH; i++) { \
            B(dtype, i) = SX(pa, i);            \
        }                                       \
    }                                           \
    k = LENGTH >> 1;                            \
    l = 0;                                      \
    r = LENGTH - 1;                             \
    PARTITION(dtype)                            \
    EVEN_ODD(dtype, LENGTH)

#define NANMEDIAN(dtype)                    \
    npy_intp    j, l, r, k, n;              \
    npy_##dtype ai;                         \
    l = 0;                                  \
    for (npy_intp i = 0; i < LENGTH; i++) { \
        ai = AX(dtype, i);                  \
        if (!bn_isnan(ai)) {                \
            B(dtype, l++) = ai;             \
        }                                   \
    }                                       \
    n = l;                                  \
    k = n >> 1;                             \
    l = 0;                                  \
    r = n - 1;                              \
    if (n == 0) {                           \
        med = BN_NAN_##dtype;               \
        goto done;                          \
    }                                       \
    PARTITION(dtype)                        \
    EVEN_ODD(dtype, n)

#define BUFFER_NEW(dtype, length) \
    npy_##dtype *buffer = calloc(length, sizeof(npy_##dtype));

#define BUFFER_DELETE free(buffer);

/* median, nanmedian ----------------------------------------------------- */

/* repeat = {'NAME': ['median', 'nanmedian'],
             'FUNC': ['MEDIAN', 'NANMEDIAN']} */
/* dtype = [['float64', 'float64'], ['float32', 'float32']] */

REDUCE_ALL(NAME, DTYPE0) {
    npy_DTYPE1 med = BN_NAN;
    INIT_ALL_RAVEL_ANY_ORDER
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        med = BN_NAN_DTYPE1;
    } else {
        BUFFER_NEW(DTYPE0, LENGTH)
        FUNC(DTYPE0)
    done:
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyFloat_FromDouble(med);
}

REDUCE_ONE(NAME, DTYPE0) {
    INIT_ONE(DTYPE1, DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN_DTYPE1)
    } else {
        npy_DTYPE1 med = BN_NAN;
        BUFFER_NEW(DTYPE0, LENGTH)
        WHILE {
            FUNC(DTYPE0)
        done:
            YPP = med;
            NEXT
        }
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */
/* repeat end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
BN_OPT_3
REDUCE_ALL(median, DTYPE0) {
    npy_DTYPE1 med;
    INIT_ALL_RAVEL_ANY_ORDER
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        med = BN_NAN_DTYPE1;
    } else {
        BUFFER_NEW(DTYPE0, LENGTH)
        MEDIAN_INT(DTYPE0)
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    DECREF_INIT_ALL_RAVEL
    return PyFloat_FromDouble(med);
}

BN_OPT_3
REDUCE_ONE(median, DTYPE0) {
    INIT_ONE(DTYPE1, DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(BN_NAN_DTYPE1)
    } else {
        npy_DTYPE1 med;
        BUFFER_NEW(DTYPE0, LENGTH)
        WHILE {
            MEDIAN_INT(DTYPE0)
            YPP = med;
            NEXT
        }
        BUFFER_DELETE
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

REDUCE_MAIN(median, 0)

static PyObject *
nanmedian(PyObject *self, PyObject *args, PyObject *kwds) {
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

/* SSE2 anynan/allnan ---------------------------------------------------- */

#if HAVE_SSE2

int inline sse2_allnan_float64(const npy_float64 *pa) {
    __m128d values = _mm_load_pd(pa);
    __m128d result = _mm_cmpneq_pd(values, values);
    /* If a value is NaN, a bit gets set to 1. As 2 64-bit floats fit in a
       128-bit vector, 11 = 3 */
    return _mm_movemask_pd(result) != 3;
}

int inline sse2_allnan_float32(const npy_float32 *pa) {
    __m128 values = _mm_load_ps(pa);
    __m128 result = _mm_cmpneq_ps(values, values);
    /* If a value is NaN, a bit gets set to 1. As 4 32-bit floats fit in a
       128-bit vector, 1111 = 15 */
    return _mm_movemask_ps(result) != 15;
}

int inline sse2_anynan_float64(const npy_float64 *pa) {
    __m128d values = _mm_load_pd(pa);
    __m128d result = _mm_cmpneq_pd(values, values);
    return _mm_movemask_pd(result) > 0;
}

int inline sse2_anynan_float32(const npy_float32 *pa) {
    __m128 values = _mm_load_ps(pa);
    __m128 result = _mm_cmpneq_ps(values, values);
    return _mm_movemask_ps(result) > 0;
}

#endif

/* anynan ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
BN_OPT_3
REDUCE_ALL(anynan, DTYPE0) {
    npy_bool f = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    if (REDUCE_CONTIGUOUS) {
        const int         VECTOR_ALIGN = 16;
        const npy_DTYPE0 *pa = PA(DTYPE0);
        const npy_intp    count = it.nits * it.length;
        npy_intp          vector_offset = (((npy_intp)pa) % VECTOR_ALIGN);
        vector_offset = vector_offset > 0 ? VECTOR_ALIGN - vector_offset : 0;
        vector_offset /= sizeof(npy_DTYPE0);
        vector_offset = vector_offset <= count ? vector_offset : count;

        const npy_intp LOOP_SIZE = 16 / sizeof(npy_DTYPE0);
        const npy_intp loop_count = (count - vector_offset) / LOOP_SIZE;
        const npy_intp residual = (count - vector_offset) % LOOP_SIZE;

        for (npy_intp i = 0; (i < vector_offset) && (f == 0); i++) {
            const npy_DTYPE0 ai = pa[i];
            if (bn_isnan(ai)) {
                f = 1;
            }
        }

#if HAVE_SSE2
        for (npy_intp i = 0; (i < loop_count) && (f == 0); i++) {
            f = sse2_anynan_DTYPE0(&pa[vector_offset + i * LOOP_SIZE]);
        }
#else
        npy_bool *f_arr = malloc(LOOP_SIZE * sizeof(npy_bool));
        for (npy_intp i = 0; (i < loop_count) && (f == 0); i++) {
            for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                f_arr[j] = bn_isnan(pa[vector_offset + i * LOOP_SIZE + j]);
            }

            for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                f += f_arr[j];
            }
        }
        free(f_arr);
#endif

        for (npy_intp j = 0; (j < residual) && (f == 0); j++) {
            const npy_DTYPE0 ai = pa[vector_offset + loop_count * LOOP_SIZE + j];
            if (bn_isnan(ai)) {
                f = 1;
            }
        }
    } else {
        WHILE {
            const npy_DTYPE0 *pa = PA(DTYPE0);
            FOR {
                const npy_DTYPE0 ai = pa[it.i * it.stride];
                if (bn_isnan(ai)) {
                    f = 1;
                    goto done;
                }
            }
            NEXT
        }
    }
done:
    BN_END_ALLOW_THREADS
    if (f) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

BN_OPT_3
REDUCE_ONE(anynan, DTYPE0) {
    npy_bool f;
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    } else {
        if (ONE_CONTIGUOUS) {
            WHILE {
                f = 0;
                const int         VECTOR_ALIGN = 16;
                const npy_DTYPE0 *pa = PA(DTYPE0);
                const npy_intp    count = LENGTH;
                npy_intp          vector_offset = (((npy_intp)pa) % VECTOR_ALIGN);
                vector_offset = vector_offset > 0 ? VECTOR_ALIGN - vector_offset : 0;
                vector_offset /= sizeof(npy_DTYPE0);
                vector_offset = vector_offset <= count ? vector_offset : count;

                const npy_intp LOOP_SIZE = 16 / sizeof(npy_DTYPE0);
                const npy_intp loop_count = (count - vector_offset) / LOOP_SIZE;
                const npy_intp residual = (count - vector_offset) % LOOP_SIZE;

                for (npy_intp i = 0; (i < vector_offset) && (f == 0); i++) {
                    const npy_DTYPE0 ai = pa[i];
                    if (bn_isnan(ai)) {
                        f = 1;
                    }
                }

#if HAVE_SSE2
                for (npy_intp i = 0; (i < loop_count) && (f == 0); i++) {
                    f = sse2_anynan_DTYPE0(&pa[vector_offset + i * LOOP_SIZE]);
                }
#else
                for (npy_intp i = 0; (i < loop_count) && (f == 0); i++) {
                    for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                        ai = pa[vector_offset + i * LOOP_SIZE + j];
                        f += bn_isnan(ai);
                    }
                }
#endif
                for (npy_intp j = 0; (j < residual) && (f == 0); j++) {
                    const npy_DTYPE0 ai = pa[vector_offset + loop_count * LOOP_SIZE + j];
                    if (bn_isnan(ai)) {
                        f = 1;
                    }
                }
                YPP = f;
                NEXT
            }
        } else if (ONE_TRANSPOSE(npy_DTYPE0)) {
            const npy_intp    LOOP_SIZE = it.nits;
            const npy_DTYPE0 *pa = PA(DTYPE0);

            for (npy_intp i = 0; i < LOOP_SIZE; i++) {
                py[i] = 0;
            }

            const npy_intp loop_count = LENGTH / 16;
            const npy_intp residual = LENGTH % 16;
            npy_intp       min_k = 0;

            for (npy_intp i = 0; i < residual; i++) {
                for (npy_intp k = min_k; k < LOOP_SIZE; k++) {
                    const npy_DTYPE0 ai = pa[i * LOOP_SIZE + k];
                    py[k] = bn_isnan(ai) ? 1 : py[k];
                    if ((k == min_k) && py[k]) {
                        min_k++;
                    }
                }
            }

            for (npy_intp i = 0; (i < loop_count) && (min_k < LOOP_SIZE - 1); i++) {
                for (npy_intp j = 0; j < 16; j++) {
                    for (npy_intp k = min_k; k < LOOP_SIZE; k++) {
                        const npy_DTYPE0 ai = pa[(i * 16 + j + residual) * LOOP_SIZE + k];
                        py[k] = bn_isnan(ai) ? 1 : py[k];
                    }
                }
                for (npy_intp k = min_k; k < LOOP_SIZE; k++) {
                    if (py[k] == 0) {
                        min_k = k;
                        break;
                    } else if (k == LOOP_SIZE - 1) {
                        min_k = k;
                    }
                }
                if (min_k == LOOP_SIZE - 1) {
                    break;
                }
            }
        } else {
            WHILE {
                f = 0;
                const npy_DTYPE0 *pa = PA(DTYPE0);
                FOR {
                    const npy_DTYPE0 ai = SI(pa);
                    if (bn_isnan(ai)) {
                        f = 1;
                        break;
                    }
                }
                YPP = f;
                NEXT
            }
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64'], ['int32']] */
BN_OPT_3
REDUCE_ALL(anynan, DTYPE0) {
    Py_RETURN_FALSE;
}

BN_OPT_3
REDUCE_ONE(anynan, DTYPE0) {
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    FILL_Y(0);
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

REDUCE_MAIN(anynan, 0)

/* allnan ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
BN_OPT_3
REDUCE_ALL(allnan, DTYPE0) {
    int f = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    if (REDUCE_CONTIGUOUS) {
        const int         VECTOR_ALIGN = 16;
        const npy_DTYPE0 *pa = PA(DTYPE0);
        const npy_intp    count = it.nits * it.length;
        npy_intp          vector_offset = (((npy_intp)pa) % VECTOR_ALIGN);
        vector_offset = vector_offset > 0 ? VECTOR_ALIGN - vector_offset : 0;
        vector_offset /= sizeof(npy_DTYPE0);
        vector_offset = vector_offset <= count ? vector_offset : count;

        const npy_intp LOOP_SIZE = 16 / sizeof(npy_DTYPE0);
        const npy_intp loop_count = (count - vector_offset) / LOOP_SIZE;
        const npy_intp residual = (count - vector_offset) % LOOP_SIZE;

        for (npy_intp i = 0; (i < vector_offset) && (f == 0); i++) {
            const npy_DTYPE0 ai = pa[i];
            if (!bn_isnan(ai)) {
                f = 1;
            }
        }

#if HAVE_SSE2
        for (npy_intp i = 0; (i < loop_count) && (f == 0); i++) {
            f = sse2_allnan_DTYPE0(&pa[vector_offset + i * LOOP_SIZE]);
        }
#else
        npy_bool *f_arr = malloc(LOOP_SIZE * sizeof(npy_bool));
        for (npy_intp i = 0; (i < loop_count) && (f == 0); i++) {
            for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                f_arr[j] = !bn_isnan(pa[vector_offset + i * LOOP_SIZE + j]);
            }

            for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                f += f_arr[j];
            }
        }
        free(f_arr);
#endif

        for (npy_intp j = 0; (j < residual) && (f == 0); j++) {
            const npy_DTYPE0 ai = pa[vector_offset + loop_count * LOOP_SIZE + j];
            if (!bn_isnan(ai)) {
                f = 1;
            }
        }
    } else {
        WHILE {
            FOR {
                const npy_DTYPE0 ai = AI(DTYPE0);
                if (!bn_isnan(ai)) {
                    f = 1;
                    goto done;
                }
            }
            NEXT
        }
    }
done:
    BN_END_ALLOW_THREADS
    if (f) {
        Py_RETURN_FALSE;
    } else {
        Py_RETURN_TRUE;
    }
}

BN_OPT_3
REDUCE_ONE(allnan, DTYPE0) {
    npy_bool f;
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(1)
    } else {
        if (ONE_CONTIGUOUS) {
            WHILE {
                f = 0;
                const int         VECTOR_ALIGN = 16;
                const npy_DTYPE0 *pa = PA(DTYPE0);
                const npy_intp    count = LENGTH;
                npy_intp          vector_offset = (((npy_intp)pa) % VECTOR_ALIGN);
                vector_offset = vector_offset > 0 ? VECTOR_ALIGN - vector_offset : 0;
                vector_offset /= sizeof(npy_DTYPE0);
                vector_offset = vector_offset <= count ? vector_offset : count;

                const npy_intp LOOP_SIZE = 16 / sizeof(npy_DTYPE0);
                const npy_intp loop_count = (count - vector_offset) / LOOP_SIZE;
                const npy_intp residual = (count - vector_offset) % LOOP_SIZE;

                for (npy_intp i = 0; (i < vector_offset) && (f == 0); i++) {
                    const npy_DTYPE0 ai = pa[i];
                    if (!bn_isnan(ai)) {
                        f = 1;
                    }
                }

#if HAVE_SSE2
                for (npy_intp i = 0; (i < loop_count) && (f == 0); i++) {
                    f = sse2_allnan_DTYPE0(&pa[vector_offset + i * LOOP_SIZE]);
                }
#else
                for (npy_intp i = 0; (i < loop_count) && (f == 0); i++) {
                    for (npy_intp j = 0; j < LOOP_SIZE; j++) {
                        ai = pa[vector_offset + i * LOOP_SIZE + j];
                        f += !bn_isnan(ai);
                    }
                }
#endif
                for (npy_intp j = 0; (j < residual) && (f == 0); j++) {
                    const npy_DTYPE0 ai = pa[vector_offset + loop_count * LOOP_SIZE + j];
                    if (!bn_isnan(ai)) {
                        f = 1;
                    }
                }
                YPP = !f;
                NEXT
            }
        } else if (ONE_TRANSPOSE(npy_DTYPE0)) {
            const npy_intp    LOOP_SIZE = it.nits;
            const npy_DTYPE0 *pa = PA(DTYPE0);

            for (npy_intp i = 0; i < LOOP_SIZE; i++) {
                py[i] = 0;
            }

            const npy_intp loop_count = LENGTH / 16;
            const npy_intp residual = LENGTH % 16;
            npy_intp       min_k = 0;

            for (npy_intp i = 0; i < residual; i++) {
                for (npy_intp k = min_k; k < LOOP_SIZE; k++) {
                    const npy_DTYPE0 ai = pa[i * LOOP_SIZE + k];
                    py[k] = !bn_isnan(ai) ? 1 : py[k];
                    if ((k == min_k) && py[k]) {
                        min_k++;
                    }
                }
            }

            for (npy_intp i = 0; (i < loop_count) && (min_k < LOOP_SIZE - 1); i++) {
                for (npy_intp j = 0; j < 16; j++) {
                    for (npy_intp k = min_k; k < LOOP_SIZE; k++) {
                        const npy_DTYPE0 ai = pa[(i * 16 + j + residual) * LOOP_SIZE + k];
                        py[k] = !bn_isnan(ai) ? 1 : py[k];
                    }
                }
                for (npy_intp k = min_k; k < LOOP_SIZE; k++) {
                    if (py[k] == 0) {
                        min_k = k;
                        break;
                    } else if (k == LOOP_SIZE - 1) {
                        min_k = k;
                    }
                }
                if (min_k == LOOP_SIZE - 1) {
                    break;
                }
            }

            for (npy_intp k = 0; k < LOOP_SIZE; k++) {
                py[k] = !py[k];
            }
        } else {
            WHILE {
                f = 0;
                const npy_DTYPE0 *pa = PA(DTYPE0);
                FOR {
                    const npy_DTYPE0 ai = SI(pa);
                    if (!bn_isnan(ai)) {
                        f = 1;
                        break;
                    }
                }
                YPP = !f;
                NEXT
            }
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64'], ['int32']] */
BN_OPT_3
REDUCE_ALL(allnan, DTYPE0) {
    if (PyArray_SIZE(a) == 0) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

BN_OPT_3
REDUCE_ONE(allnan, DTYPE0) {
    INIT_ONE(BOOL, uint8)
    BN_BEGIN_ALLOW_THREADS
    if (SIZE == 0) {
        FILL_Y(1);
    } else {
        FILL_Y(0);
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

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

static inline int
parse_args(PyObject * args,
           PyObject * kwds,
           int        has_ddof,
           PyObject **a,
           PyObject **axis,
           PyObject **ddof) {
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int       nkwds_found = 0;
        PyObject *tmp;
        switch (nargs) {
            case 2:
                if (has_ddof) {
                    *axis = PyTuple_GET_ITEM(args, 1);
                } else {
                    TYPE_ERR("wrong number of arguments");
                    return 0;
                }
            case 1:
                *a = PyTuple_GET_ITEM(args, 0);
            case 0:
                break;
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
    } else {
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
reducer(char *    name,
        PyObject *args,
        PyObject *kwds,
        fall_t    fall_float64,
        fall_t    fall_float32,
        fall_t    fall_int64,
        fall_t    fall_int32,
        fone_t    fone_float64,
        fone_t    fone_float32,
        fone_t    fone_int64,
        fone_t    fone_int32,
        int       has_ddof) {
    int ndim;
    int axis = 0; /* initialize to avoid compiler error */
    int dtype;
    int ddof;
    int reduce_all = 0;

    PyArrayObject *a;
    PyObject *     y;

    PyObject *a_obj = NULL;
    PyObject *axis_obj = Py_None;
    PyObject *ddof_obj = NULL;

    if (!parse_args(args, kwds, has_ddof, &a_obj, &axis_obj, &ddof_obj)) {
        return NULL;
    }

    /* convert to array if necessary */
    if (PyArray_Check(a_obj)) {
        a = (PyArrayObject *)a_obj;
        Py_INCREF(a);
    } else {
        a = (PyArrayObject *)PyArray_FROM_O(a_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if (PyArray_ISBYTESWAPPED(a)) {
        Py_DECREF(a);
        return slow(name, args, kwds);
    }

    /* does user want to reduce over all axes? */
    if (axis_obj == Py_None) {
        reduce_all = 1;
    } else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer or None");
            goto error;
        }
        ndim = PyArray_NDIM(a);
        if (axis < 0) {
            axis += ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds",
                             axis);
                goto error;
            }
        } else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            goto error;
        }
        if (ndim == 1) {
            reduce_all = 1;
        }
    }

    /* ddof */
    if (ddof_obj == NULL) {
        ddof = 0;
    } else {
        ddof = PyArray_PyIntAsInt(ddof_obj);
        if (error_converting(ddof)) {
            TYPE_ERR("`ddof` must be an integer");
            goto error;
        }
    }

    dtype = PyArray_TYPE(a);

    if (reduce_all == 1) {
        /* we are reducing the array along all axes */
        if (dtype == NPY_FLOAT64) {
            y = fall_float64(a, ddof);
        } else if (dtype == NPY_FLOAT32) {
            y = fall_float32(a, ddof);
        } else if (dtype == NPY_INT64) {
            y = fall_int64(a, ddof);
        } else if (dtype == NPY_INT32) {
            y = fall_int32(a, ddof);
        } else {
            y = slow(name, args, kwds);
        }
    } else {
        /* we are reducing an array with ndim > 1 over a single axis */
        if (dtype == NPY_FLOAT64) {
            y = fone_float64(a, axis, ddof);
        } else if (dtype == NPY_FLOAT32) {
            y = fone_float32(a, axis, ddof);
        } else if (dtype == NPY_INT64) {
            y = fone_int64(a, axis, ddof);
        } else if (dtype == NPY_INT32) {
            y = fone_int32(a, axis, ddof);
        } else {
            y = slow(name, args, kwds);
        }
    }

    Py_DECREF(a);

    return y;

error:
    Py_DECREF(a);
    return NULL;
}

/* docstrings ------------------------------------------------------------- */

// clang-format off

static char reduce_doc[] =
"Bottleneck functions that reduce the input array along a specified axis.";

static char nansum_doc[] =
/* MULTILINE STRING BEGIN
nansum(a, axis=None)

Sum of array elements along given axis treating NaNs as zero.

The data type (dtype) of the output is the same as the input. On 64-bit
operating systems, 32-bit input is NOT upcast to 64-bit accumulator and
return values.

Parameters
----------
a : array_like
    Array containing numbers whose sum is desired. If `a` is not an
    array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the sum is computed. The default (axis=None) is to
    compute the sum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`, with the specified axis removed.
    If `a` is a 0-d array, or if axis is None, a scalar is returned.

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
nanmean(a, axis=None)

Mean of array elements along given axis ignoring NaNs.

`float64` intermediate and return values are used for integer inputs.

Parameters
----------
a : array_like
    Array containing numbers whose mean is desired. If `a` is not an
    array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the means are computed. The default (axis=None) is to
    compute the mean of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`, with the specified axis removed.
    If `a` is a 0-d array, or if axis is None, a scalar is returned.
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
nanstd(a, axis=None, ddof=0)

Standard deviation along the specified axis, ignoring NaNs.

`float64` intermediate and return values are used for integer inputs.

Instead of a faster one-pass algorithm, a more stable two-pass algorithm
is used.

An example of a one-pass algorithm:

    >>> np.sqrt((a*a).mean() - a.mean()**2)

An example of a two-pass algorithm:

    >>> np.sqrt(((a - a.mean())**2).mean())

Note in the two-pass algorithm the mean must be found (first pass) before
the squared deviation (second pass) can be found.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
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
    An array with the same shape as `a`, with the specified axis removed.
    If `a` is a 0-d array, or if axis is None, a scalar is returned.
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
nanvar(a, axis=None, ddof=0)

Variance along the specified axis, ignoring NaNs.

`float64` intermediate and return values are used for integer inputs.

Instead of a faster one-pass algorithm, a more stable two-pass algorithm
is used.

An example of a one-pass algorithm:

    >>> (a*a).mean() - a.mean()**2

An example of a two-pass algorithm:

    >>> ((a - a.mean())**2).mean()

Note in the two-pass algorithm the mean must be found (first pass) before
the squared deviation (second pass) can be found.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
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
    An array with the same shape as `a`, with the specified axis
    removed. If `a` is a 0-d array, or if axis is None, a scalar is
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
nanmin(a, axis=None)

Minimum values along specified axis, ignoring NaNs.

When all-NaN slices are encountered, NaN is returned for that slice.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the minimum is computed. The default (axis=None) is
    to compute the minimum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`, with the specified axis removed.
    If `a` is a 0-d array, or if axis is None, a scalar is returned. The
    same dtype as `a` is returned.

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
nanmax(a, axis=None)

Maximum values along specified axis, ignoring NaNs.

When all-NaN slices are encountered, NaN is returned for that slice.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the maximum is computed. The default (axis=None) is
    to compute the maximum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`, with the specified axis removed.
    If `a` is a 0-d array, or if axis is None, a scalar is returned. The
    same dtype as `a` is returned.

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

static char nanargmin_doc[] =
/* MULTILINE STRING BEGIN
nanargmin(a, axis=None)

Indices of the minimum values along an axis, ignoring NaNs.

For all-NaN slices ``ValueError`` is raised. Unlike NumPy, the results
can be trusted if a slice contains only NaNs and Infs.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which to operate. By default (axis=None) flattened input
    is used.

See also
--------
bottleneck.nanargmax: Indices of the maximum values along an axis.
bottleneck.nanmin: Minimum values along specified axis, ignoring NaNs.

Returns
-------
index_array : ndarray
    An array of indices or a single index value.

Examples
--------
>>> a = np.array([[np.nan, 4], [2, 3]])
>>> bn.nanargmin(a)
2
>>> a.flat[2]
2.0
>>> bn.nanargmin(a, axis=0)
array([1, 1])
>>> bn.nanargmin(a, axis=1)
array([1, 0])

MULTILINE STRING END */

static char nanargmax_doc[] =
/* MULTILINE STRING BEGIN
nanargmax(a, axis=None)

Indices of the maximum values along an axis, ignoring NaNs.

For all-NaN slices ``ValueError`` is raised. Unlike NumPy, the results
can be trusted if a slice contains only NaNs and Infs.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which to operate. By default (axis=None) flattened input
    is used.

See also
--------
bottleneck.nanargmin: Indices of the minimum values along an axis.
bottleneck.nanmax: Maximum values along specified axis, ignoring NaNs.

Returns
-------
index_array : ndarray
    An array of indices or a single index value.

Examples
--------
>>> a = np.array([[np.nan, 4], [2, 3]])
>>> bn.nanargmax(a)
1
>>> a.flat[1]
4.0
>>> bn.nanargmax(a, axis=0)
array([1, 0])
>>> bn.nanargmax(a, axis=1)
array([1, 1])

MULTILINE STRING END */

static char ss_doc[] =
/* MULTILINE STRING BEGIN
ss(a, axis=None)

Sum of the square of each element along the specified axis.

Parameters
----------
a : array_like
    Array whose sum of squares is desired. If `a` is not an array, a
    conversion is attempted.
axis : {int, None}, optional
    Axis along which the sum of squares is computed. The default
    (axis=None) is to sum the squares of the flattened array.

Returns
-------
y : ndarray
    The sum of a**2 along the given axis.

Examples
--------
>>> a = np.array([1., 2., 5.])
>>> bn.ss(a)
30.0

And calculating along an axis:

>>> b = np.array([[1., 2., 5.], [2., 5., 6.]])
>>> bn.ss(b, axis=1)
array([ 30., 65.])

MULTILINE STRING END */

static char median_doc[] =
/* MULTILINE STRING BEGIN
median(a, axis=None)

Median of array elements along given axis.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the median is computed. The default (axis=None) is to
    compute the median of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`, except that the specified axis
    has been removed. If `a` is a 0d array, or if axis is None, a scalar
    is returned. `float64` return values are used for integer inputs. NaN
    is returned for a slice that contains one or more NaNs.

See also
--------
bottleneck.nanmedian: Median along specified axis ignoring NaNs.

Examples
--------
>>> a = np.array([[10, 7, 4], [3, 2, 1]])
>>> bn.median(a)
    3.5
>>> bn.median(a, axis=0)
    array([ 6.5,  4.5,  2.5])
>>> bn.median(a, axis=1)
    array([ 7.,  2.])

MULTILINE STRING END */

static char nanmedian_doc[] =
/* MULTILINE STRING BEGIN
nanmedian(a, axis=None)

Median of array elements along given axis ignoring NaNs.

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the median is computed. The default (axis=None) is to
    compute the median of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`, except that the specified axis
    has been removed. If `a` is a 0d array, or if axis is None, a scalar
    is returned. `float64` return values are used for integer inputs.

See also
--------
bottleneck.median: Median along specified axis.

Examples
--------
>>> a = np.array([[np.nan, 7, 4], [3, 2, 1]])
>>> a
array([[ nan,   7.,   4.],
       [  3.,   2.,   1.]])
>>> bn.nanmedian(a)
3.0
>> bn.nanmedian(a, axis=0)
array([ 3. ,  4.5,  2.5])
>> bn.nanmedian(a, axis=1)
array([ 5.5,  2. ])

MULTILINE STRING END */

static char anynan_doc[] =
/* MULTILINE STRING BEGIN
anynan(a, axis=None)

Test whether any array element along a given axis is NaN.

Returns the same output as np.isnan(a).any(axis)

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
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

static char allnan_doc[] =
/* MULTILINE STRING BEGIN
allnan(a, axis=None)

Test whether all array elements along a given axis are NaN.

Returns the same output as np.isnan(a).all(axis)

Note that allnan([]) is True to match np.isnan([]).all() and all([])

Parameters
----------
a : array_like
    Input array. If `a` is not an array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which NaNs are searched. The default (`axis` = ``None``)
    is to search for NaNs over a flattened input array.

Returns
-------
y : bool or ndarray
    A boolean or new `ndarray` is returned.

See also
--------
bottleneck.anynan: Test if any array element along given axis is NaN

Examples
--------
>>> bn.allnan(1)
False
>>> bn.allnan(np.nan)
True
>>> bn.allnan([1, np.nan])
False
>>> a = np.array([[1, np.nan], [1, np.nan]])
>>> bn.allnan(a)
False
>>> bn.allnan(a, axis=0)
array([False,  True], dtype=bool)

An empty array returns True:

>>> bn.allnan([])
True

which is similar to:

>>> all([])
True
>>> np.isnan([]).all()
True

MULTILINE STRING END */

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

// clang-format on

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
    reduce_def = {
        PyModuleDef_HEAD_INIT,
        "reduce",
        reduce_doc,
        -1,
        reduce_methods};
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    #define RETVAL m
PyInit_reduce(void)
#else
    #define RETVAL
initreduce(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&reduce_def);
#else
    PyObject *m = Py_InitModule3("reduce", reduce_methods, reduce_doc);
#endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
