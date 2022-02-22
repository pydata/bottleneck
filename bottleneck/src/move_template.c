// Copyright 2010-2019 Keith Goodman
// Copyright 2019 Bottleneck Developers
#include "bottleneck.h"
#include "iterators.h"
#include "move_median/move_median.h"

/*
   move_min, move_max, move_argmin, and move_argmax are based on
   the minimum on a sliding window algorithm by Richard Harter
   http://www.richardhartersworld.com/cri/2001/slidingmin.html
   Copyright Richard Harter 2009
   Released under a Simplified BSD license

   Adapted, expanded, and added NaN handling for Bottleneck:
   Copyright 2010, 2014, 2015, 2016 Keith Goodman
   Released under the Bottleneck license
*/

/* macros ---------------------------------------------------------------- */

#define INIT(dtype) \
    PyObject *y = PyArray_EMPTY(PyArray_NDIM(a), PyArray_SHAPE(a), dtype, 0); \
    iter2 it; \
    init_iter2(&it, a, y, axis);

/* low-level functions such as move_sum_float64 */
#define MOVE(name, dtype) \
    static PyObject * \
    name##_##dtype(PyArrayObject *a, \
                   int           window, \
                   int           min_count, \
                   int           axis, \
                   int           ddof)

/* top-level functions such as move_sum */
#define MOVE_MAIN(name, ddof) \
    static PyObject * \
    name(PyObject *self, PyObject *args, PyObject *kwds) \
    { \
        return mover(#name, \
                     args, \
                     kwds, \
                     name##_float64, \
                     name##_float32, \
                     name##_int64, \
                     name##_int32, \
                     ddof); \
    }

/* typedefs and prototypes ----------------------------------------------- */

/* used by move_min and move_max */
struct _pairs {
    double value;
    int death;
};
typedef struct _pairs pairs;

/* function pointer for functions passed to mover */
typedef PyObject *(*move_t)(PyArrayObject *, int, int, int, int);

static PyObject *
mover(char *name,
      PyObject *args,
      PyObject *kwds,
      move_t,
      move_t,
      move_t,
      move_t,
      int has_ddof);

/* move_sum -------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
MOVE(move_sum, DTYPE0) {
    Py_ssize_t count;
    npy_DTYPE0 asum, ai, aold;
    INIT(NPY_DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        asum = count = 0;
        WHILE0 {
            ai = AI(DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            YI(DTYPE0) = BN_NAN;
        }
        WHILE1 {
            ai = AI(DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            YI(DTYPE0) = count >= min_count ? asum : BN_NAN;
        }
        WHILE2 {
            ai = AI(DTYPE0);
            aold = AOLD(DTYPE0);
            if (ai == ai) {
                if (aold == aold) {
                    asum += ai - aold;
                } else {
                    asum += ai;
                    count++;
                }
            } else {
                if (aold == aold) {
                    asum -= aold;
                    count--;
                }
            }
            YI(DTYPE0) = count >= min_count ? asum : BN_NAN;
        }
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
MOVE(move_sum, DTYPE0) {
    npy_DTYPE1 asum;
    INIT(NPY_DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        asum = 0;
        WHILE0 {
            asum += AI(DTYPE0);
            YI(DTYPE1) = BN_NAN;
        }
        WHILE1 {
            asum += AI(DTYPE0);
            YI(DTYPE1) = asum;
        }
        WHILE2 {
            asum += AI(DTYPE0) - AOLD(DTYPE0);
            YI(DTYPE1) = asum;
        }
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


MOVE_MAIN(move_sum, 0)


/* move_mean -------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
MOVE(move_mean, DTYPE0) {
    Py_ssize_t count;
    npy_DTYPE0 asum, ai, aold, count_inv;
    INIT(NPY_DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        asum = count = 0;
        WHILE0 {
            ai = AI(DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            YI(DTYPE0) = BN_NAN;
        }
        WHILE1 {
            ai = AI(DTYPE0);
            if (ai == ai) {
                asum += ai;
                count += 1;
            }
            YI(DTYPE0) = count >= min_count ? asum / count : BN_NAN;
        }
        count_inv = 1.0 / count;
        WHILE2 {
            ai = AI(DTYPE0);
            aold = AOLD(DTYPE0);
            if (ai == ai) {
                if (aold == aold) {
                    asum += ai - aold;
                } else {
                    asum += ai;
                    count++;
                    count_inv = 1.0 / count;
                }
            } else {
                if (aold == aold) {
                    asum -= aold;
                    count--;
                    count_inv = 1.0 / count;
                }
            }
            YI(DTYPE0) = count >= min_count ? asum * count_inv : BN_NAN;
        }
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
MOVE(move_mean, DTYPE0) {
    npy_DTYPE1 asum, window_inv = 1.0 / window;
    INIT(NPY_DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        asum = 0;
        WHILE0 {
            asum += AI(DTYPE0);
            YI(DTYPE1) = BN_NAN;
        }
        WHILE1 {
            asum += AI(DTYPE0);
            *(npy_DTYPE1*)(it.py + it.i * it.ystride) = (npy_DTYPE1)asum / (it.i + 1);
            it.i++;
        }
        WHILE2 {
            asum += AI(DTYPE0) - AOLD(DTYPE0);
            YI(DTYPE1) = (npy_DTYPE1)asum * window_inv;
        }
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


MOVE_MAIN(move_mean, 0)


/* move_std, move_var ---------------------------------------------------- */

/* repeat = {'NAME': ['move_std', 'move_var'],
             'FUNC': ['sqrt',     '']} */
/* dtype = [['float64'], ['float32']] */
MOVE(NAME, DTYPE0) {
    Py_ssize_t count;
    npy_DTYPE0 delta, amean, assqdm, ai, aold, yi, count_inv, ddof_inv;
    INIT(NPY_DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        amean = assqdm = count = 0;
        WHILE0 {
            ai = AI(DTYPE0);
            if (ai == ai) {
                count += 1;
                delta = ai - amean;
                amean += delta / count;
                assqdm += delta * (ai - amean);
            }
            YI(DTYPE0) = BN_NAN;
        }
        WHILE1 {
            ai = AI(DTYPE0);
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
                yi = FUNC(assqdm / (count - ddof));
            } else {
                yi = BN_NAN;
            }
            YI(DTYPE0) = yi;
        }
        count_inv = 1.0 / count;
        ddof_inv = 1.0 / (count - ddof);
        WHILE2 {
            ai = AI(DTYPE0);
            aold = AOLD(DTYPE0);
            if (ai == ai) {
                if (aold == aold) {
                    delta = ai - aold;
                    aold -= amean;
                    amean += delta * count_inv;
                    ai -= amean;
                    assqdm += (ai + aold) * delta;
                } else {
                    count++;
                    count_inv = 1.0 / count;
                    ddof_inv = 1.0 / (count - ddof);
                    delta = ai - amean;
                    amean += delta * count_inv;
                    assqdm += delta * (ai - amean);
                }
            } else {
                if (aold == aold) {
                    count--;
                    count_inv = 1.0 / count;
                    ddof_inv = 1.0 / (count - ddof);
                    if (count > 0) {
                        delta = aold - amean;
                        amean -= delta * count_inv;
                        assqdm -= delta * (aold - amean);
                    } else {
                        amean = 0;
                        assqdm = 0;
                    }
                }
            }
            if (count >= min_count) {
                if (assqdm < 0) {
                    assqdm = 0;
                }
                yi = FUNC(assqdm * ddof_inv);
            } else {
                yi = BN_NAN;
            }
            YI(DTYPE0) = yi;
        }
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
MOVE(NAME, DTYPE0) {
    int winddof = window - ddof;
    npy_DTYPE1 delta, amean, assqdm, yi, ai, aold;
    npy_DTYPE1 window_inv = 1.0 / window, winddof_inv = 1.0 / winddof;
    INIT(NPY_DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        amean = assqdm = 0;
        WHILE0 {
            ai = AI(DTYPE0);
            delta = ai - amean;
            amean += delta / (INDEX + 1);
            assqdm += delta * (ai - amean);
            YI(DTYPE1) = BN_NAN;
        }
        WHILE1 {
            ai = AI(DTYPE0);
            delta = ai - amean;
            amean += delta / (INDEX + 1);
            assqdm += delta * (ai - amean);
            yi = FUNC(assqdm / (INDEX + 1 - ddof));
            YI(DTYPE1) = yi;
        }
        WHILE2 {
            ai = AI(DTYPE0);
            aold = AOLD(DTYPE0);
            delta = ai - aold;
            aold -= amean;
            amean += delta * window_inv;
            ai -= amean;
            assqdm += (ai + aold) * delta;
            if (assqdm < 0) {
                assqdm = 0;
            }
            YI(DTYPE1) = FUNC(assqdm * winddof_inv);
        }
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

MOVE_MAIN(NAME, 1)
/* repeat end */


/* move_min, move_max, move_argmin, move_argmax -------------------------- */

/* repeat = {'MACRO_FLOAT': ['MOVE_NANMIN', 'MOVE_NANMAX'],
             'MACRO_INT':   ['MOVE_MIN',    'MOVE_MAX'],
             'COMPARE':     ['<=',          '>='],
             'FLIP':        ['>=',          '<='],
             'BIG_FLOAT':   ['BN_INFINITY', '-BN_INFINITY']} */

#define MACRO_FLOAT(dtype, yi, code) \
    ai = AI(dtype); \
    if (ai == ai) count++; else ai = BIG_FLOAT; \
    code; \
    if (ai COMPARE extreme_pair->value) { \
        extreme_pair->value = ai; \
        extreme_pair->death = INDEX + window; \
        last = extreme_pair; \
    } else { \
        while (last->value FLIP ai) { \
            if (last == ring) last = end; \
            last--; \
        } \
        last++; \
        if (last == end) last = ring; \
        last->value = ai; \
        last->death = INDEX + window; \
    } \
    yi_tmp = yi; /* yi might contain i and YI contains i++ */ \
    YI(dtype) = yi_tmp;

#define MACRO_INT(a_dtype, y_dtype, yi, code) \
    ai = AI(a_dtype); \
    code; \
    if (ai COMPARE extreme_pair->value) { \
        extreme_pair->value = ai; \
        extreme_pair->death = INDEX + window; \
        last = extreme_pair; \
    } else { \
        while (last->value FLIP ai) { \
            if (last == ring) last = end; \
            last--; \
        } \
        last++; \
        if (last == end) last = ring; \
        last->value = ai; \
        last->death = INDEX + window; \
    } \
    yi_tmp = yi; \
    YI(y_dtype) = yi_tmp;
/* repeat end */

/* repeat = {
   'NAME': ['move_min',    'move_max',
            'move_argmin', 'move_argmax'],
   'MACRO_FLOAT': ['MOVE_NANMIN', 'MOVE_NANMAX',
                   'MOVE_NANMIN', 'MOVE_NANMAX'],
   'MACRO_INT': ['MOVE_MIN', 'MOVE_MAX',
                 'MOVE_MIN', 'MOVE_MAX'],
   'COMPARE': ['<=', '>=',
               '<=', '>='],
   'BIG_FLOAT': ['BN_INFINITY', '-BN_INFINITY',
                 'BN_INFINITY', '-BN_INFINITY'],
   'BIG_INT': ['NPY_MAX_DTYPE0', 'NPY_MIN_DTYPE0',
               'NPY_MAX_DTYPE0', 'NPY_MIN_DTYPE0'],
   'VALUE': ['extreme_pair->value',           'extreme_pair->value',
             'INDEX-extreme_pair->death+window', 'INDEX-extreme_pair->death+window']
   } */
/* dtype = [['float64'], ['float32']] */
MOVE(NAME, DTYPE0) {
    npy_DTYPE0 ai, aold, yi_tmp;
    Py_ssize_t count;
    pairs *extreme_pair;
    pairs *end;
    pairs *last;
    pairs *ring = (pairs *)malloc(window * sizeof(pairs));
    INIT(NPY_DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        count = 0;
        end = ring + window;
        last = ring;
        extreme_pair = ring;
        ai = A0(DTYPE0);
        extreme_pair->value = ai == ai ? ai : BIG_FLOAT;
        extreme_pair->death = window;
        WHILE0 {
            MACRO_FLOAT(DTYPE0,
                        BN_NAN,
                        )
        }
        WHILE1 {
            MACRO_FLOAT(DTYPE0,
                        count >= min_count ? VALUE : BN_NAN,
                        )
        }
        WHILE2 {
            MACRO_FLOAT(DTYPE0,
                        count >= min_count ? VALUE : BN_NAN,
                        aold = AOLD(DTYPE0);
                        if (aold == aold) count--;
                        if (extreme_pair->death == INDEX) {
                            extreme_pair++;
                            if (extreme_pair >= end) extreme_pair = ring;
                        })
        }
        NEXT2
    }
    free(ring);
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
MOVE(NAME, DTYPE0) {
    npy_DTYPE0 ai;
    npy_DTYPE1 yi_tmp;
    pairs *extreme_pair;
    pairs *end;
    pairs *last;
    pairs *ring = (pairs *)malloc(window * sizeof(pairs));
    INIT(NPY_DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        end = ring + window;
        last = ring;
        extreme_pair = ring;
        ai = A0(DTYPE0);
        extreme_pair->value = ai;
        extreme_pair->death = window;
        WHILE0 {
            MACRO_INT(DTYPE0,
                      DTYPE1,
                      BN_NAN,
                      )
        }
        WHILE1 {
            MACRO_INT(DTYPE0,
                      DTYPE1,
                      VALUE,
                      )
        }
        WHILE2 {
            MACRO_INT(DTYPE0,
                      DTYPE1,
                      VALUE,
                      if (extreme_pair->death == INDEX) {
                          extreme_pair++;
                          if (extreme_pair >= end) extreme_pair = ring;
                      })
        }
        NEXT2
    }
    free(ring);
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

MOVE_MAIN(NAME, 0)
/* repeat end */

/* move_median ----------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
MOVE(move_median, DTYPE0) {
    npy_DTYPE0 ai;
    mm_handle *mm = mm_new_nan(window, min_count);
    INIT(NPY_DTYPE0)
    if (window == 1) {
        mm_free(mm);
        return PyArray_Copy(a);
    }
    if (mm == NULL) {
        MEMORY_ERR("Could not allocate memory for move_median");
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        WHILE0 {
            ai = AI(DTYPE0);
            YI(DTYPE0) = mm_update_init_nan(mm, ai);
        }
        WHILE1 {
            ai = AI(DTYPE0);
            YI(DTYPE0) = mm_update_init_nan(mm, ai);
        }
        WHILE2 {
            ai = AI(DTYPE0);
            YI(DTYPE0) = mm_update_nan(mm, ai);
        }
        mm_reset(mm);
        NEXT2
    }
    mm_free(mm);
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
MOVE(move_median, DTYPE0) {
    npy_DTYPE0 ai;
    mm_handle *mm = mm_new(window, min_count);
    INIT(NPY_DTYPE1)
    if (window == 1) {
        return PyArray_CastToType(a,
                                  PyArray_DescrFromType(NPY_DTYPE1),
                                  PyArray_CHKFLAGS(a, NPY_ARRAY_F_CONTIGUOUS));
    }
    if (mm == NULL) {
        MEMORY_ERR("Could not allocate memory for move_median");
    }
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        WHILE0 {
            ai = AI(DTYPE0);
            YI(DTYPE1) = mm_update_init(mm, ai);
        }
        WHILE1 {
            ai = AI(DTYPE0);
            YI(DTYPE1) = mm_update_init(mm, ai);
        }
        WHILE2 {
            ai = AI(DTYPE0);
            YI(DTYPE1) = mm_update(mm, ai);
        }
        mm_reset(mm);
        NEXT2
    }
    mm_free(mm);
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


MOVE_MAIN(move_median, 0)


/* move_rank-------------------------------------------------------------- */

#define MOVE_RANK(dtype0, dtype1, limit) \
    Py_ssize_t j; \
    npy_##dtype0 ai, aj; \
    npy_##dtype1 g, e, n, r; \
    ai = AI(dtype0); \
    if (ai == ai) { \
        g = 0; \
        e = 1; \
        n = 1; \
        r = 0; \
        for (j = limit; j < INDEX; j++) { \
            aj = AX(dtype0, j); \
            if (aj == aj) { \
                n++; \
                if (ai > aj) { \
                    g += 2; \
                } else if (ai == aj) { \
                    e++; \
                } \
            } \
        } \
        if (n < min_count) { \
            r = BN_NAN; \
        } else if (n == 1) { \
            r = 0.0; \
        } else { \
            r = 0.5 * (g + e - 1.0); \
            r = r / (n - 1.0); \
            r = 2.0 * (r - 0.5); \
        } \
    } else { \
        r = BN_NAN; \
    } \

/* dtype = [['float64', 'float64'], ['float32', 'float32']] */
MOVE(move_rank, DTYPE0) {
    INIT(NPY_DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        WHILE0 {
            YI(DTYPE1) = BN_NAN;
        }
        WHILE1 {
            MOVE_RANK(DTYPE0, DTYPE1, 0)
            YI(DTYPE1) = r;
        }
        WHILE2 {
            MOVE_RANK(DTYPE0, DTYPE1, INDEX - window + 1)
            YI(DTYPE1) = r;
        }
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64', 'float64'], ['int32', 'float64']] */
MOVE(move_rank, DTYPE0) {
    Py_ssize_t j;
    npy_DTYPE0 ai, aj;
    npy_DTYPE1 g, e, r, window_inv = 0.5 * 1.0 / (window - 1);
    INIT(NPY_DTYPE1)
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        WHILE0 {
            YI(DTYPE1) = BN_NAN;
        }
        WHILE1 {
            ai = AI(DTYPE0);
            g = 0;
            e = 1;
            r = 0;
            for (j = 0; j < INDEX; j++) {
                aj = AX(DTYPE0, j);
                if (ai > aj) {
                    g += 2;
                } else if (ai == aj) {
                    e++;
                }
            }
            if (INDEX < min_count - 1) {
                r = BN_NAN;
            } else if (INDEX == 0) {
                r = 0.0;
            } else {
                r = 0.5 * (g + e - 1.0);
                r = r / INDEX;
                r = 2.0 * (r - 0.5);
            }
            YI(DTYPE1) = r;
        }
        WHILE2 {
            ai = AI(DTYPE0);
            g = 0;
            e = 1;
            r = 0;
            for (j = INDEX - window + 1; j < INDEX; j++) {
                aj = AX(DTYPE0, j);
		if (ai > aj) {
                    g += 2;
                } else if (ai == aj) {
                    e++;
                }
            }
            if (window == 1) {
                r = 0.0;
            } else {
                r = window_inv * (g + e - 1.0);
                r = 2.0 * (r - 0.5);
            }
            YI(DTYPE1) = r;
        }
        NEXT2
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */


MOVE_MAIN(move_rank, 0)


/* python strings -------------------------------------------------------- */

PyObject *pystr_a = NULL;
PyObject *pystr_window = NULL;
PyObject *pystr_min_count = NULL;
PyObject *pystr_axis = NULL;
PyObject *pystr_ddof = NULL;

static int
intern_strings(void) {
    pystr_a = PyString_InternFromString("a");
    pystr_window = PyString_InternFromString("window");
    pystr_min_count = PyString_InternFromString("min_count");
    pystr_axis = PyString_InternFromString("axis");
    pystr_ddof = PyString_InternFromString("ddof");
    return pystr_a && pystr_window && pystr_min_count &&
           pystr_axis && pystr_ddof;
}

/* mover ----------------------------------------------------------------- */

static inline int
parse_args(PyObject *args,
           PyObject *kwds,
           int has_ddof,
           PyObject **a,
           PyObject **window,
           PyObject **min_count,
           PyObject **axis,
           PyObject **ddof) {
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int nkwds_found = 0;
        PyObject *tmp;
        switch (nargs) {
            case 4:
                if (has_ddof) {
                    *axis = PyTuple_GET_ITEM(args, 3);
                } else {
                    TYPE_ERR("wrong number of arguments");
                    return 0;
                }
            case 3: *min_count = PyTuple_GET_ITEM(args, 2);
            case 2: *window = PyTuple_GET_ITEM(args, 1);
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
        if (nargs + nkwds_found > 4 + has_ddof) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    } else {
        switch (nargs) {
            case 5:
                if (has_ddof) {
                    *ddof = PyTuple_GET_ITEM(args, 4);
                } else {
                    TYPE_ERR("wrong number of arguments");
                    return 0;
                }
            case 4:
                *axis = PyTuple_GET_ITEM(args, 3);
            case 3:
                *min_count = PyTuple_GET_ITEM(args, 2);
            case 2:
                *window = PyTuple_GET_ITEM(args, 1);
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
mover(char *name,
      PyObject *args,
      PyObject *kwds,
      move_t move_float64,
      move_t move_float32,
      move_t move_int64,
      move_t move_int32,
      int has_ddof) {

    int mc;
    int window;
    int axis;
    int ddof;
    int dtype;
    int ndim;

    Py_ssize_t length;

    PyArrayObject *a;
    PyObject *y;

    PyObject *a_obj = NULL;
    PyObject *window_obj = NULL;
    PyObject *min_count_obj = Py_None;
    PyObject *axis_obj = NULL;
    PyObject *ddof_obj = NULL;

    if (!parse_args(args, kwds, has_ddof, &a_obj, &window_obj,
                    &min_count_obj, &axis_obj, &ddof_obj)) {
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

    /* window */
    window = PyArray_PyIntAsInt(window_obj);
    if (error_converting(window)) {
        TYPE_ERR("`window` must be an integer");
        goto error;
    }

    /* min_count */
    if (min_count_obj == Py_None) {
        mc = window;
    } else {
        mc = PyArray_PyIntAsInt(min_count_obj);
        if (error_converting(mc)) {
            TYPE_ERR("`min_count` must be an integer or None");
            goto error;
        }
        if (mc > window) {
            PyErr_Format(PyExc_ValueError,
                         "min_count (%d) cannot be greater than window (%d)",
                         mc, window);
            goto error;
        } else if (mc <= 0) {
            VALUE_ERR("`min_count` must be greater than zero.");
            goto error;
        }
    }

    ndim = PyArray_NDIM(a);

    /* defend against 0d beings */
    if (ndim == 0) {
        VALUE_ERR("moving window functions require ndim > 0");
        goto error;
    }

    /* defend against the axis of negativity */
    if (axis_obj == NULL) {
        axis = ndim - 1;
    } else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer");
            goto error;
        }
        if (axis < 0) {
            axis += ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis);
                goto error;
            }
        } else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            goto error;
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

    length = PyArray_DIM(a, axis);
    if ((window < 1) || (window > length)) {
        PyErr_Format(PyExc_ValueError,
                     "Moving window (=%d) must between 1 and %zu, inclusive",
                     window, length);
        goto error;
    }

    dtype = PyArray_TYPE(a);

    if (dtype == NPY_float64) {
        y = move_float64(a, window, mc, axis, ddof);
    } else if (dtype == NPY_float32) {
        y = move_float32(a, window, mc, axis, ddof);
    } else if (dtype == NPY_int64) {
        y = move_int64(a, window, mc, axis, ddof);
    } else if (dtype == NPY_int32) {
        y = move_int32(a, window, mc, axis, ddof);
    } else {
        y = slow(name, args, kwds);
    }

    Py_DECREF(a);

    return y;

error:
    Py_DECREF(a);
    return NULL;

}

/* docstrings ------------------------------------------------------------- */

static char move_doc[] =
"Bottleneck moving window functions.";

static char move_sum_doc[] =
/* MULTILINE STRING BEGIN
move_sum(a, window, min_count=None, axis=-1)

Moving window sum along the specified axis, optionally ignoring NaNs.

This function cannot handle input arrays that contain Inf. When the
window contains Inf, the output will correctly be Inf. However, when Inf
moves out of the window, the remaining output values in the slice will
incorrectly be NaN.

Parameters
----------
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
>>> a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_sum(a, window=2)
array([ nan,   3.,   5.,  nan,  nan])
>>> bn.move_sum(a, window=2, min_count=1)
array([ 1.,  3.,  5.,  3.,  5.])

MULTILINE STRING END */

static char move_mean_doc[] =
/* MULTILINE STRING BEGIN
move_mean(a, window, min_count=None, axis=-1)

Moving window mean along the specified axis, optionally ignoring NaNs.

This function cannot handle input arrays that contain Inf. When the
window contains Inf, the output will correctly be Inf. However, when Inf
moves out of the window, the remaining output values in the slice will
incorrectly be NaN.

Parameters
----------
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
>>> a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_mean(a, window=2)
array([ nan,  1.5,  2.5,  nan,  nan])
>>> bn.move_mean(a, window=2, min_count=1)
array([ 1. ,  1.5,  2.5,  3. ,  5. ])

MULTILINE STRING END */

static char move_std_doc[] =
/* MULTILINE STRING BEGIN
move_std(a, window, min_count=None, axis=-1, ddof=0)

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
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
>>> a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_std(a, window=2)
array([ nan,  0.5,  0.5,  nan,  nan])
>>> bn.move_std(a, window=2, min_count=1)
array([ 0. ,  0.5,  0.5,  0. ,  0. ])

MULTILINE STRING END */

static char move_var_doc[] =
/* MULTILINE STRING BEGIN
move_var(a, window, min_count=None, axis=-1, ddof=0)

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
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
>>> a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_var(a, window=2)
array([ nan,  0.25,  0.25,  nan,  nan])
>>> bn.move_var(a, window=2, min_count=1)
array([ 0. ,  0.25,  0.25,  0. ,  0. ])

MULTILINE STRING END */

static char move_min_doc[] =
/* MULTILINE STRING BEGIN
move_min(a, window, min_count=None, axis=-1)

Moving window minimum along the specified axis, optionally ignoring NaNs.

float64 output is returned for all input data types.

Parameters
----------
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
    The moving minimum of the input array along the specified axis. The
    output has the same shape as the input. The dtype of the output is
    always float64.

Examples
--------
>>> a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_min(a, window=2)
array([ nan,   1.,   2.,  nan,  nan])
>>> bn.move_min(a, window=2, min_count=1)
array([ 1.,  1.,  2.,  3.,  5.])

MULTILINE STRING END */

static char move_max_doc[] =
/* MULTILINE STRING BEGIN
move_max(a, window, min_count=None, axis=-1)

Moving window maximum along the specified axis, optionally ignoring NaNs.

float64 output is returned for all input data types.

Parameters
----------
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
    The moving maximum of the input array along the specified axis. The
    output has the same shape as the input. The dtype of the output is
    always float64.

Examples
--------
>>> a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
>>> bn.move_max(a, window=2)
array([ nan,   2.,   3.,  nan,  nan])
>>> bn.move_max(a, window=2, min_count=1)
array([ 1.,  2.,  3.,  3.,  5.])

MULTILINE STRING END */

static char move_argmin_doc[] =
/* MULTILINE STRING BEGIN
move_argmin(a, window, min_count=None, axis=-1)

Moving window index of minimum along the specified axis, optionally
ignoring NaNs.

Index 0 is at the rightmost edge of the window. For example, if the array
is monotonically decreasing (increasing) along the specified axis then
the output array will contain zeros (window-1).

If there is a tie in input values within a window, then the rightmost
index is returned.

float64 output is returned for all input data types.

Parameters
----------
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
    The moving index of minimum values of the input array along the
    specified axis. The output has the same shape as the input. The dtype
    of the output is always float64.

Examples
--------
>>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
>>> bn.move_argmin(a, window=2)
array([ nan,   1.,   1.,   1.,   1.])

>>> a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
>>> bn.move_argmin(a, window=2)
array([ nan,   0.,   0.,   0.,   0.])

>>> a = np.array([2.0, 3.0, 4.0, 1.0, 7.0, 5.0, 6.0])
>>> bn.move_argmin(a, window=3)
array([ nan,  nan,   2.,   0.,   1.,   2.,   1.])

MULTILINE STRING END */

static char move_argmax_doc[] =
/* MULTILINE STRING BEGIN
move_argmax(a, window, min_count=None, axis=-1)

Moving window index of maximum along the specified axis, optionally
ignoring NaNs.

Index 0 is at the rightmost edge of the window. For example, if the array
is monotonically increasing (decreasing) along the specified axis then
the output array will contain zeros (window-1).

If there is a tie in input values within a window, then the rightmost
index is returned.

float64 output is returned for all input data types.

Parameters
----------
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
    The moving index of maximum values of the input array along the
    specified axis. The output has the same shape as the input. The dtype
    of the output is always float64.

Examples
--------
>>> a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
>>> bn.move_argmax(a, window=2)
array([ nan,   0.,   0.,   0.,   0.])

>>> a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
>>> bn.move_argmax(a, window=2)
array([ nan,   1.,   1.,   1.,   1.])

>>> a = np.array([2.0, 3.0, 4.0, 1.0, 7.0, 5.0, 6.0])
>>> bn.move_argmax(a, window=3)
array([ nan,  nan,   0.,   1.,   0.,   1.,   2.])

MULTILINE STRING END */

static char move_median_doc[] =
/* MULTILINE STRING BEGIN
move_median(a, window, min_count=None, axis=-1)

Moving window median along the specified axis, optionally ignoring NaNs.

float64 output is returned for all input data types.

Parameters
----------
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
    The moving median of the input array along the specified axis. The
    output has the same shape as the input.

Examples
--------
>>> a = np.array([1.0, 2.0, 3.0, 4.0])
>>> bn.move_median(a, window=2)
array([ nan,  1.5,  2.5,  3.5])
>>> bn.move_median(a, window=2, min_count=1)
array([ 1. ,  1.5,  2.5,  3.5])

MULTILINE STRING END */

static char move_rank_doc[] =
/* MULTILINE STRING BEGIN
move_rank(a, window, min_count=None, axis=-1)

Moving window ranking along the specified axis, optionally ignoring NaNs.

The output is normalized to be between -1 and 1. For example, with a
window width of 3 (and with no ties), the possible output values are
-1, 0, 1.

Ties are broken by averaging the rankings. See the examples below.

The runtime depends almost linearly on `window`. The more NaNs there are
in the input array, the shorter the runtime.

Parameters
----------
a : ndarray
    Input array. If `a` is not an array, a conversion is attempted.
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
    The moving ranking along the specified axis. The output has the same
    shape as the input. For integer input arrays, the dtype of the output
    is float64.

Examples
--------
With window=3 and no ties, there are 3 possible output values, i.e.
[-1., 0., 1.]:

>>> a = np.array([1, 2, 3, 9, 8, 7, 5, 6, 4])
>>> bn.move_rank(a, window=3)
    array([ nan,  nan,   1.,   1.,   0.,  -1.,  -1.,   0.,  -1.])

Ties are broken by averaging the rankings of the tied elements:

>>> a = np.array([1, 2, 3, 3, 3, 4])
>>> bn.move_rank(a, window=3)
    array([ nan,  nan,  1. ,  0.5,  0. ,  1. ])

In an increasing sequence, the moving window ranking is always equal to 1:

>>> a = np.array([1, 2, 3, 4, 5])
>>> bn.move_rank(a, window=2)
    array([ nan,   1.,   1.,   1.,   1.])

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
move_methods[] = {
    {"move_sum",    (PyCFunction)move_sum,    VARKEY, move_sum_doc},
    {"move_mean",   (PyCFunction)move_mean,   VARKEY, move_mean_doc},
    {"move_std",    (PyCFunction)move_std,    VARKEY, move_std_doc},
    {"move_var",    (PyCFunction)move_var,    VARKEY, move_var_doc},
    {"move_min",    (PyCFunction)move_min,    VARKEY, move_min_doc},
    {"move_max",    (PyCFunction)move_max,    VARKEY, move_max_doc},
    {"move_argmin", (PyCFunction)move_argmin, VARKEY, move_argmin_doc},
    {"move_argmax", (PyCFunction)move_argmax, VARKEY, move_argmax_doc},
    {"move_median", (PyCFunction)move_median, VARKEY, move_median_doc},
    {"move_rank",   (PyCFunction)move_rank,   VARKEY, move_rank_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
move_def = {
   PyModuleDef_HEAD_INIT,
   "move",
   move_doc,
   -1,
   move_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_move(void)
#else
#define RETVAL
initmove(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&move_def);
    #else
        PyObject *m = Py_InitModule3("move", move_methods, move_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
