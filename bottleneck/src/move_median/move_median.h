#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <bn_config.h>

typedef size_t idx_t;
typedef double ai_t;

#if BINARY_TREE==1
    #define NUM_CHILDREN 2
#else
    /* maximum of 8 due to the manual loop-unrolling used in the code */
    #define NUM_CHILDREN 8
#endif

/* Find indices of parent and first child */
#define P_IDX(i) ((i) - 1) / NUM_CHILDREN
#define FC_IDX(i) NUM_CHILDREN * (i) + 1

/* are we in the small heap (SM), large heap (LH) or NaN array (NA)? */
#define SH 0
#define LH 1
#define NA 2

#define FIRST_LEAF(n) (idx_t) ceil((n - 1) / (double)NUM_CHILDREN)

struct _mm_node {
    int              region; /* SH small heap, LH large heap, NA nan array */
    ai_t             ai;     /* The node's value */
    idx_t            idx;    /* The node's index in the heap or nan array */
    struct _mm_node *next;   /* The next node in order of insertion */
};
typedef struct _mm_node mm_node;

struct _mm_handle {
    idx_t     window;    /* window size */
    int       odd;       /* is window even (0) or odd (1) */
    idx_t     min_count; /* Same meaning as in bn.move_median */
    idx_t     n_s;       /* Number of nodes in the small heap */
    idx_t     n_l;       /* Number of nodes in the large heap */
    idx_t     n_n;       /* Number of nodes in the nan array */
    mm_node **s_heap;    /* The max heap of small ai */
    mm_node **l_heap;    /* The min heap of large ai */
    mm_node **n_array;   /* The nan array */
    mm_node **nodes;     /* s_heap and l_heap point into this array */
    mm_node  *node_data; /* Pointer to memory location where nodes live */
    mm_node  *oldest;    /* The oldest node */
    mm_node  *newest;    /* The newest node (most recent insert) */
    idx_t s_first_leaf;  /* All nodes this index or greater are leaf nodes */
    idx_t l_first_leaf;  /* All nodes this index or greater are leaf nodes */
};
typedef struct _mm_handle mm_handle;

/* non-nan functions */
mm_handle *mm_new(const idx_t window, idx_t min_count);
ai_t mm_update_init(mm_handle *mm, ai_t ai);
ai_t mm_update(mm_handle *mm, ai_t ai);

/* nan functions */
mm_handle *mm_new_nan(const idx_t window, idx_t min_count);
ai_t mm_update_init_nan(mm_handle *mm, ai_t ai);
ai_t mm_update_nan(mm_handle *mm, ai_t ai);

/* functions common to non-nan and nan cases */
void mm_reset(mm_handle *mm);
void mm_free(mm_handle *mm);

/* Copied from Cython ---------------------------------------------------- */

/* NaN */
#ifdef NAN
    #define MM_NAN() ((float) NAN)
#else
    static inline float MM_NAN(void) {
        float value;
        memset(&value, 0xFF, sizeof(value));
        return value;
    }
#endif
