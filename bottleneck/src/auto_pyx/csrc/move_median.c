#if NOCYTHON==1

    #include <stdio.h>
    #include <stddef.h>
    #include <stdlib.h>
    #include <math.h>
    #include <assert.h>

    typedef size_t idx_t;
    typedef double ai_t;

    /* The number of children has a maximum of 8 due to the manual loop-
     * unrolling used in the code below. */
    const int NUM_CHILDREN = 2;

#else

    typedef npy_float64 ai_t;
    typedef npy_intp idx_t;
    const int NUM_CHILDREN = 8;

#endif

// Minimum of two numbers.
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Find indices of parent and first child
#define P_IDX(i) ((i) - 1) / NUM_CHILDREN
#define FC_IDX(i) NUM_CHILDREN * (i) + 1

#define SWAP_NODES(heap, idx1, node1, idx2, node2) \
heap[idx1] = node2;                                \
heap[idx2] = node1;                                \
node1->idx = idx2;                                 \
node2->idx = idx1;                                 \
idx1       = idx2

// are we in the small heap (SM), large heap (LH) or NaN array (NA)?
#define SH 0
#define LH 1
#define NA 2

#define FIRST_LEAF(n) ceil((n - 1) / (double)NUM_CHILDREN)


/*
-----------------------------------------------------------------------------
  Data structures
-----------------------------------------------------------------------------
*/

struct _mm_node {
    int              region; // SH small heap, LH large heap, NA nan array
    ai_t             ai;     // The node's value
    idx_t            idx;    // The node's index in the heap or nan array
    struct _mm_node *next;   // The next node in order of insertion
};
typedef struct _mm_node mm_node;

struct _mm_handle {
    idx_t     window;    // window size
    idx_t     n_s;       // Number of nodes in the small heap
    idx_t     n_l;       // Number of nodes in the large heap
    idx_t     n_n;       // Number of nodes in the nan array
    idx_t     min_count; // Same meaning as in bn.move_median
    mm_node **s_heap;    // The max heap of small ai
    mm_node **l_heap;    // The min heap of large ai
    mm_node **n_array;   // The nan array
    mm_node **nodes;     // All nodes. s_heap and l_heap point into this array
    mm_node  *node_data; // Pointer to memory location where nodes live
    mm_node  *oldest;    // The oldest node
    mm_node  *newest;    // The newest node (most recent insert)
    idx_t s_first_leaf;  // All nodes at this index or greater are leaf nodes
    idx_t l_first_leaf;  // All nodes at this index or greater are leaf nodes
};
typedef struct _mm_handle mm_handle;


/*
-----------------------------------------------------------------------------
  Prototypes
-----------------------------------------------------------------------------
*/

// top-level non-nan functions
inline mm_handle *mm_new(const idx_t window, idx_t min_count);
inline ai_t mm_update_init(mm_handle *mm, ai_t ai);
inline ai_t mm_update(mm_handle *mm, ai_t ai);

// top-level nan functions
inline mm_handle *mm_new_nan(const idx_t window, idx_t min_count);
inline ai_t mm_update_init_nan(mm_handle *mm, ai_t ai);
inline ai_t mm_update_nan(mm_handle *mm, ai_t ai);

// top-level functions common to non-nan and nan cases
inline void mm_reset(mm_handle *mm);
inline void mm_free(mm_handle *mm);

// helper functions
inline ai_t mm_get_median(mm_handle *mm);
inline void heapify_small_node(mm_handle *mm, idx_t idx);
inline void heapify_large_node(mm_handle *mm, idx_t idx);
inline idx_t mm_get_smallest_child(mm_node **heap, idx_t window, idx_t idx,
                                   mm_node **child);
inline idx_t mm_get_largest_child(mm_node **heap, idx_t window, idx_t idx,
                                  mm_node **child);
inline void mm_move_up_small(mm_node **heap, idx_t idx, mm_node *node,
                             idx_t p_idx, mm_node *parent);
inline void mm_move_down_small(mm_node **heap, idx_t window, idx_t idx,
                               mm_node *node);
inline void mm_move_down_large(mm_node **heap, idx_t idx, mm_node *node,
                               idx_t p_idx, mm_node *parent);
inline void mm_move_up_large(mm_node **heap, idx_t window, idx_t idx,
                             mm_node *node);
inline void mm_swap_heap_heads(mm_node **s_heap, idx_t n_s, mm_node **l_heap,
                               idx_t n_l, mm_node *s_node, mm_node *l_node);

// debug
ai_t *mm_move_median(ai_t *arr, idx_t length, idx_t window, idx_t min_count);
int mm_assert_equal(ai_t *actual, ai_t *desired, ai_t *input, idx_t length,
                    char *err_msg);
int mm_unit_test(void);
void mm_dump(mm_handle *mm);
void mm_print_binary_heap(mm_node **heap, idx_t n_array, idx_t oldest_idx,
                          idx_t newest_idx);
void mm_check(mm_handle *mm);
void mm_print_chain(mm_handle *mm);
void mm_print_line(void);
void mm_print_node(mm_node *node);


/*
-----------------------------------------------------------------------------
  Top-level non-nan functions
-----------------------------------------------------------------------------
*/

/* At the start of bn.move_median two heaps are created. One heap contains the
 * small values (a max heap); the other heap contains the large values
 * (a min heap). And the handle contains information about the heaps. It is
 * the handle that is returned by the function. */
inline mm_handle *
mm_new(const idx_t window, idx_t min_count)
{

    mm_handle *mm = malloc(sizeof(mm_handle));
    mm->nodes = malloc(window * sizeof(mm_node*));
    mm->node_data = malloc(window * sizeof(mm_node));

    mm->s_heap = mm->nodes;
    mm->l_heap = &mm->nodes[window / 2 + window % 2];

    mm->window = window;
    mm->min_count = min_count;

    mm_reset(mm);

    return mm;
}


/* Insert a new value, ai, into one of the heaps. Use this function when
 * the heaps contains less than window-1 values. Returns the median value.
 * Once there are window-1 values in the heap, switch to using mm_update. */
inline ai_t
mm_update_init(mm_handle *mm, ai_t ai)
{

    mm_node *node = NULL;
    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;

    node = &mm->node_data[n_s + n_l];
    node->ai = ai;

    if (n_s == 0) {
        // the first node to appear in a heap
        mm->s_heap[0] = node;
        node->region = SH;
        node->idx = 0;
        if (n_s + n_l == 0) {
            // only need to set the oldest node once
            mm->oldest = node;
        } else {
            mm->newest->next = node;
        }
        mm->n_s = 1;
        mm->s_first_leaf = 0;
    }
    else
    {
        // at least one node already exists in the heaps

        mm->newest->next = node;
        if (n_s > n_l)
        {
            // add new node to large heap
            mm->l_heap[n_l] = node;
            node->region = LH;
            node->idx = n_l;
            ++mm->n_l;
            mm->l_first_leaf = FIRST_LEAF(mm->n_l);
            heapify_large_node(mm, n_l);
        }
        else
        {
            // add new node to small heap
            mm->s_heap[n_s] = node;
            node->region = SH;
            node->idx = n_s;
            ++mm->n_s;
            mm->s_first_leaf = FIRST_LEAF(mm->n_s);
            heapify_small_node(mm, n_s);
        }
    }

    mm->newest = node;

    return mm_get_median(mm);
}


/* Insert a new value, ai, into the double heap structure. Use this function
 * when the double heap contains at least window-1 values. Returns the median
 * value. If there are less than window-1 values in the heap, use
 * mm_update_init. */
inline ai_t
mm_update(mm_handle *mm, ai_t ai)
{
    // node is oldest node with ai of newest node
    mm_node *node = mm->oldest;
    node->ai = ai;

    // update oldest, newest
    mm->oldest = mm->oldest->next;
    mm->newest->next = node;
    mm->newest = node;

    if (node->region == SH) {
        heapify_small_node(mm, node->idx);
    } else {
        heapify_large_node(mm, node->idx);
    }

    return mm_get_median(mm);
}


/*
-----------------------------------------------------------------------------
  Top-level nan functions
-----------------------------------------------------------------------------
*/

/* At the start of bn.move_median two heaps are created. One heap contains the
 * small values (a max heap); the other heap contains the large values
 * (a min heap). And the handle contains information about the heaps. It is
 * the handle that is returned by the function. */
inline mm_handle *
mm_new_nan(const idx_t window, idx_t min_count)
{

    mm_handle *mm = malloc(sizeof(mm_handle));
    mm->nodes = malloc(2 * window * sizeof(mm_node*));
    mm->node_data = malloc(window * sizeof(mm_node));

    mm->s_heap = mm->nodes;
    mm->l_heap = &mm->nodes[window / 2 + window % 2];
    mm->n_array = &mm->nodes[window];

    mm->window = window;
    mm->min_count = min_count;

    mm_reset(mm);

    return mm;
}


/* Insert a new value, ai, into one of the heaps. Use this function when
 * the heaps contains less than window-1 values. Returns the median value.
 * Once there are window-1 values in the heap, switch to using mm_update. */
inline ai_t
mm_update_init_nan(mm_handle *mm, ai_t ai)
{

    mm_node *node = NULL;
    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;
    idx_t n_n = mm->n_n;

    node = &mm->node_data[n_s + n_l + n_n];
    node->ai = ai;

    if (isnan(ai)) {
        mm->n_array[n_n] = node;
        node->region = NA;
        node->idx = n_n;
        if (n_s + n_l + n_n == 0) {
            // only need to set the oldest node once
            mm->oldest = node;
        } else {
            mm->newest->next = node;
        }
        ++mm->n_n;
    } else {
        if (n_s == 0) {
            // the first node to appear in a heap
            mm->s_heap[0] = node;
            node->region = SH;
            node->idx = 0;
            if (n_s + n_l + n_n == 0) {
                // only need to set the oldest node once
                mm->oldest = node;
            } else {
                mm->newest->next = node;
            }
            mm->n_s = 1;
            mm->s_first_leaf = 0;
        }
        else
        {
            // at least one node already exists in the heaps

            mm->newest->next = node;
            if (n_s > n_l)
            {
                // add new node to large heap
                mm->l_heap[n_l] = node;
                node->region = LH;
                node->idx = n_l;
                ++mm->n_l;
                mm->l_first_leaf = FIRST_LEAF(mm->n_l);
                heapify_large_node(mm, n_l);
            }
            else
            {
                // add new node to small heap
                mm->s_heap[n_s] = node;
                node->region = SH;
                node->idx = n_s;
                ++mm->n_s;
                mm->s_first_leaf = FIRST_LEAF(mm->n_s);
                heapify_small_node(mm, n_s);
            }
        }
    }

    mm->newest = node;

    return mm_get_median(mm);
}


/* Insert a new value, ai, into the double heap structure. Use this function
 * when the double heap contains at least window-1 values. Returns the median
 * value. If there are less than window-1 values in the heap, use
 * mm_update_init. */
inline ai_t
mm_update_nan(mm_handle *mm, ai_t ai)
{

    // node is oldest node with ai of newest node
    mm_node *node = mm->oldest;
    idx_t idx = node->idx;
    node->ai = ai;

    // update oldest, newest
    mm->oldest = mm->oldest->next;
    mm->newest->next = node;
    mm->newest = node;

    mm_node **l_heap = mm->l_heap;
    mm_node **s_heap = mm->s_heap;
    mm_node **n_array = mm->n_array;

    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;
    idx_t n_n = mm->n_n;

    mm_node *node2;

    if (isnan(ai)) {

        if (node->region == SH) {

            /* Oldest node is in the small heap and needs to be moved
             * to the nan array. Resulting hole in the small heap will be
             * filled with the rightmost leaf of the last row of the small
             * heap. */

            // insert node into nan array
            node->region = NA;
            node->idx = n_n;
            n_array[n_n] = node;
            ++mm->n_n;

            // plug small heap hole
            --mm->n_s;
            if (mm->n_s == 0) {
                mm->s_first_leaf = 0;
                if (n_l > 0) {

                    // move head node from the large heap to the small heap
                    node2 = mm->l_heap[0];
                    node2->region = SH;
                    s_heap[0] = node2;
                    mm->n_s = 1;
                    mm->s_first_leaf = 0;

                    // plug hole in large heap
                    node2= mm->l_heap[mm->n_l - 1];
                    node2->idx = 0;
                    l_heap[0] = node2;
                    --mm->n_l;
                    if (mm->n_l == 0) {
                        mm->l_first_leaf = 0;
                    } else {
                        mm->l_first_leaf = FIRST_LEAF(mm->n_l);
                    }
                    heapify_large_node(mm, 0);

                }
            } else {
                s_heap[idx] = s_heap[n_s - 1];
                s_heap[idx]->idx = idx;
                if (mm->n_s < mm->n_l) {

                    // move head node from the large heap to the small heap
                    node2 = mm->l_heap[0];
                    node2->idx = mm->n_s;
                    node2->region = SH;
                    s_heap[mm->n_s] = node2;
                    ++mm->n_s;
                    mm->l_first_leaf = FIRST_LEAF(mm->n_s);
                    heapify_small_node(mm, node2->idx);

                    // plug hole in large heap
                    node2= mm->l_heap[mm->n_l - 1];
                    node2->idx = 0;
                    l_heap[0] = node2;
                    --mm->n_l;
                    if (mm->n_l == 0) {
                        mm->l_first_leaf = 0;
                    } else {
                        mm->l_first_leaf = FIRST_LEAF(mm->n_l);
                    }
                    heapify_large_node(mm, 0);

                } else {
                mm->s_first_leaf = FIRST_LEAF(mm->n_s);
                heapify_small_node(mm, idx);
                }
            }

        } else if (node->region == LH) {

            /* Oldest node is in the large heap and needs to be moved
             * to the nan array. Resulting hole in the large heap will be
             * filled with the rightmost leaf of the last row of the large
             * heap. */

            // insert node into nan array
            node->region = NA;
            node->idx = n_n;
            n_array[n_n] = node;
            ++mm->n_n;

            // plug large heap hole
            l_heap[idx] = l_heap[n_l - 1];
            l_heap[idx]->idx = idx;
            --mm->n_l;
            if (mm->n_l == 0) {
                mm->l_first_leaf = 0;
            } else {
                mm->l_first_leaf = FIRST_LEAF(mm->n_l);
            }

            if (mm->n_l < mm->n_s - 1) {

                // move head node from the small heap to the large heap
                node2 = mm->s_heap[0];
                node2->idx = mm->n_l;
                node2->region = LH;
                l_heap[mm->n_l] = node2;
                ++mm->n_l;
                mm->l_first_leaf = FIRST_LEAF(mm->n_l);
                heapify_large_node(mm, node2->idx);

                // plug hole in small heap
                node2= mm->s_heap[mm->n_s - 1];
                node2->idx = 0;
                s_heap[0] = node2;
                --mm->n_s;
                if (mm->n_s == 0) {
                    mm->s_first_leaf = 0;
                } else {
                    mm->s_first_leaf = FIRST_LEAF(mm->n_s);
                }
                heapify_small_node(mm, 0);

            }

            // reorder large heap if needed
            heapify_large_node(mm, idx);

        } else if (node->region == NA) {

            //  insert node into nan heap
            n_array[idx] = node;

        }
    } else {

        if (node->region == SH) {
            heapify_small_node(mm, idx);
        }
        else if (node->region == LH) {
            heapify_large_node(mm, idx);
        }
        else {

            // ai is not NaN but oldest node is in nan array

            if (n_s > n_l) {

                // insert into large heap
                node->region = LH;
                node->idx = n_l;
                l_heap[n_l] = node;
                ++mm->n_l;
                mm->l_first_leaf = FIRST_LEAF(mm->n_l);

                // plug nan array hole
                if (n_n > 2) {
                    n_array[idx] = n_array[n_n - 1];
                    n_array[idx]->idx = idx;
                }
                --mm->n_n;

                // reorder large heap if needed
                heapify_large_node(mm, n_l);

            } else {

                // insert into small heap
                node->region = SH;
                node->idx = n_s;
                s_heap[n_s] = node;
                ++mm->n_s;
                mm->s_first_leaf = FIRST_LEAF(mm->n_s);

                // plug nan array hole
                if (n_n > 2) {
                    n_array[idx] = n_array[n_n - 1];
                    n_array[idx]->idx = idx;
                }
                --mm->n_n;

                // reorder small heap if needed
                heapify_small_node(mm, n_s);

            }
        }

    }

    return mm_get_median(mm);
}


/*
-----------------------------------------------------------------------------
  Top-level functions common to nan and non-nan cases
-----------------------------------------------------------------------------
*/

/* At the end of each slice the double heap is reset (mm_reset) to prepare
 * for the next slice. In the 2d input array case (with axis=1), each slice
 * is a row of the input array. */
inline void
mm_reset(mm_handle *mm)
{
    mm->n_l = 0;
    mm->n_s = 0;
    mm->n_n = 0;
    mm->oldest = NULL;
    mm->newest = NULL;
    mm->s_first_leaf = 0;
    mm->l_first_leaf = 0;
}


/*  After bn.move_median is done, free the memory */
inline void
mm_free(mm_handle *mm)
{
    free(mm->node_data);
    free(mm->nodes);
    free(mm);
}


/*
-----------------------------------------------------------------------------
  Utility functions
-----------------------------------------------------------------------------
*/

/* Return the current median value when there are less than window values
 * in the double heap. */
inline ai_t
mm_get_median(mm_handle *mm)
{
    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;

    idx_t numel_total = n_l + n_s;

    if (numel_total < mm->min_count)
        return NAN;

    idx_t effective_window_size = min(mm->window, numel_total);

    if (effective_window_size % 2 == 1) {
        if (n_l > n_s)
            return mm->l_heap[0]->ai;
        else
            return mm->s_heap[0]->ai;
    }
    else
        return (mm->s_heap[0]->ai + mm->l_heap[0]->ai) / 2;
}


inline void
heapify_small_node(mm_handle *mm, idx_t idx)
{
    idx_t idx2;
    mm_node *node;
    mm_node *node2;
    mm_node **s_heap;
    mm_node **l_heap;
    idx_t n_s, n_l;
    ai_t ai;

    s_heap = mm->s_heap;
    l_heap = mm->l_heap;
    node = s_heap[idx];
    n_s = mm->n_s;
    n_l = mm->n_l;
    ai = node->ai;

    // Internal or leaf node.
    if (idx > 0) {
        idx2 = P_IDX(idx);
        node2 = s_heap[idx2];

        // Move up.
        if (ai > node2->ai) {
            mm_move_up_small(s_heap, idx, node, idx2, node2);

            // Maybe swap between heaps.
            node2 = l_heap[0];
            if ((node2 != NULL) && (ai > node2->ai)) {
                mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
            }
        }

        // Move down.
        else if (idx < mm->s_first_leaf) {
            mm_move_down_small(s_heap, n_s, idx, node);
        }
    }

    // Head node.
    else {
        node2 = l_heap[0];
        if (n_l > 0 && ai > node2->ai) {
            mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
        } else {
            mm_move_down_small(s_heap, n_s, idx, node);
        }
    }
}


inline void
heapify_large_node(mm_handle *mm, idx_t idx)
{
    idx_t idx2;
    mm_node *node;
    mm_node *node2;
    mm_node **s_heap;
    mm_node **l_heap;
    idx_t n_s, n_l;
    ai_t ai;

    s_heap = mm->s_heap;
    l_heap = mm->l_heap;
    node = l_heap[idx];
    n_s = mm->n_s;
    n_l = mm->n_l;
    ai = node->ai;

    // Internal or leaf node.
    if (idx > 0) {
        idx2 = P_IDX(idx);
        node2 = l_heap[idx2];

        // Move down.
        if (ai < node2->ai) {
            mm_move_down_large(l_heap, idx, node, idx2, node2);

            // Maybe swap between heaps.
            node2 = s_heap[0];
            if (ai < node2->ai) {
                mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
            }
        }

        // Move up.
        else if (idx < mm->l_first_leaf) {
            mm_move_up_large(l_heap, n_l, idx, node);
        }
    }

    // Head node.
    else {
        node2 = s_heap[0];
        if (n_s > 0 && ai < node2->ai) {
            mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
        } else {
            mm_move_up_large(l_heap, n_l, idx, node);
        }
    }

}


/*
 * Return the index of the smallest child of the node. The pointer
 * child will also be set.
 */
inline idx_t
mm_get_smallest_child(mm_node **heap, idx_t window, idx_t idx, mm_node **child)
{
    idx_t i0 = FC_IDX(idx);
    idx_t i1 = i0 + NUM_CHILDREN;
    i1 = min(i1, window);

    switch(i1 - i0) {
        case  8: if (heap[i0 + 7]->ai < heap[idx]->ai) { idx = i0 + 7; }
        case  7: if (heap[i0 + 6]->ai < heap[idx]->ai) { idx = i0 + 6; }
        case  6: if (heap[i0 + 5]->ai < heap[idx]->ai) { idx = i0 + 5; }
        case  5: if (heap[i0 + 4]->ai < heap[idx]->ai) { idx = i0 + 4; }
        case  4: if (heap[i0 + 3]->ai < heap[idx]->ai) { idx = i0 + 3; }
        case  3: if (heap[i0 + 2]->ai < heap[idx]->ai) { idx = i0 + 2; }
        case  2: if (heap[i0 + 1]->ai < heap[idx]->ai) { idx = i0 + 1; }
        case  1: if (heap[i0    ]->ai < heap[idx]->ai) { idx = i0;     }
    }

    *child = heap[idx];
    return idx;
}


/*
 * Return the index of the largest child of the node. The pointer
 * child will also be set.
 */
inline idx_t
mm_get_largest_child(mm_node **heap, idx_t window, idx_t idx, mm_node **child)
{
    idx_t i0 = FC_IDX(idx);
    idx_t i1 = i0 + NUM_CHILDREN;
    i1 = min(i1, window);

    switch(i1 - i0) {
        case  8: if (heap[i0 + 7]->ai > heap[idx]->ai) { idx = i0 + 7; }
        case  7: if (heap[i0 + 6]->ai > heap[idx]->ai) { idx = i0 + 6; }
        case  6: if (heap[i0 + 5]->ai > heap[idx]->ai) { idx = i0 + 5; }
        case  5: if (heap[i0 + 4]->ai > heap[idx]->ai) { idx = i0 + 4; }
        case  4: if (heap[i0 + 3]->ai > heap[idx]->ai) { idx = i0 + 3; }
        case  3: if (heap[i0 + 2]->ai > heap[idx]->ai) { idx = i0 + 2; }
        case  2: if (heap[i0 + 1]->ai > heap[idx]->ai) { idx = i0 + 1; }
        case  1: if (heap[i0    ]->ai > heap[idx]->ai) { idx = i0;     }
    }

    *child = heap[idx];
    return idx;
}


/* Move the given node up through the heap to the appropriate position. */
inline void
mm_move_up_small(mm_node **heap, idx_t idx, mm_node *node, idx_t p_idx,
                 mm_node *parent)
{
    do {
        SWAP_NODES(heap, idx, node, p_idx, parent);
        if (idx == 0) {
            break;
        }
        p_idx = P_IDX(idx);
        parent = heap[p_idx];
    } while (node->ai > parent->ai);
}


/* Move the given node down through the heap to the appropriate position. */
inline void
mm_move_down_small(mm_node **heap, idx_t window, idx_t idx, mm_node *node)
{
    mm_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = mm_get_largest_child(heap, window, idx, &child);

    while (ai < child->ai) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = mm_get_largest_child(heap, window, idx, &child);
    }
}


/* Move the given node down through the heap to the appropriate position. */
inline void
mm_move_down_large(mm_node **heap, idx_t idx, mm_node *node, idx_t p_idx,
                   mm_node *parent)
{
    do {
        SWAP_NODES(heap, idx, node, p_idx, parent);
        if (idx == 0) {
            break;
        }
        p_idx = P_IDX(idx);
        parent = heap[p_idx];
    } while (node->ai < parent->ai);
}


/* Move the given node up through the heap to the appropriate position. */
inline void
mm_move_up_large(mm_node **heap, idx_t window, idx_t idx, mm_node *node)
{
    mm_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = mm_get_smallest_child(heap, window, idx, &child);

    while (ai > child->ai) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = mm_get_smallest_child(heap, window, idx, &child);
    }
}


/* Swap the heap heads. */
inline void
mm_swap_heap_heads(mm_node **s_heap, idx_t n_s, mm_node **l_heap, idx_t n_l,
                   mm_node *s_node, mm_node *l_node)
{
    s_node->region = LH;
    l_node->region = SH;
    s_heap[0] = l_node;
    l_heap[0] = s_node;
    mm_move_down_small(s_heap, n_s, 0, l_node);
    mm_move_up_large(l_heap, n_l, 0, s_node);
}


/*
-----------------------------------------------------------------------------
  Debug utilities
-----------------------------------------------------------------------------
*/

int main(void)
{
    return mm_unit_test();
}


/* moving window median of 1d arrays returns output array */
ai_t *mm_move_median(ai_t *arr, idx_t length, idx_t window, idx_t min_count)
{
    mm_handle *mm;
    ai_t *out;
    idx_t i;

    out = malloc(length * sizeof(ai_t));
    mm = mm_new_nan(window, min_count);
    for (i=0; i < length; i++) {
        if (i < window) {
            out[i] = mm_update_init_nan(mm, arr[i]);
        } else {
            out[i] = mm_update_nan(mm, arr[i]);
        }
        if (i == window) {
            mm_print_line();
            printf("window complete; switch to mm_update\n");
        }
        mm_print_line();
        printf("inserting ai = %f\n", arr[i]);
        mm_print_chain(mm);
        mm_dump(mm);
        printf("\nmedian = %f\n\n", out[i]);
        mm_check(mm);
    }
    mm_free(mm);

    return out;
}


/* assert that two arrays are equal */
int mm_assert_equal(ai_t *actual, ai_t *desired, ai_t *input, idx_t length,
                 char *err_msg)
{
    idx_t i;
    int failed = 0;

    mm_print_line();
    printf("%s\n", err_msg);
    mm_print_line();
    printf("input    actual   desired\n");
    for (i=0; i < length; i++)
    {
        if (isnan(actual[i]) && isnan(desired[i])) {
            printf("%8f %8f %8f\n", input[i], actual[i], desired[i]);
        }
        else if (actual[i] != desired[i]) {
            failed = 1;
            printf("%8f %8f %8f BUG\n", input[i], actual[i], desired[i]);
        }
        else
            printf("%8f %8f %8f\n", input[i], actual[i], desired[i]);
    }

    return failed;
}


int mm_unit_test(void)
{
    ai_t arr_input[] = {6,  5,   7, INFINITY,  1,  INFINITY,   NAN,    NAN};
    ai_t desired[] =   {6,  5.5, 6, 6.5,       6,  6.5,       7,    INFINITY};
    ai_t *actual;
    int window = 6;
    int min_count = 1;
    int length;
    char *err_msg;
    int failed;

    length = sizeof(arr_input) / sizeof(*arr_input);
    err_msg = malloc(1024 * sizeof *err_msg);
    sprintf(err_msg, "move_median failed with window=%d, min_count=%d",
            window, min_count);

    actual = mm_move_median(arr_input, length, window, min_count);
    failed = mm_assert_equal(actual, desired, arr_input, length, err_msg);

    free(actual);
    free(err_msg);

    return failed;
}


void mm_print_node(mm_node *node)
{
    printf("\n\n%d small\n", node->region);
    printf("%d idx\n", node->idx);
    printf("%f ai\n", node->ai);
    printf("%p next\n\n", node->next);
}


void mm_print_chain(mm_handle *mm)
{
    idx_t i;
    mm_node *node;

    printf("\nchain\n");
    node = mm->oldest;
    printf("\t%6.2f region %d idx %d addr %p\n", node->ai, node->region,
           node->idx, node);
    for (i=1; i < mm->n_s + mm->n_l + mm->n_n; i++) {
        node = node->next;
        printf("\t%6.2f region %d idx %d addr %p\n", node->ai, node->region,
               node->idx, node);
    }
}


void mm_check(mm_handle *mm)
{

    int ndiff;
    idx_t i;
    mm_node *child;
    mm_node *parent;

    // small heap
    for (i=0; i<mm->n_s; i++) {
        assert(mm->s_heap[i]->idx == i);
        assert(mm->s_heap[i]->ai == mm->s_heap[i]->ai);
        if (i > 0) {
            child = mm->s_heap[i];
            parent = mm->s_heap[P_IDX(child->idx)];
            assert(child->ai <= parent->ai);
        }
    }

    // large heap
    for (i=0; i<mm->n_l; i++) {
        assert(mm->l_heap[i]->idx == i);
        assert(mm->l_heap[i]->ai == mm->l_heap[i]->ai);
        if (i > 0) {
            child = mm->l_heap[i];
            parent = mm->l_heap[P_IDX(child->idx)];
            assert(child->ai >= parent->ai);
        }
    }

    // nan array
    for (i=0; i<mm->n_n; i++) {
         assert(mm->n_array[i]->idx == i);
         assert(mm->n_array[i]->ai != mm->n_array[i]->ai);
    }

    // handle
    assert(mm->window >= mm->n_s + mm->n_l + mm->n_n);
    assert(mm->min_count <= mm->window);
    if (mm->n_s == 0) {
        assert(mm->s_first_leaf == 0);
    } else {
        assert(mm->s_first_leaf == FIRST_LEAF(mm->n_s));
    }
    if (mm->n_l == 0) {
        assert(mm->l_first_leaf == 0);
    } else {
        assert(mm->l_first_leaf == FIRST_LEAF(mm->n_l));
    }
    ndiff = (int)mm->n_s - (int)mm->n_l;
    if (ndiff < 0) {
        ndiff *= -1;
    }
    assert(ndiff <= 1);

    if (mm->n_s > 0 && mm->n_l > 0) {
        assert(mm->s_heap[0]->ai <= mm->l_heap[0]->ai);
    }
}


/* Print the two heaps to the screen */
void mm_dump(mm_handle *mm)
{
    int i;
    idx_t idx;

    if (!mm) {
        printf("mm is empty");
        return;
    }

    printf("\nhandle\n");
    printf("\t%2d window\n", mm->window);
    printf("\t%2d n_s\n", mm->n_s);
    printf("\t%2d n_l\n", mm->n_l);
    printf("\t%2d n_n\n", mm->n_n);
    printf("\t%2d min_count\n", mm->min_count);
    printf("\t%2d s_first_leaf\n", mm->s_first_leaf);
    printf("\t%2d l_first_leaf\n", mm->l_first_leaf);

    if (NUM_CHILDREN == 2) {

        // binary heap

        int idx0;
        int idx1;

        printf("\nsmall heap\n");
        idx0 = -1;
        if (mm->oldest->region == SH) {
            idx0 = mm->oldest->idx;
        }
        idx1 = -1;
        if (mm->newest->region == SH) {
            idx1 = mm->newest->idx;
        }
        mm_print_binary_heap(mm->s_heap, mm->n_s, idx0, idx1);
        printf("\nlarge heap\n");
        idx0 = -1;
        if (mm->oldest->region == LH) {
            idx0 = mm->oldest->idx;
        }
        idx1 = -1;
        if (mm->newest->region == LH) {
            idx1 = mm->newest->idx;
        }
        mm_print_binary_heap(mm->l_heap, mm->n_l, idx0, idx1);
        printf("\nnan array\n");
        idx0 = -1;
        if (mm->oldest->region == NA) {
            idx0 = mm->oldest->idx;
        }
        idx1 = -1;
        if (mm->newest->region == NA) {
            idx1 = mm->newest->idx;
        }
        for(i = 0; i < (int)mm->n_n; ++i) {
            idx = mm->n_array[i]->idx;
            if (i == idx0 && i == idx1) {
                printf("\t%i >%f<\n", idx, mm->n_array[i]->ai);
            } else if (i == idx0) {
                printf("\t%i >%f\n", idx, mm->n_array[i]->ai);
            } else if (i == idx1) {
                printf("\t%i  %f<\n", idx, mm->n_array[i]->ai);
            } else {
                printf("\t%i  %f\n", idx, mm->n_array[i]->ai);
            }
        }

    } else {

        // not a binary heap

        if (mm->oldest)
            printf("\n\nFirst: %f\n", (double)mm->oldest->ai);
        if (mm->newest)
            printf("Last: %f\n", (double)mm->newest->ai);

        printf("\n\nSmall heap:\n");
        for(i = 0; i < (int)mm->n_s; ++i) {
            printf("%i %f\n", (int)mm->s_heap[i]->idx, mm->s_heap[i]->ai);
        }
        printf("\n\nLarge heap:\n");
        for(i = 0; i < (int)mm->n_l; ++i) {
            printf("%i %f\n", (int)mm->l_heap[i]->idx, mm->l_heap[i]->ai);
        }
        printf("\n\nNaN heap:\n");
        for(i = 0; i < (int)mm->n_n; ++i) {
            printf("%i %f\n", (int)mm->n_array[i]->idx, mm->n_array[i]->ai);
        }
    }
}


/* Code to print a binary tree from http://stackoverflow.com/a/13755783
 * Code modified for bottleneck's needs. */
void
mm_print_binary_heap(mm_node **heap, idx_t n_array, idx_t oldest_idx,
                     idx_t newest_idx)
{
    const int line_width = 77;
    int print_pos[n_array];
    int i, j, k, pos, x=1, level=0;

    print_pos[0] = 0;
    for(i=0,j=1; i<(int)n_array; i++,j++) {
        pos = print_pos[(i-1)/2];
        pos +=  (i%2?-1:1)*(line_width/(pow(2,level+1))+1);

        for (k=0; k<pos-x; k++) printf("%c",i==0||i%2?' ':'-');
        if (i == (int)oldest_idx) {
            printf(">%.2f", heap[i]->ai);
        } else if (i == (int)newest_idx) {
            printf("%.2f<", heap[i]->ai);
        } else {
            printf("%.2f", heap[i]->ai);
        }

        print_pos[i] = x = pos+1;
        if (j==pow(2,level)) {
            printf("\n");
            level++;
            x = 1;
            j = 0;
        }
    }
}


void mm_print_line(void)
{
    int i, width = 70;
    for (i=0; i < width; i++)
        printf("-");
    printf("\n");
}
