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

#endif


/*
-----------------------------------------------------------------------------
  Node and handle structs
-----------------------------------------------------------------------------
*/

struct _ww_node {
    int              region; // 0 for large heap; 1 small; 2 nan array
    idx_t            idx;    // The node's index in the heap or nan array
    ai_t             ai;     // The node's value
    struct _ww_node *next;   // The next node in order of insertion
};
typedef struct _ww_node ww_node;

struct _ww_handle {
    idx_t     window;    // window size
    idx_t     n_s;       // The number of elements in the small heap
    idx_t     n_l;       // The number of elements in the large heap
    idx_t     n_n;       // The number of nodes in the nan array
    idx_t     min_count; // Same meaning as in bn.move_median
    ww_node **s_heap;    // The max heap
    ww_node **l_heap;    // The min heap
    ww_node **n_array;   // The nan array
    ww_node **nodes;     // All nodes. s_heap and l_heap point into this array
    ww_node  *node_data; // Pointer to memory location where nodes live
    ww_node  *nan_data;  // Pointer to memory location where nodes live
    ww_node  *oldest;    // The oldest node
    ww_node  *newest;    // The newest node (most recent insert)
    idx_t s_first_leaf;  // All nodes at this index or greater are leaf nodes
    idx_t l_first_leaf;  // All nodes at this index or greater are leaf nodes
};
typedef struct _ww_handle ww_handle;


/*
-----------------------------------------------------------------------------
  Prototypes
-----------------------------------------------------------------------------
*/

// top-level functions
inline ww_handle *ww_new(const idx_t window, idx_t min_count);
inline ai_t ww_update_init(ww_handle *ww, ai_t ai);
inline ai_t ww_update(ww_handle *ww, ai_t ai);
inline void ww_reset(ww_handle *ww);
inline void ww_free(ww_handle *ww);

// helper functions
inline ai_t ww_get_median(ww_handle *ww);
inline void heapify_small_node(ww_handle *ww, idx_t idx);
inline void heapify_large_node(ww_handle *ww, idx_t idx);
inline idx_t ww_get_smallest_child(ww_node **heap, idx_t window, idx_t idx,
                                   ww_node **child);
inline idx_t ww_get_largest_child(ww_node **heap, idx_t window, idx_t idx,
                                  ww_node **child);
inline void ww_move_up_small(ww_node **heap, idx_t idx, ww_node *node,
                             idx_t p_idx, ww_node *parent);
inline void ww_move_down_small(ww_node **heap, idx_t window, idx_t idx,
                               ww_node *node);
inline void ww_move_down_large(ww_node **heap, idx_t idx, ww_node *node,
                               idx_t p_idx, ww_node *parent);
inline void ww_move_up_large(ww_node **heap, idx_t window, idx_t idx,
                             ww_node *node);
inline void ww_swap_heap_heads(ww_node **s_heap, idx_t n_s, ww_node **l_heap,
                               idx_t n_l, ww_node *s_node, ww_node *l_node);

// debug
ai_t *ww_move_median(ai_t *arr, idx_t length, idx_t window, idx_t min_count);
int ww_assert_equal(ai_t *actual, ai_t *desired, ai_t *input, idx_t length,
                    char *err_msg);
int ww_unit_test(void);
void ww_dump(ww_handle *ww);
void ww_print_binary_heap(ww_node **heap, idx_t n_array, idx_t oldest_idx,
                          idx_t newest_idx);
void ww_check(ww_handle *ww);
void ww_print_chain(ww_handle *ww);
void ww_print_line(void);
void ww_print_node(ww_node *node);


/*
-----------------------------------------------------------------------------
  Top-level functions
-----------------------------------------------------------------------------
*/

/* At the start of bn.move_median two heaps are created. One heap contains the
 * small values (max heap); the other heap contains the large values
 * (min heap). And the handle contains information about the heaps. It is the
 * handle that is returned by the function. */
inline ww_handle *
ww_new(const idx_t window, idx_t min_count)
{

    ww_handle *ww = malloc(sizeof(ww_handle));
    ww->nodes = malloc(2 * window * sizeof(ww_node*));
    ww->node_data = malloc(window * sizeof(ww_node));
    ww->nan_data = malloc(window * sizeof(ww_node));

    ww->s_heap = ww->nodes;
    ww->l_heap = &ww->nodes[window / 2 + window % 2];
    ww->n_array = &ww->nodes[window];

    ww->window = window;
    ww->min_count = min_count;

    ww_reset(ww);

    return ww;
}


/* Insert a new value, ai, into one of the heaps. Use this function when
 * the heaps contains less than window-1 values. Returns the median value.
 * Once there are window-1 values in the heap, switch to using ww_update. */
inline ai_t
ww_update_init(ww_handle *ww, ai_t ai)
{

    // local variables
    ww_node *node = NULL;
    idx_t n_s = ww->n_s;
    idx_t n_l = ww->n_l;
    idx_t n_n = ww->n_n;

    if (isnan(ai)) {
        node = &ww->nan_data[n_n];
        ww->n_array[n_n] = node;
        node->region = 2;
        node->idx = n_n;
        node->ai = ai;
        if (n_s + n_l + n_n == 0) {
            ww->oldest = node;
        } else {
            ww->newest->next = node;
        }
        node->next = NULL;
        ww->newest = node;
        ++ww->n_n;
    } else {
        node = &ww->node_data[n_s + n_l];
        if (n_s == 0) {
            // The first node.

            ww->s_heap[0] = node;
            node->region = 1;
            node->idx = 0;
            node->ai = ai;
            if (n_s + n_l + n_n == 0) {
                ww->oldest = node;
            } else {
                ww->newest->next = node;
            }
            node->next = NULL;
            ww->newest = node;

            ww->n_s = 1;
            ww->s_first_leaf = 0;

        }
        else
        {
            // Nodes after the first.

            node->next = ww->oldest;
            ww->oldest = node;

            if (n_s > n_l)
            {
                // Add to the large heap.

                ww->l_heap[n_l] = node;
                node->region = 0;
                node->idx = n_l;

                ++ww->n_l;
                ww->l_first_leaf = ceil((ww->n_l - 1) / (double)NUM_CHILDREN);
            }
            else
            {
                // Add to the small heap.

                ww->s_heap[n_s] = node;
                node->region = 1;
                node->idx = n_s;

                ++ww->n_s;
                ww->s_first_leaf = ceil((ww->n_s - 1) / (double)NUM_CHILDREN);
            }

            ww_update(ww, ai);
        }
    }

    return ww_get_median(ww);
}


/* Insert a new value, ai, into the double heap structure. Use this function
 * when the double heap contains at least window-1 values. Returns the median
 * value. If there are less than window-1 values in the heap, use
 * ww_update_init. */
inline ai_t
ww_update(ww_handle *ww, ai_t ai)
{

    // Nodes and indices.
    ww_node *node = ww->oldest;

    // and update oldest, newest
    ww->oldest = ww->oldest->next;
    ww->newest->next = node;
    ww->newest = node;

    // Local variables.
    idx_t idx = node->idx;

    ww_node **l_heap = ww->l_heap;
    ww_node **s_heap = ww->s_heap;
    ww_node **n_array = ww->n_array;
    idx_t n_s = ww->n_s;
    idx_t n_l = ww->n_l;
    idx_t n_n = ww->n_n;

    ww_node *node2;

    if (isnan(ai)) {

        if (node->region == 1) {

            /* Oldest node is in the small heap and needs to be moved
             * to the nan array. Resulting hole in the small heap will be
             * filled with the rightmost leaf of the last row of the small
             * heap. */

            // insert node into nan array
            node->region = 2;
            node->idx = n_n;
            node->ai = ai;
            n_array[n_n] = node;
            ++ww->n_n;

            // plug small heap hole
            --ww->n_s;
            if (ww->n_s == 0) {
                ww->s_first_leaf = 0;
                if (n_l > 0) {

                    // move head node from the large heap to the small heap
                    node2 = ww->l_heap[0];
                    node2->region = 1;
                    s_heap[0] = node2;
                    ww->n_s = 1;
                    ww->s_first_leaf = 0;

                    // plug hole in large heap
                    node2= ww->l_heap[ww->n_l - 1];
                    node2->idx = 0;
                    l_heap[0] = node2;
                    --ww->n_l;
                    if (ww->n_l == 0) {
                        ww->l_first_leaf = 0;
                    } else {
                        ww->l_first_leaf = ceil((ww->n_l - 1) / (double)NUM_CHILDREN);
                    }
                    heapify_large_node(ww, 0);

                }
            } else {
                s_heap[idx] = s_heap[n_s - 1];
                s_heap[idx]->idx = idx;
                ww->s_first_leaf = ceil((ww->n_s - 1) / (double)NUM_CHILDREN);
            }

            // reorder small heap if needed
            heapify_small_node(ww, idx);

        } else if (node->region == 0) {

            /* Oldest node is in the large heap and needs to be moved
             * to the nan array. Resulting hole in the large heap will be
             * filled with the rightmost leaf of the last row of the large
             * heap. */

            // insert node into nan array
            node->region = 2;
            node->idx = n_n;
            node->ai = ai;
            n_array[n_n] = node;
            ++ww->n_n;

            // plug large heap hole
            l_heap[idx] = l_heap[n_l - 1];
            l_heap[idx]->idx = idx;
            --ww->n_l;
            if (ww->n_l == 0) {
                ww->l_first_leaf = 0;
            } else {
                ww->l_first_leaf = ceil((ww->n_l - 1) / (double)NUM_CHILDREN);
            }

            if (ww->n_l < ww->n_s - 1) {

                // move head node from the small heap to the large heap
                node2 = ww->s_heap[0];
                node2->idx = ww->n_l;
                node2->region = 0;
                l_heap[ww->n_l] = node2;
                ++ww->n_l;
                ww->l_first_leaf = ceil((ww->n_l - 1) / (double)NUM_CHILDREN);
                heapify_large_node(ww, node2->idx);

                // plug hole in small heap
                node2= ww->s_heap[ww->n_s - 1];
                node2->idx = 0;
                s_heap[0] = node2;
                --ww->n_s;
                if (ww->n_s == 0) {
                    ww->s_first_leaf = 0;
                } else {
                    ww->s_first_leaf = ceil((ww->n_s - 1) / (double)NUM_CHILDREN);
                }
                heapify_small_node(ww, 0);

            }

            // reorder large heap if needed
            heapify_large_node(ww, idx);

        } else if (node->region == 2) {

            //  insert node into nan heap
            n_array[idx] = node;

        }
    } else {

        if (node->region == 1) {
            node->ai = ai;
            heapify_small_node(ww, idx);
        }
        else if (node->region == 0) {
            node->ai = ai;
            heapify_large_node(ww, idx);
        }
        else {

            // ai is not NaN but oldest node is in nan array

            if (n_s > n_l) {

                // insert into large heap
                node->region = 0;
                node->idx = n_l;
                node->ai = ai;
                l_heap[n_l] = node;
                ++ww->n_l;
                ww->l_first_leaf = ceil((ww->n_l - 1) / (double)NUM_CHILDREN);

                // plug nan array hole
                if (n_n > 2) {
                    n_array[idx] = n_array[n_n - 1];
                    n_array[idx]->idx = idx;
                }
                --ww->n_n;

                // reorder large heap if needed
                heapify_large_node(ww, n_l);

            } else {

                // insert into small heap
                node->region = 1;
                node->idx = n_s;
                node->ai = ai;
                s_heap[n_s] = node;
                ++ww->n_s;
                ww->s_first_leaf = ceil((ww->n_s - 1) / (double)NUM_CHILDREN);

                // plug nan array hole
                if (n_n > 2) {
                    n_array[idx] = n_array[n_n - 1];
                    n_array[idx]->idx = idx;
                }
                --ww->n_n;

                // reorder small heap if needed
                heapify_small_node(ww, n_s);

            }
        }

    }

    return ww_get_median(ww);
}


/* At the end of each slice the double heap is reset (ww_reset) to prepare
 * for the next slice. In the 2d input array case (with axis=1), each slice
 * is a row of the input array. */
inline void
ww_reset(ww_handle *ww)
{
    ww->n_l = 0;
    ww->n_s = 0;
    ww->n_n = 0;
    ww->oldest = NULL;
    ww->newest = NULL;
    ww->s_first_leaf = 0;
    ww->l_first_leaf = 0;
}


/*  After bn.move_median is done, free the memory */
inline void
ww_free(ww_handle *ww)
{
    free(ww->nan_data);
    free(ww->node_data);
    free(ww->nodes);
    free(ww);
}


/*
-----------------------------------------------------------------------------
  Utility functions
-----------------------------------------------------------------------------
*/

/* Return the current median value when there are less than window values
 * in the double heap. */
inline ai_t
ww_get_median(ww_handle *ww)
{
    idx_t n_s = ww->n_s;
    idx_t n_l = ww->n_l;

    idx_t numel_total = n_l + n_s;

    if (numel_total < ww->min_count)
        return NAN;

    idx_t effective_window_size = min(ww->window, numel_total);

    if (effective_window_size % 2 == 1) {
        if (n_l > n_s)
            return ww->l_heap[0]->ai;
        else
            return ww->s_heap[0]->ai;
    }
    else
        return (ww->s_heap[0]->ai + ww->l_heap[0]->ai) / 2;
}


inline void
heapify_small_node(ww_handle *ww, idx_t idx)
{
    idx_t idx2;
    ww_node *node;
    ww_node *node2;
    ww_node **s_heap;
    ww_node **l_heap;
    idx_t n_s, n_l;
    ai_t ai;

    s_heap = ww->s_heap;
    l_heap = ww->l_heap;
    node = s_heap[idx];
    n_s = ww->n_s;
    n_l = ww->n_l;
    ai = node->ai;

    // Internal or leaf node.
    if (idx > 0) {
        idx2 = P_IDX(idx);
        node2 = s_heap[idx2];

        // Move up.
        if (ai > node2->ai) {
            ww_move_up_small(s_heap, idx, node, idx2, node2);

            // Maybe swap between heaps.
            node2 = l_heap[0];
            if ((node2 != NULL) && (ai > node2->ai)) {
                ww_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
            }
        }

        // Move down.
        else if (idx < ww->s_first_leaf) {
            ww_move_down_small(s_heap, n_s, idx, node);
        }
    }

    // Head node.
    else {
        node2 = l_heap[0];
        if (n_l > 0 && ai > node2->ai) {
            ww_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
        } else {
            ww_move_down_small(s_heap, n_s, idx, node);
        }
    }
}


inline void
heapify_large_node(ww_handle *ww, idx_t idx)
{
    idx_t idx2;
    ww_node *node;
    ww_node *node2;
    ww_node **s_heap;
    ww_node **l_heap;
    idx_t n_s, n_l;
    ai_t ai;

    s_heap = ww->s_heap;
    l_heap = ww->l_heap;
    node = l_heap[idx];
    n_s = ww->n_s;
    n_l = ww->n_l;
    ai = node->ai;

    // Internal or leaf node.
    if (idx > 0) {
        idx2 = P_IDX(idx);
        node2 = l_heap[idx2];

        // Move down.
        if (ai < node2->ai) {
            ww_move_down_large(l_heap, idx, node, idx2, node2);

            // Maybe swap between heaps.
            node2 = s_heap[0];
            if (ai < node2->ai) {
                ww_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
            }
        }

        // Move up.
        else if (idx < ww->l_first_leaf) {
            ww_move_up_large(l_heap, n_l, idx, node);
        }
    }

    // Head node.
    else {
        node2 = s_heap[0];
        if (n_s > 0 && ai < node2->ai) {
            ww_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
        } else {
            ww_move_up_large(l_heap, n_l, idx, node);
        }
    }

}


/*
 * Return the index of the smallest child of the node. The pointer
 * child will also be set.
 */
inline idx_t
ww_get_smallest_child(ww_node **heap, idx_t window, idx_t idx, ww_node **child)
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
ww_get_largest_child(ww_node **heap, idx_t window, idx_t idx, ww_node **child)
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
ww_move_up_small(ww_node **heap, idx_t idx, ww_node *node, idx_t p_idx,
                 ww_node *parent)
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
ww_move_down_small(ww_node **heap, idx_t window, idx_t idx, ww_node *node)
{
    ww_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = ww_get_largest_child(heap, window, idx, &child);

    while (ai < child->ai) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = ww_get_largest_child(heap, window, idx, &child);
    }
}


/* Move the given node down through the heap to the appropriate position. */
inline void
ww_move_down_large(ww_node **heap, idx_t idx, ww_node *node, idx_t p_idx,
                   ww_node *parent)
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
ww_move_up_large(ww_node **heap, idx_t window, idx_t idx, ww_node *node)
{
    ww_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = ww_get_smallest_child(heap, window, idx, &child);

    while (ai > child->ai) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = ww_get_smallest_child(heap, window, idx, &child);
    }
}


/* Swap the heap heads. */
inline void
ww_swap_heap_heads(ww_node **s_heap, idx_t n_s, ww_node **l_heap, idx_t n_l,
                   ww_node *s_node, ww_node *l_node)
{
    s_node->region = 0;
    l_node->region = 1;
    s_heap[0] = l_node;
    l_heap[0] = s_node;
    ww_move_down_small(s_heap, n_s, 0, l_node);
    ww_move_up_large(l_heap, n_l, 0, s_node);
}


/*
-----------------------------------------------------------------------------
  Debug utilities
-----------------------------------------------------------------------------
*/

int main(void)
{
    return ww_unit_test();
}


/* moving window median of 1d arrays returns output array */
ai_t *ww_move_median(ai_t *arr, idx_t length, idx_t window, idx_t min_count)
{
    ww_handle *ww;
    ai_t *out;
    idx_t i;

    out = malloc(length * sizeof(ai_t));
    ww = ww_new(window, min_count);
    for (i=0; i < length; i++) {
        if (i < window) {
            out[i] = ww_update_init(ww, arr[i]);
        } else {
            out[i] = ww_update(ww, arr[i]);
        }
        if (i == window) {
            ww_print_line();
            printf("window complete; switch to ww_update\n");
        }
        ww_print_line();
        printf("inserting ai = %f\n", arr[i]);
        ww_print_chain(ww);
        ww_dump(ww);
        printf("\nmedian = %f\n\n", out[i]);
        ww_check(ww);
    }
    ww_free(ww);

    return out;
}


/* assert that two arrays are equal */
int ww_assert_equal(ai_t *actual, ai_t *desired, ai_t *input, idx_t length,
                 char *err_msg)
{
    idx_t i;
    int failed = 0;

    ww_print_line();
    printf("%s\n", err_msg);
    ww_print_line();
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


int ww_unit_test(void)
{
    ai_t arr_input[] = {1,  5,  0,  4,   NAN,   6,   NAN,    NAN};
    ai_t desired[] =   {1,  3,  1,  2.5 ,  4,   4,   5,      5};
    ai_t *actual;
    int window = 3;
    int min_count = 1;
    int length;
    char *err_msg;
    int failed;

    length = sizeof(arr_input) / sizeof(*arr_input);
    err_msg = malloc(1024 * sizeof *err_msg);
    sprintf(err_msg, "move_median failed with window=%d, min_count=%d",
            window, min_count);

    actual = ww_move_median(arr_input, length, window, min_count);
    failed = ww_assert_equal(actual, desired, arr_input, length, err_msg);

    free(actual);
    free(err_msg);

    return failed;
}


void ww_print_node(ww_node *node)
{
    printf("\n\n%d small\n", node->region);
    printf("%d idx\n", node->idx);
    printf("%f ai\n", node->ai);
    printf("%p next\n\n", node->next);
}


void ww_print_chain(ww_handle *ww)
{
    idx_t i;
    ww_node *node;

    printf("\nchain\n");
    node = ww->oldest;
    printf("\t%6.2f region %d idx %d addr %p\n", node->ai, node->region,
           node->idx, node);
    for (i=1; i < ww->n_s + ww->n_l + ww->n_n; i++) {
        node = node->next;
        printf("\t%6.2f region %d idx %d addr %p\n", node->ai, node->region,
               node->idx, node);
    }
}


void ww_check(ww_handle *ww)
{

    int ndiff;
    idx_t i;
    ww_node *child;
    ww_node *parent;

    // small heap
    for (i=0; i<ww->n_s; i++) {
        assert(ww->s_heap[i]->idx == i);
        assert(ww->s_heap[i]->ai == ww->s_heap[i]->ai);
        if (i > 0) {
            child = ww->s_heap[i];
            parent = ww->s_heap[P_IDX(child->idx)];
            assert(child->ai <= parent->ai);
        }
    }

    // large heap
    for (i=0; i<ww->n_l; i++) {
        assert(ww->l_heap[i]->idx == i);
        assert(ww->l_heap[i]->ai == ww->l_heap[i]->ai);
        if (i > 0) {
            child = ww->l_heap[i];
            parent = ww->l_heap[P_IDX(child->idx)];
            assert(child->ai >= parent->ai);
        }
    }

    // nan array
    for (i=0; i<ww->n_n; i++) {
         assert(ww->n_array[i]->idx == i);
         assert(ww->n_array[i]->ai != ww->n_array[i]->ai);
    }

    // handle
    assert(ww->window >= ww->n_s + ww->n_l + ww->n_n);
    assert(ww->min_count <= ww->window);
    if (ww->n_s == 0) {
        assert(ww->s_first_leaf == 0);
    } else {
        assert(ww->s_first_leaf == ceil((ww->n_s - 1) / (double)NUM_CHILDREN));
    }
    if (ww->n_l == 0) {
        assert(ww->l_first_leaf == 0);
    } else {
        assert(ww->l_first_leaf == ceil((ww->n_l - 1) / (double)NUM_CHILDREN));
    }
    ndiff = (int)ww->n_s - (int)ww->n_l;
    if (ndiff < 0) {
        ndiff *= -1;
    }
    assert(ndiff <= 1);

    if (ww->n_s > 0 && ww->n_l > 0) {
        assert(ww->s_heap[0]->ai <= ww->l_heap[0]->ai);
    }
}


/* Print the two heaps to the screen */
void ww_dump(ww_handle *ww)
{
    int i;
    idx_t idx;

    if (!ww) {
        printf("ww is empty");
        return;
    }

    printf("\nhandle\n");
    printf("\t%2d window\n", ww->window);
    printf("\t%2d n_s\n", ww->n_s);
    printf("\t%2d n_l\n", ww->n_l);
    printf("\t%2d n_n\n", ww->n_n);
    printf("\t%2d min_count\n", ww->min_count);
    printf("\t%2d s_first_leaf\n", ww->s_first_leaf);
    printf("\t%2d l_first_leaf\n", ww->l_first_leaf);

    if (NUM_CHILDREN == 2) {

        // binary heap

        int idx0;
        int idx1;

        printf("\nsmall heap\n");
        idx0 = -1;
        if (ww->oldest->region == 1) {
            idx0 = ww->oldest->idx;
        }
        idx1 = -1;
        if (ww->newest->region == 1) {
            idx1 = ww->newest->idx;
        }
        ww_print_binary_heap(ww->s_heap, ww->n_s, idx0, idx1);
        printf("\nlarge heap\n");
        idx0 = -1;
        if (ww->oldest->region == 0) {
            idx0 = ww->oldest->idx;
        }
        idx1 = -1;
        if (ww->newest->region == 0) {
            idx1 = ww->newest->idx;
        }
        ww_print_binary_heap(ww->l_heap, ww->n_l, idx0, idx1);
        printf("\nnan array\n");
        idx0 = -1;
        if (ww->oldest->region == 2) {
            idx0 = ww->oldest->idx;
        }
        idx1 = -1;
        if (ww->newest->region == 2) {
            idx1 = ww->newest->idx;
        }
        for(i = 0; i < (int)ww->n_n; ++i) {
            idx = ww->n_array[i]->idx;
            if (i == idx0 && i == idx1) {
                printf("\t%i >%f<\n", idx, ww->n_array[i]->ai);
            } else if (i == idx0) {
                printf("\t%i >%f\n", idx, ww->n_array[i]->ai);
            } else if (i == idx1) {
                printf("\t%i  %f<\n", idx, ww->n_array[i]->ai);
            } else {
                printf("\t%i  %f\n", idx, ww->n_array[i]->ai);
            }
        }

    } else {

        // not a binary heap

        if (ww->oldest)
            printf("\n\nFirst: %f\n", (double)ww->oldest->ai);
        if (ww->newest)
            printf("Last: %f\n", (double)ww->newest->ai);

        printf("\n\nSmall heap:\n");
        for(i = 0; i < (int)ww->n_s; ++i) {
            printf("%i %f\n", (int)ww->s_heap[i]->idx, ww->s_heap[i]->ai);
        }
        printf("\n\nLarge heap:\n");
        for(i = 0; i < (int)ww->n_l; ++i) {
            printf("%i %f\n", (int)ww->l_heap[i]->idx, ww->l_heap[i]->ai);
        }
        printf("\n\nNaN heap:\n");
        for(i = 0; i < (int)ww->n_n; ++i) {
            printf("%i %f\n", (int)ww->n_array[i]->idx, ww->n_array[i]->ai);
        }
    }
}


/* Code to print a binary tree from http://stackoverflow.com/a/13755783
 * Code modified for bottleneck's needs. */
void
ww_print_binary_heap(ww_node **heap, idx_t n_array, idx_t oldest_idx,
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


void ww_print_line(void)
{
    int i, width = 70;
    for (i=0; i < width; i++)
        printf("-");
    printf("\n");
}
