#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

typedef size_t idx_t;
typedef double ai_t;

/*
 * The number of children has a maximum of 8 due to the manual loop-
 * unrolling used in the code below.
 */
const int NUM_CHILDREN = 2;

// Minimum of two numbers.
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Find indices of parent and first child
#define P_IDX(i) ((i) - 1) / NUM_CHILDREN
#define FC_IDX(i) NUM_CHILDREN * (i) + 1


/*
-----------------------------------------------------------------------------
  Node and handle structs (no NaN handling)
-----------------------------------------------------------------------------
*/

struct _zz_node {
    int              small; // 1 if the node is in the small heap
    idx_t            idx;   // The node's index in the heap array
    ai_t             ai;    // The node's value
    struct _zz_node *next;  // The next node in order of insertion
};
typedef struct _zz_node zz_node;

struct _zz_handle {
    idx_t     window;    // window size
    idx_t     n_s;       // The number of elements in the small heap
    idx_t     n_l;       // The number of elements in the large heap
    idx_t     n_n;
    idx_t     min_count; // Same meaning as in bn.move_median
    zz_node **s_heap;    // The max heap
    zz_node **l_heap;    // The min heap
    zz_node **n_heap;
    zz_node **nodes;     // All nodes. s_heap and l_heap point into this array
    zz_node  *node_data; // Pointer to memory location where nodes live
    zz_node  *nan_data;
    zz_node  *oldest;    // The oldest node
    zz_node  *newest;    // The newest node (most recent insert)
    idx_t s_first_leaf;  // All nodes at this index or greater are leaf nodes
    idx_t l_first_leaf;  // All nodes at this index or greater are leaf nodes
};
typedef struct _zz_handle zz_handle;

/*
-----------------------------------------------------------------------------
  Prototype
-----------------------------------------------------------------------------
*/

// top-level functions
inline zz_handle *zz_new(const idx_t window, idx_t min_count);
inline ai_t zz_update_init(zz_handle *zz, ai_t ai);
inline ai_t zz_update(zz_handle *zz, ai_t ai);
inline void zz_reset(zz_handle *zz);
inline void zz_free(zz_handle *zz);

// helper functions
inline ai_t zz_get_median(zz_handle *zz);
inline idx_t zz_get_smallest_child(zz_node **heap, idx_t window, idx_t idx,
                                   zz_node **child);
inline idx_t zz_get_largest_child(zz_node **heap, idx_t window, idx_t idx,
                                  zz_node **child);
inline void zz_move_up_small(zz_node **heap, idx_t idx, zz_node *node,
                             idx_t p_idx, zz_node *parent);
inline void zz_move_down_small(zz_node **heap, idx_t window, idx_t idx,
                               zz_node *node);
inline void zz_move_down_large(zz_node **heap, idx_t idx, zz_node *node,
                               idx_t p_idx, zz_node *parent);
inline void zz_move_up_large(zz_node **heap, idx_t window, idx_t idx,
                             zz_node *node);
inline void zz_swap_heap_heads(zz_node **s_heap, idx_t n_s, zz_node **l_heap,
                               idx_t n_l, zz_node *s_node, zz_node *l_node);

// debug
void zz_dump(zz_handle *zz);
void zz_print_binary_heap(zz_node **heap, idx_t n_heap, idx_t oldest_idx,
                          idx_t newest_idx);
void zz_check(zz_handle *zz);
void print_chain(zz_handle *zz);
void print_line(void);


/* moving window median of 1d arrays returns output array */
ai_t *move_median(ai_t *arr, idx_t length, idx_t window, idx_t min_count)
{
    zz_handle *zz;
    ai_t *out;
    idx_t i;
    //int debug=0;

    out = malloc(length * sizeof(ai_t));
    zz = zz_new(window, min_count);
    for (i=0; i < length; i++) {
        if (i < window) {
            out[i] = zz_update_init(zz, arr[i]);
        } else {
            out[i] = zz_update(zz, arr[i]);
        }
        if (i == window) {
            print_line();
            printf("window complete; switch to zz_update\n");
        }
        print_line();
        printf("inserting ai = %f\n", arr[i]);
        print_chain(zz);
        zz_dump(zz);
        printf("\nmedian = %f\n\n", out[i]);
        zz_check(zz);
    }
    zz_free(zz);

    return out;
}


/* assert that two arrays are equal */
int assert_equal(ai_t *actual, ai_t *desired, ai_t *input, idx_t length,
                 char *err_msg)
{
    idx_t i;
    int failed = 0;

    print_line();
    printf("%s\n", err_msg);
    print_line();
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


int unit_test(void)
{
    ai_t arr_input[] = {1,  -5,   7,   0,   2,  NAN,   6,   3};
    ai_t desired[] = {1. , -2. ,  1. ,  0. ,  2. ,  1. ,  4. ,  4.5};
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

    actual = move_median(arr_input, length, window, min_count);
    failed = assert_equal(actual, desired, arr_input, length, err_msg);

    free(actual);
    free(err_msg);

    return failed;
}


int main(void)
{
    return unit_test();
}


void zz_print_node(zz_node *node)
{
    printf("\n\n%d small\n", node->small);
    printf("%d idx\n", node->idx);
    printf("%f ai\n", node->ai);
    printf("%p next\n\n", node->next);
}


/*
-----------------------------------------------------------------------------
  Top-level functions
-----------------------------------------------------------------------------
*/

/* At the start of bn.move_median two heaps are created. One heap contains the
 * small values (max heap); the other heap contains the large values
 * (min heap). And the handle contains information about the heaps. It is the
 * handle that is returned by the function. */
inline zz_handle *
zz_new(const idx_t window, idx_t min_count)
{

    zz_handle *zz = malloc(sizeof(zz_handle));
    zz->nodes = malloc(2 * window * sizeof(zz_node*));
    zz->node_data = malloc(window * sizeof(zz_node));
    zz->nan_data = malloc(window * sizeof(zz_node));

    zz->s_heap = zz->nodes;
    zz->l_heap = &zz->nodes[window / 2 + window % 2];
    zz->n_heap = &zz->nodes[window];

    zz->window = window;
    zz->min_count = min_count;

    zz_reset(zz);

    return zz;
}


/* Insert a new value, ai, into one of the heaps. Use this function when
 * the heaps contains less than window-1 values. Returns the median value.
 * Once there are window-1 values in the heap, switch to using zz_update. */
inline ai_t
zz_update_init(zz_handle *zz, ai_t ai)
{

    // local variables
    zz_node *node = NULL;
    idx_t n_s = zz->n_s;
    idx_t n_l = zz->n_l;
    idx_t n_n = zz->n_n;

    if (isnan(ai)) {
        node = &zz->nan_data[n_n];
        zz->n_heap[n_n] = node;
        node->small = 2;
        node->idx = n_n;
        node->ai = ai;
        if (n_s + n_l + n_n == 0) {
            zz->oldest = node;
        } else {
            zz->newest->next = node;
        }
        node->next = NULL;
        zz->newest = node;
        ++zz->n_n;
    } else {
        node = &zz->node_data[n_s + n_l];
        if (n_s == 0) {
            // The first node.

            zz->s_heap[0] = node;
            node->small = 1;
            node->idx = 0;
            node->ai = ai;
            if (n_s + n_l + n_n == 0) {
                zz->oldest = node;
            } else {
                zz->newest->next = node;
            }
            node->next = NULL;
            zz->newest = node;

            zz->n_s = 1;
            zz->s_first_leaf = 0;

        }
        else
        {
            // Nodes after the first.

            node->next = zz->oldest;
            zz->oldest = node;

            if (n_s > n_l)
            {
                // Add to the large heap.

                zz->l_heap[n_l] = node;
                node->small = 0;
                node->idx = n_l;

                ++zz->n_l;
                zz->l_first_leaf = ceil((zz->n_l - 1) / (double)NUM_CHILDREN);
            }
            else
            {
                // Add to the small heap.

                zz->s_heap[n_s] = node;
                node->small = 1;
                node->idx = n_s;

                ++zz->n_s;
                zz->s_first_leaf = ceil((zz->n_s - 1) / (double)NUM_CHILDREN);
            }

            zz_update(zz, ai);
        }
    }

    return zz_get_median(zz);
}

/*
 * Swap nodes.
 */
#define MM_SWAP_NODES(heap, idx1, node1, idx2, node2) \
heap[idx1] = node2;                              \
heap[idx2] = node1;                              \
node1->idx = idx2;                               \
node2->idx = idx1;                               \
idx1       = idx2


/* Insert a new value, ai, into the double heap structure. Use this function
 * when the double heap contains at least window-1 values. Returns the median
 * value. If there are less than window-1 values in the heap, use
 * zz_update_init. */
inline ai_t
zz_update(zz_handle *zz, ai_t ai)
{

    // Nodes and indices.
    zz_node *node = zz->oldest;

    // and update oldest, newest
    zz->oldest = zz->oldest->next;
    zz->newest->next = node;
    zz->newest = node;

    // Replace value of node
    node->ai = ai;

    // Local variables.
    idx_t idx = node->idx;

    zz_node **l_heap = zz->l_heap;
    zz_node **s_heap = zz->s_heap;
    zz_node **n_heap = zz->n_heap;
    idx_t n_s = zz->n_s;
    idx_t n_l = zz->n_l;
    idx_t n_n = zz->n_n;

    zz_node *node2;
    idx_t idx2;


    if (isnan(ai)) {

        if (node->small == 1) {

            /* Oldest node, node, is in the small heap and needs to be moved
             * to the nan array. Resulting hole in the small heap will be
             * filled with the rightmost leaf of the last row of the small
             * heap. */

            //  insert node into nan heap
            node->small = 2;
            node->idx = n_n;
            node->ai = ai;
            n_heap[n_n] = node;
            ++zz->n_n;

            // plug small heap hole
            s_heap[idx] = s_heap[n_s - 1];
            --zz->n_s;

            // reorder small heap if ne
            node = s_heap[idx];

            // Internal or leaf node.
            if (idx > 0) {
                idx2 = P_IDX(idx);
                node2 = s_heap[idx2];

                // Move up.
                if (ai > node2->ai) {
                    zz_move_up_small(s_heap, idx, node, idx2, node2);

                    // Maybe swap between heaps.
                    node2 = l_heap[0];
                    if ((node2 != NULL) && (ai > node2->ai)) {
                        zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
                    }
                }

                // Move down.
                else if (idx < zz->s_first_leaf) {
                    zz_move_down_small(s_heap, n_s, idx, node);
                }
            }

            // Head node.
            else {
                node2 = l_heap[0];
                if (ai > node2->ai) {
                    zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
                } else {
                    zz_move_down_small(s_heap, n_s, idx, node);
                }
            }

        } else if (node->small == 0) {

            /* Oldest node, node, is in the large heap and needs to be moved
             * to the nan array. Resulting hole in the large heap will be
             * filled with the rightmost leaf of the last row of the large
             * heap. */

            //  insert node into nan heap
            node->small = 2;
            node->idx = n_n;
            node->ai = ai;
            n_heap[n_n] = node;
            ++zz->n_n;

            // plug small heap hole
            l_heap[idx] = l_heap[n_l - 1];
            --zz->n_l;

            if (zz->n_l < zz->n_s - 1) {
                // move a node from the small heap to the large

                node2 = zz->s_heap[0];
                node2->idx = zz->n_l;
                node2->small = 0;
                l_heap[zz->n_l] = node2;
                ++zz->n_l;
                zz->l_first_leaf = ceil((zz->n_l - 1) / (double)NUM_CHILDREN);
                // TODO reorder heap

                node2= zz->s_heap[zz->n_s - 1];
                node2->idx = 0;
                s_heap[0] = node2;
                --zz->n_s;
                zz->s_first_leaf = ceil((zz->n_s - 1) / (double)NUM_CHILDREN);
                // TODO reorder heap
            }

            // reorder small heap if needed
            node = l_heap[idx];

            // Internal or leaf node.
            if (idx > 0) {
                idx2 = P_IDX(idx);
                node2 = l_heap[idx2];

                // Move down.
                if (ai < node2->ai) {
                    zz_move_down_large(l_heap, idx, node, idx2, node2);

                    // Maybe swap between heaps.
                    node2 = s_heap[0];
                    if (ai < node2->ai) {
                        zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
                    }
                }

                // Move up.
                else if (idx < zz->l_first_leaf) {
                    zz_move_up_large(l_heap, n_l, idx, node);
                }
            }

            // Head node.
            else {
                node2 = s_heap[0];
                if (ai < node2->ai) {
                    zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
                } else {
                    zz_move_up_large(l_heap, n_l, idx, node);
                }
            }

        } else if (node->small == 2) {

            //  insert node into nan heap
            n_heap[idx] = node;

        }
    } else {

        // In small heap.
        if (node->small == 1) {

            // Internal or leaf node.
            if (idx > 0) {
                idx2 = P_IDX(idx);
                node2 = s_heap[idx2];

                // Move up.
                if (ai > node2->ai) {
                    zz_move_up_small(s_heap, idx, node, idx2, node2);

                    // Maybe swap between heaps.
                    node2 = l_heap[0];
                    if ((node2 != NULL) && (ai > node2->ai)) {
                        zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
                    }
                }

                // Move down.
                else if (idx < zz->s_first_leaf) {
                    zz_move_down_small(s_heap, n_s, idx, node);
                }
            }

            // Head node.
            else {
                node2 = l_heap[0];
                if (ai > node2->ai) {
                    zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
                } else {
                    zz_move_down_small(s_heap, n_s, idx, node);
                }
            }
        }

        // In large heap.
        else if (node->small == 0) {

            // Internal or leaf node.
            if (idx > 0) {
                idx2 = P_IDX(idx);
                node2 = l_heap[idx2];

                // Move down.
                if (ai < node2->ai) {
                    zz_move_down_large(l_heap, idx, node, idx2, node2);

                    // Maybe swap between heaps.
                    node2 = s_heap[0];
                    if (ai < node2->ai) {
                        zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
                    }
                }

                // Move up.
                else if (idx < zz->l_first_leaf) {
                    zz_move_up_large(l_heap, n_l, idx, node);
                }
            }

            // Head node.
            else {
                node2 = s_heap[0];
                if (ai < node2->ai) {
                    zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
                } else {
                    zz_move_up_large(l_heap, n_l, idx, node);
                }
            }
        }

        else {
            // ai is not NaN and oldest node is in nan array

            if (n_s > n_l) {
                // insert into large heap

                node->small = 0;
                node->idx = n_l;
                node->ai = ai;
                l_heap[n_l] = node;
                ++zz->n_l;

                // plug nan array hole
                n_heap[idx] = n_heap[n_n - 1];
                n_heap[idx]->idx = idx;
                --zz->n_n;

                // reorder large heap if needed

                idx = n_l;
                // Internal or leaf node.
                if (idx > 0) {
                    idx2 = P_IDX(idx);
                    node2 = l_heap[idx2];

                    // Move down.
                    if (ai < node2->ai) {
                        zz_move_down_large(l_heap, idx, node, idx2, node2);

                        // Maybe swap between heaps.
                        node2 = s_heap[0];
                        if (ai < node2->ai) {
                            zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
                        }
                    }

                    // Move up.
                    else if (idx < zz->l_first_leaf) {
                        zz_move_up_large(l_heap, n_l, idx, node);
                    }
                }

                // Head node.
                else {
                    node2 = s_heap[0];
                    if (ai < node2->ai) {
                        zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
                    } else {
                        zz_move_up_large(l_heap, n_l, idx, node);
                    }
                }

            } else {
                // insert into small heap

                node->small = 1;
                node->idx = n_s;
                node->ai = ai;
                s_heap[n_s] = node;
                ++zz->n_s;

                // plug nan array hole
                n_heap[idx] = n_heap[n_n - 1];
                n_heap[idx]->idx = idx;
                --zz->n_n;

                // reorder small heap if needed

                idx = n_s;
                // Internal or leaf node.
                if (idx > 0) {
                    idx2 = P_IDX(idx);
                    node2 = s_heap[idx2];

                    // Move up.
                    if (ai > node2->ai) {
                        zz_move_up_small(s_heap, idx, node, idx2, node2);

                        // Maybe swap between heaps.
                        node2 = l_heap[0];
                        if ((node2 != NULL) && (ai > node2->ai)) {
                            zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
                        }
                    }

                    // Move down.
                    else if (idx < zz->s_first_leaf) {
                        zz_move_down_small(s_heap, n_s, idx, node);
                    }
                }

                // Head node.
                else {
                    node2 = l_heap[0];
                    if (ai > node2->ai) {
                        zz_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
                    } else {
                        zz_move_down_small(s_heap, n_s, idx, node);
                    }
                }

            }
        }

    }

    return zz_get_median(zz);
}


/* At the end of each slice the double heap is reset (zz_reset) to prepare
 * for the next slice. In the 2d input array case (with axis=1), each slice
 * is a row of the input array. */
inline void
zz_reset(zz_handle *zz)
{
    zz->n_l = 0;
    zz->n_s = 0;
    zz->n_n = 0;
    zz->oldest = NULL;
    zz->newest = NULL;
}


/*  After bn.move_median is done, free the memory */
inline void
zz_free(zz_handle *zz)
{
    free(zz->nan_data);
    free(zz->node_data);
    free(zz->nodes);
    free(zz);
}


/*
-----------------------------------------------------------------------------
  Utility functions
-----------------------------------------------------------------------------
*/

/* Return the current median value when there are less than window values
 * in the double heap. */
inline ai_t
zz_get_median(zz_handle *zz)
{
    idx_t n_s = zz->n_s;
    idx_t n_l = zz->n_l;

    idx_t numel_total = n_l + n_s;

    if (numel_total < zz->min_count)
        return NAN;

    idx_t effective_window_size = min(zz->window, numel_total);

    if (effective_window_size % 2 == 1) {
        if (n_l > n_s)
            return zz->l_heap[0]->ai;
        else
            return zz->s_heap[0]->ai;
    }
    else
        return (zz->s_heap[0]->ai + zz->l_heap[0]->ai) / 2;
}


/*
 * Return the index of the smallest child of the node. The pointer
 * child will also be set.
 */
inline idx_t
zz_get_smallest_child(zz_node **heap, idx_t window, idx_t idx, zz_node **child)
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
zz_get_largest_child(zz_node **heap, idx_t window, idx_t idx, zz_node **child)
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



/*
 * Move the given node up through the heap to the appropriate position.
 */
inline void
zz_move_up_small(zz_node **heap, idx_t idx, zz_node *node, idx_t p_idx,
                 zz_node *parent)
{
    do {
        MM_SWAP_NODES(heap, idx, node, p_idx, parent);
        if (idx == 0) {
            break;
        }
        p_idx = P_IDX(idx);
        parent = heap[p_idx];
    } while (node->ai > parent->ai);
}


/*
 * Move the given node down through the heap to the appropriate position.
 */
inline void
zz_move_down_small(zz_node **heap, idx_t window, idx_t idx, zz_node *node)
{
    zz_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = zz_get_largest_child(heap, window, idx, &child);

    while (ai < child->ai) {
        MM_SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = zz_get_largest_child(heap, window, idx, &child);
    }
}


/*
 * Move the given node down through the heap to the appropriate
 * position.
 */
inline void
zz_move_down_large(zz_node **heap, idx_t idx, zz_node *node, idx_t p_idx,
                   zz_node *parent)
{
    do {
        MM_SWAP_NODES(heap, idx, node, p_idx, parent);
        if (idx == 0) {
            break;
        }
        p_idx = P_IDX(idx);
        parent = heap[p_idx];
    } while (node->ai < parent->ai);
}



/*
 * Move the given node up through the heap to the appropriate position.
 */
inline void
zz_move_up_large(zz_node **heap, idx_t window, idx_t idx, zz_node *node)
{
    zz_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = zz_get_smallest_child(heap, window, idx, &child);

    while (ai > child->ai) {
        MM_SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = zz_get_smallest_child(heap, window, idx, &child);
    }
}


/*
 * Swap the heap heads.
 */
inline void
zz_swap_heap_heads(zz_node **s_heap, idx_t n_s, zz_node **l_heap, idx_t n_l,
                   zz_node *s_node, zz_node *l_node)
{
    s_node->small = 0;
    l_node->small = 1;
    s_heap[0] = l_node;
    l_heap[0] = s_node;
    zz_move_down_small(s_heap, n_s, 0, l_node);
    zz_move_up_large(l_heap, n_l, 0, s_node);
}


/*
-----------------------------------------------------------------------------
  Debug utilities
-----------------------------------------------------------------------------
*/


void print_chain(zz_handle *zz)
{
    idx_t i;
    zz_node *node;

    printf("\nchain\n");
    node = zz->oldest;
    printf("\t%6.2f region %d idx %d addr %p\n", node->ai, node->small,
           node->idx, node);
    for (i=1; i < zz->n_s + zz->n_l + zz->n_n; i++) {
        node = node->next;
        printf("\t%6.2f region %d idx %d addr %p\n", node->ai, node->small,
               node->idx, node);
    }
}


void zz_check(zz_handle *zz)
{

    int ndiff;
    idx_t i;
    zz_node *child;
    zz_node *parent;

    // small heap
    for (i=0; i<zz->n_s; i++) {
         if (isnan(zz->s_heap[i]->ai)) {
            printf(">>>>>>>>> small heap contains NaN <<<<<<<<\n");
         }
    }
    for (i=1; i<zz->n_s; i++) {
        child = zz->s_heap[i];
        parent = zz->s_heap[P_IDX(child->idx)];
        if (child->ai > parent->ai) {
           printf("-----------> small heap is BAD <---------\n");
        }
    }

    // large heap
    for (i=0; i<zz->n_l; i++) {
         if (isnan(zz->l_heap[i]->ai)) {
            printf(">>>>>>>>> large heap contains NaN <<<<<<<<\n");
         }
    }
    for (i=1; i<zz->n_l; i++) {
        child = zz->l_heap[i];
        parent = zz->l_heap[P_IDX(child->idx)];
        if (child->ai < parent->ai) {
           printf("-----------> large heap is BAD <---------\n");
        }
    }

    // nan array
    for (i=0; i<zz->n_n; i++) {
         if (!isnan(zz->n_heap[i]->ai)) {
            printf(">>>>>>>>> nan array contains non-NaN <<<<<<<<\n");
         }
    }

    // handle
    assert(zz->window >= zz->n_s + zz->n_l + zz->n_n);
    assert(zz->min_count <= zz->window);
    if (zz->n_s == 0) {
        assert(zz->s_first_leaf == 0);
    } else {
        assert(zz->s_first_leaf == ceil((zz->n_s - 1) / (double)NUM_CHILDREN));
    }
    if (zz->n_l == 0) {
        assert(zz->l_first_leaf == 0);
    } else {
        assert(zz->l_first_leaf == ceil((zz->n_l - 1) / (double)NUM_CHILDREN));
    }
    ndiff = (int)zz->n_s - (int)zz->n_l;
    if (ndiff < 0) {
        ndiff *= -1;
    }
    assert(ndiff <= 1);

    if (zz->n_s > 0 && zz->n_l > 0) {
        assert(zz->s_heap[0]->ai <= zz->l_heap[0]->ai);
    }
}


/*
 * Print the two heaps to the screen.
 */
void zz_dump(zz_handle *zz)
{
    int i;
    idx_t idx;

    if (!zz) {
        printf("zz is empty");
        return;
    }

    printf("\nhandle\n");
    printf("\t%2d window\n", zz->window);
    printf("\t%2d n_s\n", zz->n_s);
    printf("\t%2d n_l\n", zz->n_l);
    printf("\t%2d n_n\n", zz->n_n);
    printf("\t%2d min_count\n", zz->min_count);
    printf("\t%2d s_first_leaf\n", zz->s_first_leaf);
    printf("\t%2d l_first_leaf\n", zz->l_first_leaf);

    if (NUM_CHILDREN == 2) {

        // binary heap

        int idx0;
        int idx1;

        printf("\nsmall heap\n");
        idx0 = -1;
        if (zz->oldest->small == 1) {
            idx0 = zz->oldest->idx;
        }
        idx1 = -1;
        if (zz->newest->small == 1) {
            idx1 = zz->newest->idx;
        }
        zz_print_binary_heap(zz->s_heap, zz->n_s, idx0, idx1);
        printf("\nlarge heap\n");
        idx0 = -1;
        if (zz->oldest->small == 0) {
            idx0 = zz->oldest->idx;
        }
        idx1 = -1;
        if (zz->newest->small == 0) {
            idx1 = zz->newest->idx;
        }
        zz_print_binary_heap(zz->l_heap, zz->n_l, idx0, idx1);
        printf("\nnan array\n");
        idx0 = -1;
        if (zz->oldest->small == 2) {
            idx0 = zz->oldest->idx;
        }
        idx1 = -1;
        if (zz->newest->small == 2) {
            idx1 = zz->newest->idx;
        }
        for(i = 0; i < (int)zz->n_n; ++i) {
            idx = zz->n_heap[i]->idx;
            if (i == idx0 && i == idx1) {
                printf("\t%i >%f<\n", idx, zz->n_heap[i]->ai);
            } else if (i == idx0) {
                printf("\t%i >%f\n", idx, zz->n_heap[i]->ai);
            } else if (i == idx1) {
                printf("\t%i  %f<\n", idx, zz->n_heap[i]->ai);
            } else {
                printf("\t%i  %f\n", idx, zz->n_heap[i]->ai);
            }
        }

    } else {

        // not a binary heap

        if (zz->oldest)
            printf("\n\nFirst: %f\n", (double)zz->oldest->ai);
        if (zz->newest)
            printf("Last: %f\n", (double)zz->newest->ai);

        printf("\n\nSmall heap:\n");
        for(i = 0; i < (int)zz->n_s; ++i) {
            printf("%i %f\n", (int)zz->s_heap[i]->idx, zz->s_heap[i]->ai);
        }
        printf("\n\nLarge heap:\n");
        for(i = 0; i < (int)zz->n_l; ++i) {
            printf("%i %f\n", (int)zz->l_heap[i]->idx, zz->l_heap[i]->ai);
        }
        printf("\n\nNaN heap:\n");
        for(i = 0; i < (int)zz->n_n; ++i) {
            printf("%i %f\n", (int)zz->n_heap[i]->idx, zz->n_heap[i]->ai);
        }
    }
}


/* Code to print a binary tree from http://stackoverflow.com/a/13755783
 * Code modified for bottleneck's needs. */
void
zz_print_binary_heap(zz_node **heap, idx_t n_heap, idx_t oldest_idx,
                     idx_t newest_idx)
{
    const int line_width = 77;
    int print_pos[n_heap];
    int i, j, k, pos, x=1, level=0;

    print_pos[0] = 0;
    for(i=0,j=1; i<(int)n_heap; i++,j++) {
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


void print_line(void)
{
    int i, width = 70;
    for (i=0; i < width; i++)
        printf("-");
    printf("\n");
}
