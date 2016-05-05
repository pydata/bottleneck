/*
 Copyright (c) 2011 J. David Lee. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

 1. Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above
 copyright notice, this list of conditions and the following
 disclaimer in the documentation and/or other materials provided
 with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 DAMAGE.
 */

typedef npy_intp idx_t;
typedef npy_float64 ai_t;

/*
 * The number of children has a maximum of 8 due to the manual loop-
 * unrolling used in the code below.
 */
const int NUM_CHILDREN = 8;

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

struct _mm_node {
    int              small; // 1 if the node is in the small heap
    idx_t            idx;   // The node's index in the heap array
    ai_t             ai;    // The node's value
    struct _mm_node *next;  // The next node in order of insertion
};
typedef struct _mm_node mm_node;

struct _mm_handle {
    idx_t     window;    // window size
    int       odd;       // 1 if the window size is odd, 0 otherwise
    idx_t     n_s;       // The number of elements in the small heap
    idx_t     n_l;       // The number of elements in the large heap
    idx_t     min_count; // Same meaning as in bn.move_median
    mm_node **s_heap;    // The max heap
    mm_node **l_heap;    // The min heap
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
  Prototypes (no NaN handling)
-----------------------------------------------------------------------------
*/

// top-level functions
inline mm_handle *mm_new(const idx_t window, idx_t min_count);
inline ai_t mm_update_init(mm_handle *mm, ai_t ai);
inline ai_t mm_update(mm_handle *mm, ai_t ai);
inline void mm_reset(mm_handle *mm);
inline void mm_free(mm_handle *mm);

// helper functions
inline ai_t mm_get_median_init(mm_handle *mm);
inline ai_t mm_get_median(mm_handle *mm);
inline idx_t mm_get_smallest_child(mm_node **heap, idx_t window, idx_t idx,
                                   mm_node **child);
inline idx_t mm_get_largest_child(mm_node **heap, idx_t window, idx_t idx,
                                  mm_node **child);
inline void mm_move_up_small(mm_node **heap, idx_t window, idx_t idx,
                             mm_node *node, idx_t p_idx, mm_node *parent);
inline void mm_move_down_small(mm_node **heap, idx_t window, idx_t idx,
                               mm_node *node);
inline void mm_move_down_large(mm_node **heap, idx_t window, idx_t idx,
                               mm_node *node, idx_t p_idx, mm_node *parent);
inline void mm_move_up_large(mm_node **heap, idx_t window, idx_t idx,
                             mm_node *node);
inline void mm_swap_heap_heads(mm_node **s_heap, idx_t n_s, mm_node **l_heap,
                               idx_t n_l, mm_node *s_node, mm_node *l_node);

// debug
void mm_dump(mm_handle *mm);
void print_binary_heap(mm_node **heap, idx_t n_heap, idx_t oldest_idx,
                       idx_t newest_idx);


/*
-----------------------------------------------------------------------------
  Top-level functions (no NaN handling)
-----------------------------------------------------------------------------
*/

/* At the start of bn.move_median two heaps are created. One heap contains the
 * small values (max heap); the other heap contains the large values
 * (min heap). And the handle contains information about the heaps. It is the
 * handle that is returned by the function. */
inline mm_handle *
mm_new(const idx_t window, idx_t min_count)
{
    // only malloc once, this guarantees cache friendly execution
    // and easier code for cleanup
    // this change was profiled to make a 5%-10% difference in performance
    char* memory_block = malloc(sizeof(mm_handle) +
                                window * (sizeof(mm_node*) + sizeof(mm_node)));
    if (memory_block == NULL)
        return NULL;

    char* curr_mem_ptr = memory_block;

    mm_handle *mm = (mm_handle*)curr_mem_ptr;

    curr_mem_ptr += sizeof(mm_handle);
    mm->nodes = (mm_node**) curr_mem_ptr;

    curr_mem_ptr += sizeof(mm_node*) * window;
    mm->node_data = (mm_node*) curr_mem_ptr;

    mm->window = window;
    mm->odd = window % 2;
    mm->s_heap = mm->nodes;
    mm->l_heap = &mm->nodes[window / 2 + window % 2];
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

    // local variables
    mm_node *node = NULL;
    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;
    // double check these in debug, to catch overflows
    //check_asserts(mm);

    node = &mm->node_data[n_s + n_l];

    if (n_s == 0) {
        // The first node.

        mm->s_heap[0] = node;
        node->small = 1;
        node->idx = 0;
        node->ai = ai;
        node->next = mm->l_heap[0];

        mm->n_s = 1;
        mm->oldest = mm->newest = node;
        mm->s_first_leaf = 0;

    }
    else
    {
        // Nodes after the first.

        node->next = mm->oldest;
        mm->oldest = node;

        if (n_s > n_l)
        {
            // Add to the large heap.

            mm->l_heap[n_l] = node;
            node->small = 0;
            node->idx = n_l;

            ++mm->n_l;
            mm->l_first_leaf = ceil((n_l - 1) / (double)NUM_CHILDREN);
        }
        else
        {
            // Add to the small heap.

            mm->s_heap[n_s] = node;
            node->small = 1;
            node->idx = n_s;

            ++mm->n_s;
            mm->s_first_leaf = ceil((n_s - 1) / (double)NUM_CHILDREN);
        }

        mm_update(mm, ai);
    }

    return mm_get_median_init(mm);
}


/* Insert a new value, ai, into the double heap structure. Use this function
 * when the double heap contains at least window-1 values. Returns the median
 * value. If there are less than window-1 values in the heap, use
 * mm_update_init. */
inline ai_t
mm_update(mm_handle *mm, ai_t ai)
{
    // Nodes and indices.
    mm_node *node = mm->oldest;

    // and update oldest, newest
    mm->oldest = mm->oldest->next;
    mm->newest->next = node;
    mm->newest = node;

    // Replace value of node
    node->ai = ai;

    // Local variables.
    idx_t idx = node->idx;

    mm_node **l_heap = mm->l_heap;
    mm_node **s_heap = mm->s_heap;
    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;

    mm_node *node2;
    idx_t idx2;

    // In small heap.
    if (node->small) {

        // Internal or leaf node.
        if (idx > 0) {
            idx2 = P_IDX(idx);
            node2 = s_heap[idx2];

            // Move up.
            if (ai > node2->ai) {
                mm_move_up_small(s_heap, n_s, idx, node, idx2, node2);

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
            if (ai > node2->ai) {
                mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
            } else {
                mm_move_down_small(s_heap, n_s, idx, node);
            }
        }
    }

    // In large heap.
    else {

        // Internal or leaf node.
        if (idx > 0) {
            idx2 = P_IDX(idx);
            node2 = l_heap[idx2];

            // Move down.
            if (ai < node2->ai) {
                mm_move_down_large(l_heap, n_l, idx, node, idx2, node2);

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
            if (ai < node2->ai) {
                mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
            } else {
                mm_move_up_large(l_heap, n_l, idx, node);
            }
        }
    }

    return mm_get_median(mm);
}


/* At the end of each slice the double heap is reset (mm_reset) to prepare
 * for the next slice. In the 2d input array case (with axis=1), each slice
 * is a row of the input array. */
inline void
mm_reset(mm_handle *mm)
{
    mm->n_l = 0;
    mm->n_s = 0;
    mm->oldest = NULL;
    mm->newest = NULL;
}


/*  After bn.move_median is done, free the memory */
inline void
mm_free(mm_handle *mm)
{
    free(mm);
}


/*
-----------------------------------------------------------------------------
  Utility functions (no NaN handling)
-----------------------------------------------------------------------------
*/

/* Return the current median value when there are less than window values
 * in the double heap. */
inline ai_t
mm_get_median_init(mm_handle *mm)
{
    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;

    //check_asserts(mm);

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


/* Return the current median value when there are at least window values
 * in the double heap. */
inline ai_t
mm_get_median(mm_handle *mm)
{
    if (mm->odd) {
        return mm->s_heap[0]->ai;
    } else {
        return (mm->s_heap[0]->ai + mm->l_heap[0]->ai) / 2.0;
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


/*
 * Swap nodes.
 */
#define MM_SWAP_NODES(heap, idx1, node1, idx2, node2) \
heap[idx1] = node2;                              \
heap[idx2] = node1;                              \
node1->idx = idx2;                               \
node2->idx = idx1;                               \
idx1       = idx2


/*
 * Move the given node up through the heap to the appropriate position.
 */
inline void
mm_move_up_small(mm_node **heap, idx_t window, idx_t idx, mm_node *node,
                 idx_t p_idx, mm_node *parent)
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
mm_move_down_small(mm_node **heap, idx_t window, idx_t idx, mm_node *node)
{
    mm_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = mm_get_largest_child(heap, window, idx, &child);

    while (ai < child->ai) {
        MM_SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = mm_get_largest_child(heap, window, idx, &child);
    }
}


/*
 * Move the given node down through the heap to the appropriate
 * position.
 */
inline void
mm_move_down_large(mm_node **heap, idx_t window, idx_t idx, mm_node *node,
                   idx_t p_idx, mm_node *parent)
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
mm_move_up_large(mm_node **heap, idx_t window, idx_t idx, mm_node *node)
{
    mm_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = mm_get_smallest_child(heap, window, idx, &child);

    while (ai > child->ai) {
        MM_SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = mm_get_smallest_child(heap, window, idx, &child);
    }
}


/*
 * Swap the heap heads.
 */
inline void
mm_swap_heap_heads(mm_node **s_heap, idx_t n_s, mm_node **l_heap, idx_t n_l,
                   mm_node *s_node, mm_node *l_node)
{
    s_node->small = 0;
    l_node->small = 1;
    s_heap[0] = l_node;
    l_heap[0] = s_node;
    mm_move_down_small(s_heap, n_s, 0, l_node);
    mm_move_up_large(l_heap, n_l, 0, s_node);
}


/*
-----------------------------------------------------------------------------
  Debug utilities (no NaN handling)
-----------------------------------------------------------------------------
*/

/*
 * Print the two heaps to the screen.
 */
void mm_dump(mm_handle *mm)
{
    if (!mm) {
        printf("mm is empty");
        return;
    }

    if (NUM_CHILDREN == 2) {

        // binary heap

        int idx0;
        int idx1;

        printf("\n\nSmall heap:\n");
        idx0 = -1;
        if (mm->oldest->small == 1) {
            idx0 = mm->oldest->idx;
        }
        idx1 = -1;
        if (mm->newest->small == 1) {
            idx1 = mm->newest->idx;
        }
        print_binary_heap(mm->s_heap, mm->n_s, idx0, idx1);
        printf("\n\nLarge heap:\n");
        idx0 = -1;
        if (mm->oldest->small == 0) {
            idx0 = mm->oldest->idx;
        }
        idx1 = -1;
        if (mm->newest->small == 0) {
            idx1 = mm->newest->idx;
        }
        print_binary_heap(mm->l_heap, mm->n_l, idx0, idx1);
    
    } else {

        // not a binary heap

        idx_t i;
        if (mm->oldest)
            printf("\n\nFirst: %f\n", (double)mm->oldest->ai);
        if (mm->newest)
            printf("Last: %f\n", (double)mm->newest->ai);

        printf("\n\nSmall heap:\n");
        for(i = 0; i < mm->n_s; ++i) {
            printf("%i %f\n", (int)mm->s_heap[i]->idx, mm->s_heap[i]->ai);
        }
        printf("\n\nLarge heap:\n");
        for(i = 0; i < mm->n_l; ++i) {
            printf("%i %f\n", (int)mm->l_heap[i]->idx, mm->l_heap[i]->ai);
        }
    }
}


/* Code to print a binary tree from http://stackoverflow.com/a/13755783
 * Code modified for bottleneck's needs. */
void
print_binary_heap(mm_node **heap, idx_t n_heap, idx_t oldest_idx,
                  idx_t newest_idx)
{
    const int line_width = 77;
    int print_pos[n_heap];
    int i, j, k, pos, x=1, level=0;

    print_pos[0] = 0;
    for(i=0,j=1; i<n_heap; i++,j++) {
        pos = print_pos[(i-1)/2];
        pos +=  (i%2?-1:1)*(line_width/(pow(2,level+1))+1);

        for (k=0; k<pos-x; k++) printf("%c",i==0||i%2?' ':'-');
        if (i == oldest_idx) {
            printf(">%.2f", heap[i]->ai);
        } else if (i == newest_idx) {
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


/*
-----------------------------------------------------------------------------
  Node and handle structs (NaN capable)
-----------------------------------------------------------------------------
*/

struct _zz_node {
    int              small; // 1 if the node is in the small heap
    idx_t            idx;   // The node's index in the heap array
    ai_t             ai;    // The node's value
    struct _zz_node *next;  // The next node in order of insertion

    // double linked list for nan tracking
    struct _zz_node *next_nan;  // The next nan node in order of insertion
    struct _zz_node *prev_nan;  // The prev nan node in order of insertion
};
typedef struct _zz_node zz_node;

struct _zz_handle {
    idx_t     window;    // window size
    idx_t     n_s_nan;   // number of nans in min heap
    idx_t     n_l_nan;   // number of nans in max heap
    idx_t     n_s;       // The number of elements in the min heap
    idx_t     n_l;       // The number of elements in the max heap
    idx_t     min_count; // Same as in bn.move_median
    zz_node **s_heap;    // The max heap
    zz_node **l_heap;    // The min heap
    zz_node **nodes;     // All nodes. s_heap and l_heap point into this array
    zz_node  *node_data; // Pointer to memory location where nodes live
    zz_node  *oldest;    // The oldest node
    zz_node  *newest;    // The newest node (most recent insert)
    idx_t s_first_leaf;  // All nodes at this index or greater are leaf nodes
    idx_t l_first_leaf;  // All nodes at this index or greater are leaf nodes

    // + and - infinity array
    zz_node  *oldest_nan_s;  // The node added first to the list of nodes
    zz_node  *newest_nan_s;  // The last (most recent) node added
    zz_node  *oldest_nan_l;  // The node added first to the list of nodes
    zz_node  *newest_nan_l;  // The last (most recent) node added

    idx_t max_s_heap_size;
};
typedef struct _zz_handle zz_handle;


/*
-----------------------------------------------------------------------------
  Prototypes (NaN capable)
-----------------------------------------------------------------------------
*/

// top-level functions
inline zz_handle *zz_new(const idx_t window, idx_t min_count);
inline ai_t zz_update_init(zz_handle *zz, ai_t ai);
inline ai_t zz_update(zz_handle *zz, ai_t ai);
inline void zz_reset(zz_handle *zz);
inline void zz_free(zz_handle *zz);

// helper functions
inline void zz_update_withnan(zz_handle *zz, ai_t ai);
inline void zz_update_nonan(zz_handle *zz, ai_t ai);
inline void zz_insert_nan(zz_handle *zz);
inline void zz_update_helper(zz_handle *zz, zz_node *node, ai_t ai);
inline void zz_update_withnan_skipevict(zz_handle *zz, ai_t ai);
inline void move_nan_helper(zz_handle *zz, zz_node* new_newest);
inline void move_nan_from_s_to_l(zz_handle *zz);
inline void move_nan_from_l_to_s(zz_handle *zz);
inline ai_t zz_get_median(zz_handle *zz);
inline idx_t get_smallest_child(zz_node **heap, idx_t window, idx_t idx,
                                zz_node **child);
inline idx_t get_largest_child(zz_node **heap, idx_t window, idx_t idx,
                               zz_node **child);
inline void move_up_small(zz_node **heap, idx_t window, idx_t idx,
                          zz_node *node, idx_t p_idx, zz_node *parent);
inline void move_down_small(zz_node **heap, idx_t window, idx_t idx,
                            zz_node *node);
inline void move_down_large(zz_node **heap, idx_t window, idx_t idx,
                            zz_node *node, idx_t p_idx, zz_node *parent);
inline void move_up_large(zz_node **heap, idx_t window, idx_t idx,
                          zz_node *node);
inline void swap_heap_heads(zz_node **s_heap, idx_t n_s, zz_node **l_heap,
                            idx_t n_l, zz_node *s_node, zz_node *l_node);

// debug
void zz_dump(zz_handle *zz);
void check_asserts(zz_handle *zz);

// --------------------------------------------------------------------------
// zz_new, zz_reset, zz_free
//
// At the start of bn.move_median a new double heap is created (zz_new). One
// heap contains the small values; the other heap contains the large values.
// And the hanlde contains information about the heaps.
//
// At the end of each slice the double heap is reset (zz_reset) to prepare
// for the next slice. In the 2d input array case (with axis=1), each slice
// is a row of the input array.
//
// After bn.move_median is done, memory is freed (zz_free).

inline zz_handle *zz_new(const idx_t window, idx_t min_count)
{
    // window -- The total number of values in the double heap.
    // Return: The zz_handle structure, uninitialized.

    // only malloc once, this guarantees cache friendly execution
    // and easier code for cleanup
    // this change was profiled to make a 5%-10% difference in performance
    char* memory_block = malloc(sizeof(zz_handle) + window * (sizeof(zz_node*)
                                + sizeof(zz_node)));

    if (memory_block == NULL)
        return NULL;

    char* curr_mem_ptr = memory_block;

    zz_handle *zz = (zz_handle*)curr_mem_ptr;

    curr_mem_ptr += sizeof(zz_handle);
    zz->nodes = (zz_node**) curr_mem_ptr;

    curr_mem_ptr += sizeof(zz_node*) * window;
    zz->node_data = (zz_node*) curr_mem_ptr;

    zz->max_s_heap_size = window / 2 + window % 2;
    zz->window = window;
    zz->s_heap = zz->nodes;
    zz->l_heap = &zz->nodes[zz->max_s_heap_size];
    zz->min_count = min_count;

    zz_reset(zz);

    return zz;
}


inline ai_t zz_update_init(zz_handle *zz, ai_t ai)
{
    /*
     * Insert initial aiues into the double heap structure.
     *
     * Arguments:
     * zz  -- The double heap structure.
     * idx -- The index of the value running from 0 to window - 1.
     * ai -- The aiue to insert.
     */

    // Some local variables.
    zz_node *node = NULL;
    idx_t n_s = zz->n_s;
    idx_t n_l = zz->n_l;
    idx_t n_s_nan = zz->n_s_nan;
    idx_t n_l_nan = zz->n_l_nan;
    // double check these in debug, to catch overflows
    //check_asserts(zz);

    int is_nan_ai = isnan(ai);

    node = &zz->node_data[n_s + n_l];
    node->next_nan = NULL;

    // The first node.
    if (n_s == 0) {
        zz->n_s_nan = is_nan_ai;

        zz->s_heap[0] = node;
        node->small = 1;
        node->idx = 0;
        node->next = zz->l_heap[0];

        zz->n_s = 1;
        zz->oldest = zz->newest = node;
        zz->s_first_leaf = 0;

        if (is_nan_ai)
        {
            node->ai = -INFINITY;
            node->next_nan = NULL;
            node->prev_nan = NULL;
            zz->oldest_nan_s = node;
            zz->newest_nan_s = node;
        }
        else
            node->ai = ai;
    }
    else
    {
        // Nodes after the first.

        if (is_nan_ai)
        {
            zz_insert_nan(zz);
            //check_asserts(zz);
        }
        else
        {
            node->next = zz->oldest;
            zz->oldest = node;

            idx_t nonnan_n_s = n_s - n_s_nan;
            idx_t nonnan_n_l = n_l - n_l_nan;

            if ((n_s == zz->max_s_heap_size) | (nonnan_n_s > nonnan_n_l))
            {
                // Add to the large heap.

                zz->l_heap[n_l] = node;
                node->small = 0;
                node->idx = n_l;

                ++zz->n_l;
                zz->l_first_leaf = ceil((n_l - 1) / (double)NUM_CHILDREN);
            }
            else
            {
                // Add to the small heap.

                zz->s_heap[n_s] = node;
                node->small = 1;
                node->idx = n_s;

                ++zz->n_s;
                zz->s_first_leaf = ceil((n_s - 1) / (double)NUM_CHILDREN);
            }

            zz_update_nonan(zz, ai);
        }
    }

    return zz_get_median(zz);
}


inline ai_t zz_update(zz_handle *zz, ai_t ai)
{
    idx_t n_s = zz->n_s;
    idx_t n_l = zz->n_l;
    idx_t n_s_nan = zz->n_s_nan;
    idx_t n_l_nan = zz->n_l_nan;
    idx_t nonnan_n_s = n_s - n_s_nan;
    idx_t nonnan_n_l = n_l - n_l_nan;

    if (isnan(ai))
    {
        // double check these in debug, to catch overflows
        //check_asserts(zz);

        // try to keep the heaps balanced, so we can try to avoid the nan
        // rebalancing penalty. makes significant difference when % of nans
        // is large and window size is also large
        zz_node* node_to_evict = zz->oldest;
        ai_t to_evict = node_to_evict->ai;
        idx_t evict_effect_s = 0;
        idx_t evict_effect_l = 0;
        if (isinf(to_evict)) {
            if (node_to_evict->small)
                evict_effect_s = 1;
            else
                evict_effect_l = 1;
        }

        if ((nonnan_n_s + evict_effect_s) > (nonnan_n_l + evict_effect_l))
            zz_update_withnan(zz, -INFINITY); // add to min heap
        else
            zz_update_withnan(zz, INFINITY); // add to max heap
    } else {
        // Note: we could still be evicting nans here, so call the nan safe
        // function

        // I tried an if-then here to call a non nan function if
        // we are not evicting nan, but penalty to check was
        // too high.  please be careful and measure before trying
        // to optimize here
        zz_update_withnan(zz, ai);
    }

    // these could've been updated, so we regrab them
    n_s_nan = zz->n_s_nan;
    n_l_nan = zz->n_l_nan;

    nonnan_n_s = n_s - n_s_nan;
    nonnan_n_l = n_l - n_l_nan;

    if (nonnan_n_l == nonnan_n_s + 2)
        move_nan_from_s_to_l(zz); // max heap is too big...
    else if (nonnan_n_s == nonnan_n_l + 2)
        move_nan_from_l_to_s(zz); // min heap is too big...

    // double check these in debug, to catch overflows
    //check_asserts(zz);

    return zz_get_median(zz);
}


inline void zz_reset(zz_handle *zz)
{
    zz->n_l = 0;
    zz->n_s = 0;
    zz->n_l_nan = 0;
    zz->n_s_nan = 0;

    zz->oldest_nan_s = NULL;
    zz->newest_nan_s = NULL;
    zz->oldest_nan_l = NULL;
    zz->newest_nan_l = NULL;
    zz->oldest = NULL;
    zz->newest = NULL;
}

inline void zz_free(zz_handle *zz)
{
    free(zz);
}

// --------------------------------------------------------------------------
// As we loop through a slice of the input array in bn.move_median, each
// array element, ai, must be inserted into the heap.
//
// If you know that the input array does not contain NaNs (e.g. integer input
// arrays) then you can use the faster zz_update_movemedian_nonan.
//
// If the input array may contain NaNs then use the slower
// zz_update_movemedian_possiblenan


// --------------------------------------------------------------------------
// Insert a new value, ai, into the double heap structure.
//
// Don't call these directly. Instead these functions are called by
// zz_update_movemedian_nonan and zz_update_movemedian_possiblenan
//
// zz_update_init is for when double heap contains less than window values
// zz_update_nonan ai is not NaN and double heap is already full
// zz_update ai might be NaN and double heap is already full

inline void zz_update_withnan(zz_handle *zz, ai_t ai) {
    // Nodes and indices.
    zz_node *node = zz->oldest;

    if (isinf(node->ai)) {
        // if we are removing a nan
        if (node->small) {
            --zz->n_s_nan;

            if (node == zz->oldest_nan_s) {
                zz_node* next_ptr = zz->oldest_nan_s->next_nan;
                zz->oldest_nan_s = next_ptr;
                if (next_ptr == NULL)
                    zz->newest_nan_s = NULL;
                else
                    next_ptr->prev_nan = NULL; /* the current nan is the first
                                                *  one */
            } else {
                assert(node->prev_nan != NULL);
                zz_node* newest_node = node->prev_nan;
                newest_node->next_nan = node->next_nan;
                if (node->next_nan == NULL)
                    zz->newest_nan_s = newest_node;
                else
                    node->next_nan->prev_nan = newest_node;
                node->next_nan = NULL;
            }
        } else {
            --zz->n_l_nan;

            if (node == zz->oldest_nan_l) {
                zz_node* next_ptr = zz->oldest_nan_l->next_nan;
                zz->oldest_nan_l = next_ptr;
                if (next_ptr == NULL)
                    zz->newest_nan_l = NULL;
                else
                    next_ptr->prev_nan = NULL; /* the current nan is the first
                                                *  one */
            } else {
                assert(node->prev_nan != NULL);
                zz_node* newest_node = node->prev_nan;
                newest_node->next_nan = node->next_nan;
                if (node->next_nan == NULL)
                    zz->newest_nan_l = newest_node;
                else
                    node->next_nan->prev_nan = newest_node;
                node->next_nan = NULL;
            }
        }
    }

    zz_update_withnan_skipevict(zz, ai);
}


inline void zz_update_nonan(zz_handle *zz, ai_t ai)
{
    // Nodes and indices.
    zz_node *node = zz->oldest;

    // and update first, newest
    zz->oldest = zz->oldest->next;
    zz->newest->next = node;
    zz->newest = node;

    zz_update_helper(zz, node, ai);
}


// --------------------------------------------------------------------------
// Helper functions for inserting new values into the heaps, i.e., updating
// the heaps.

inline void zz_insert_nan(zz_handle *zz)
    // insert a nan, during initialization phase.
{

    // Local variables.
    ai_t ai = 0;
    idx_t n_s = zz->n_s;
    idx_t n_l = zz->n_l;
    idx_t n_s_nan = zz->n_s_nan;
    idx_t n_l_nan = zz->n_l_nan;

    // Nodes and indices.
    zz_node *node = &zz->node_data[n_s + n_l];
    node->next = zz->oldest;
    zz->oldest = node;

    //check_asserts(zz);

    int l_heap_full = (n_l == (zz->window - zz->max_s_heap_size));
    int s_heap_full = (n_s == zz->max_s_heap_size);
    if ((s_heap_full | (n_s_nan > n_l_nan)) & (l_heap_full == 0)) {
        // Add to the large heap.

        zz->l_heap[n_l] = node;
        node->small = 0;
        node->idx = n_l;

        ++zz->n_l;
        zz->l_first_leaf = ceil((n_l - 1) / (double)NUM_CHILDREN);

        ai = INFINITY;
    } else {
        // Add to the small heap.

        zz->s_heap[n_s] = node;
        node->small = 1;
        node->idx = n_s;

        ++zz->n_s;
        zz->s_first_leaf = ceil((n_s - 1) / (double)NUM_CHILDREN);

        ai = -INFINITY;
    }

    zz_update_withnan_skipevict(zz, ai);
}

inline void zz_update_helper(zz_handle *zz, zz_node *node, ai_t ai)
{
    // Replace value of node
    node->ai = ai;

    // Local variables.
    idx_t    idx = node->idx;

    zz_node **l_heap = zz->l_heap;
    zz_node **s_heap = zz->s_heap;
    idx_t n_s = zz->n_s;
    idx_t n_l = zz->n_l;

    zz_node *node2;
    idx_t    idx2;


    // In small heap.
    if (node->small) {

        // Internal or leaf node.
        if (idx > 0) {
            idx2 = P_IDX(idx);
            node2 = s_heap[idx2];

            // Move up.
            if (ai > node2->ai) {
                move_up_small(s_heap, n_s, idx, node, idx2, node2);

                // Maybe swap between heaps.
                node2 = (n_l>0) ? l_heap[0] : NULL; /* needed because we
                                                     * could've only inserted
                                                     * nan and then a # */
                if ((node2 != NULL) && (ai > node2->ai)) {
                    swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
                }
            }

            // Move down.
            else if (idx < zz->s_first_leaf) {
                move_down_small(s_heap, n_s, idx, node);
            }
        }

        // Head node.
        else {
            node2 = (n_l>0) ? l_heap[0] : NULL; /* needed because we could've
                                                 *  only inserted nan and then
                                                 *  a # */
            if ((node2 != NULL) && (ai > node2->ai)) {
                swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
            } else {
                move_down_small(s_heap, n_s, idx, node);
            }
        }
    }

    // In large heap.
    else {

        // Internal or leaf node.
        if (idx > 0) {
            idx2 = P_IDX(idx);
            node2 = l_heap[idx2];

            // Move down.
            if (ai < node2->ai) {
                move_down_large(l_heap, n_l, idx, node, idx2, node2);

                // Maybe swap between heaps.
                node2 = (n_s>0) ? s_heap[0] : NULL;
                if ((node2 != NULL) && (ai < node2->ai)) {
                    swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
                }
            }

            // Move up.
            else if (idx < zz->l_first_leaf) {
                move_up_large(l_heap, n_l, idx, node);
            }
        }

        // Head node.
        else {
            node2 = (n_s>0) ? s_heap[0] : NULL;
            if ((node2 != NULL) && (ai < node2->ai)) {
                swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
            } else {
                move_up_large(l_heap, n_l, idx, node);
            }
        }
    }
}


inline void zz_update_withnan_skipevict(zz_handle *zz, ai_t ai) {
    if (isinf(ai))
    {
        zz_node *node = zz->oldest;
        // we are adding a new nan
        node->next_nan = NULL;

        if (ai>0) {
            ++zz->n_l_nan;
            if (zz->oldest_nan_l == NULL) {
                zz->oldest_nan_l = node;
                zz->newest_nan_l = node;
                node->prev_nan = NULL;
            } else {
                assert(node->next_nan == NULL);
                assert(node!=zz->newest_nan_l);
                zz->newest_nan_l->next_nan = node;
                node->prev_nan = zz->newest_nan_l;
                assert(node->next_nan == NULL);
                zz->newest_nan_l = node;
            }
        } else {
            ++zz->n_s_nan;
            if (zz->oldest_nan_s == NULL) {
                zz->oldest_nan_s = node;
                zz->newest_nan_s = node;
                node->prev_nan = NULL;
            } else {
                zz->newest_nan_s->next_nan = node;
                node->prev_nan = zz->newest_nan_s;
                zz->newest_nan_s = node;
            }
        }
    }

    zz_update_nonan(zz, ai);
}

// --------------------------------------------------------------------------

inline void move_nan_helper(zz_handle *zz, zz_node* new_newest)
{
    assert(new_newest != NULL);

    ai_t old_ai = new_newest->ai;
    ai_t new_ai = -old_ai;

    assert(isinf(old_ai));

    new_newest->ai = new_ai;
    zz_update_helper(zz, new_newest, new_ai);
}

inline void move_nan_from_s_to_l(zz_handle *zz)
{
    // move nan from s to l
    assert(zz->oldest_nan_s != NULL);

    zz_node* new_newest = zz->oldest_nan_s;
    assert(isinf(new_newest->ai));
    zz->oldest_nan_s = zz->oldest_nan_s->next_nan;

    if (zz->oldest_nan_s != NULL)
        zz->oldest_nan_s->prev_nan = NULL;
    else
        zz->newest_nan_s = NULL; // that was our last nan on this side

    new_newest->next_nan = NULL;

    if (zz->oldest_nan_l == NULL) {
        zz->oldest_nan_l = new_newest;
        new_newest->prev_nan = NULL;
    } else {
        zz->newest_nan_l->next_nan = new_newest;
        new_newest->prev_nan = zz->newest_nan_l;
    }

    zz->newest_nan_l = new_newest;

    --zz->n_s_nan;
    ++zz->n_l_nan;

    move_nan_helper(zz, new_newest);
}

inline void move_nan_from_l_to_s(zz_handle *zz)
{
    // move nan from l to s
    assert(zz->oldest_nan_l != NULL);

    zz_node* new_newest = zz->oldest_nan_l;
    assert(isinf(new_newest->ai));
    zz->oldest_nan_l = zz->oldest_nan_l->next_nan;

    if (zz->oldest_nan_l != NULL)
        zz->oldest_nan_l->prev_nan = NULL;
    else
        zz->newest_nan_l = NULL; // that was our last nan on this side

    new_newest->next_nan = NULL;

    if (zz->oldest_nan_s == NULL) {
        zz->oldest_nan_s = new_newest;
        new_newest->prev_nan = NULL;
    } else {
        zz->newest_nan_s->next_nan = new_newest;
        new_newest->prev_nan = zz->newest_nan_s;
    }

    zz->newest_nan_s = new_newest;

    --zz->n_l_nan;
    ++zz->n_s_nan;

    move_nan_helper(zz, new_newest);
}

// --------------------------------------------------------------------------


/*
 * Return the current median value.
 */
inline ai_t zz_get_median(zz_handle *zz)
{
    idx_t n_s = zz->n_s;
    idx_t n_l = zz->n_l;
    idx_t n_s_nan = zz->n_s_nan;
    idx_t n_l_nan = zz->n_l_nan;

    //check_asserts(zz);

    idx_t nonnan_n_l = n_l - n_l_nan;
    idx_t nonnan_n_s = n_s - n_s_nan;
    idx_t numel_total = nonnan_n_l + nonnan_n_s;

    if (numel_total < zz->min_count)
        return NAN;

    idx_t effective_window_size = min(zz->window, numel_total);

    if (effective_window_size % 2 == 1) {
        if (nonnan_n_l > nonnan_n_s)
            return zz->l_heap[0]->ai;
        else
            return zz->s_heap[0]->ai;
    }
    else
        return (zz->s_heap[0]->ai + zz->l_heap[0]->ai) / 2;
}


// --------------------------------------------------------------------------
// utility functions

/*
 * Return the index of the smallest child of the node. The pointer
 * child will also be set.
 */
inline idx_t get_smallest_child(zz_node **heap,
                                  idx_t     window,
                                  idx_t     idx,
                                  zz_node  **child)
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
inline idx_t get_largest_child(zz_node **heap,
                          idx_t     window,
                          idx_t     idx,
                          zz_node  **child)
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
 * Swap nodes.
 */
#define SWAP_NODES(heap, idx1, node1, idx2, node2) \
heap[idx1] = node2;                              \
heap[idx2] = node1;                              \
node1->idx = idx2;                               \
node2->idx = idx1;                               \
idx1       = idx2


/*
 * Move the given node up through the heap to the appropriate position.
 */
inline void move_up_small(zz_node **heap,
                          idx_t     window,
                          idx_t     idx,
                          zz_node  *node,
                          idx_t     p_idx,
                          zz_node  *parent)
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


/*
 * Move the given node down through the heap to the appropriate position.
 */
inline void move_down_small(zz_node **heap,
                            idx_t     window,
                            idx_t     idx,
                            zz_node  *node)
{
    zz_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = get_largest_child(heap, window, idx, &child);

    while (ai < child->ai) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = get_largest_child(heap, window, idx, &child);
    }
}


/*
 * Move the given node down through the heap to the appropriate
 * position.
 */
inline void move_down_large(zz_node **heap,
                            idx_t     window,
                            idx_t     idx,
                            zz_node  *node,
                            idx_t     p_idx,
                            zz_node  *parent)
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



/*
 * Move the given node up through the heap to the appropriate position.
 */
inline void move_up_large(zz_node **heap,
                          idx_t     window,
                          idx_t     idx,
                          zz_node  *node)
{
    zz_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = get_smallest_child(heap, window, idx, &child);

    while (ai > child->ai) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = get_smallest_child(heap, window, idx, &child);
    }
}


/*
 * Swap the heap heads.
 */
inline void swap_heap_heads(zz_node **s_heap,
                            idx_t     n_s,
                            zz_node **l_heap,
                            idx_t     n_l,
                            zz_node  *s_node,
                            zz_node  *l_node)
{
    s_node->small = 0;
    l_node->small = 1;
    s_heap[0] = l_node;
    l_heap[0] = s_node;
    move_down_small(s_heap, n_s, 0, l_node);
    move_up_large(l_heap, n_l, 0, s_node);
}

// --------------------------------------------------------------------------
// debug utilities

/*
 * Print the two heaps to the screen.
 */
void zz_dump(zz_handle *zz)
{
    if (!zz) {
        printf("zz is empty");
        return;
    }
    idx_t i;
    if (zz->oldest)
        printf("\n\nFirst: %f\n", (double)zz->oldest->ai);

    if (zz->newest)
        printf("Last: %f\n", (double)zz->newest->ai);


    printf("\n\nSmall heap:\n");
    for(i = 0; i < zz->n_s; ++i) {
        printf("%i %f\n", (int)zz->s_heap[i]->idx, zz->s_heap[i]->ai);
    }

    printf("\n\nLarge heap:\n");
    for(i = 0; i < zz->n_l; ++i) {
        printf("%i %f\n", (int)zz->l_heap[i]->idx, zz->l_heap[i]->ai);
    }
}

void check_asserts(zz_handle *zz)
{
    zz_dump(zz);
    assert(zz->n_s >= zz->n_s_nan);
    assert(zz->n_l >= zz->n_l_nan);
    idx_t valid_s = zz->n_s - zz->n_s_nan;
    idx_t valid_l = zz->n_l - zz->n_l_nan;

    // use valid_s and valid_l or get compiler warnings
    // these lines do nothing
    idx_t duzzy = valid_l + valid_s;
    if (duzzy > 100) {duzzy = 99;}

    assert(valid_s < 5000000000); //most likely an overflow
    assert(valid_l < 5000000000); //most likely an overflow

    assert(zz->n_s_nan < 5000000000); //most likely an overflow
    assert(zz->n_l_nan < 5000000000); //most likely an overflow

    if (zz->oldest_nan_l)
    {
        assert(zz->newest_nan_l);
        assert(zz->newest_nan_l->next_nan == NULL);
        assert(zz->oldest_nan_l->prev_nan == NULL);
    }
    else
        assert(zz->newest_nan_l == NULL);

    if (zz->oldest_nan_s)
    {
        assert(zz->newest_nan_s);
        assert(zz->newest_nan_s->next_nan == NULL);
        assert(zz->oldest_nan_s->prev_nan == NULL);
    }
    else
        assert(zz->newest_nan_s == NULL);

    size_t len = 0;
    zz_node* iter = zz->oldest_nan_l;
    while (iter!=NULL)
    {
        assert(isinf(iter->ai));
        assert(len <= zz->n_l);
        if (iter->next_nan != NULL)
        {
            assert(iter->prev_nan != iter->next_nan);
            assert(iter->next_nan->prev_nan == iter);
        }
        iter = iter->next_nan;
        ++len;
    }

    len = 0;
    iter = zz->oldest_nan_s;
    while (iter!=NULL)
    {
        assert(isinf(iter->ai));
        assert(len <= zz->n_s);
        if (iter->next_nan != NULL)
        {
            assert(iter->prev_nan != iter->next_nan);
            assert(iter->next_nan->prev_nan == iter);
        }
        iter = iter->next_nan;
        ++len;
    }

    // since valid_l and valid_s are signed, these will overflow and we don't
    // have to check for diffs of -5, etc.
    assert(
           ((valid_l - valid_s) <= 1)
           || ((valid_s - valid_l) <= 1)
           );


    assert(zz->n_s <= zz->max_s_heap_size);
}

