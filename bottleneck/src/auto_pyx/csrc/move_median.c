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

// Swap nodes
#define SWAP_NODES(heap, idx1, node1, idx2, node2) \
heap[idx1] = node2;                                \
heap[idx2] = node1;                                \
node1->idx = idx2;                                 \
node2->idx = idx1;                                 \
idx1       = idx2


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
            mm->l_first_leaf = ceil((mm->n_l - 1) / (double)NUM_CHILDREN);
        }
        else
        {
            // Add to the small heap.

            mm->s_heap[n_s] = node;
            node->small = 1;
            node->idx = n_s;

            ++mm->n_s;
            mm->s_first_leaf = ceil((mm->n_s - 1) / (double)NUM_CHILDREN);
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
 * Move the given node up through the heap to the appropriate position.
 */
inline void
mm_move_up_small(mm_node **heap, idx_t window, idx_t idx, mm_node *node,
                 idx_t p_idx, mm_node *parent)
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


/*
 * Move the given node down through the heap to the appropriate
 * position.
 */
inline void
mm_move_down_large(mm_node **heap, idx_t window, idx_t idx, mm_node *node,
                   idx_t p_idx, mm_node *parent)
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
