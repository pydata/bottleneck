/*
   Copyright (c) 2011 J. David Lee. All rights reserved.
   Released under a Simplified BSD license

   Adapted, expanded, and added NaN handling for Bottleneck:
   Copyright 2016 Keith Goodman
   Released under the Bottleneck license
*/

#include "move_median.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define SWAP_NODES(heap, idx1, node1, idx2, node2) \
heap[idx1] = node2;                                \
heap[idx2] = node1;                                \
node1->idx = idx2;                                 \
node2->idx = idx1;                                 \
idx1       = idx2


/*
-----------------------------------------------------------------------------
  Prototypes
-----------------------------------------------------------------------------
*/

/* helper functions */
static inline ai_t mm_get_median(mm_handle *mm);
static inline void heapify_small_node(mm_handle *mm, idx_t idx);
static inline void heapify_large_node(mm_handle *mm, idx_t idx);
static inline idx_t mm_get_smallest_child(mm_node **heap, idx_t window,
                                             idx_t idx, mm_node **child);
static inline idx_t mm_get_largest_child(mm_node **heap, idx_t window,
                                            idx_t idx, mm_node **child);
static inline void mm_move_up_small(mm_node **heap, idx_t idx,
                                       mm_node *node, idx_t p_idx,
                                       mm_node *parent);
static inline void mm_move_down_small(mm_node **heap, idx_t window,
                                         idx_t idx, mm_node *node);
static inline void mm_move_down_large(mm_node **heap, idx_t idx,
                                         mm_node *node, idx_t p_idx,
                                         mm_node *parent);
static inline void mm_move_up_large(mm_node **heap, idx_t window, idx_t idx,
                                       mm_node *node);
static inline void mm_swap_heap_heads(mm_node **s_heap, idx_t n_s,
                                         mm_node **l_heap, idx_t n_l,
                                         mm_node *s_node, mm_node *l_node);


/*
-----------------------------------------------------------------------------
  Top-level non-nan functions
-----------------------------------------------------------------------------
*/

/* At the start of bn.move_median two heaps are created. One heap contains the
 * small values (a max heap); the other heap contains the large values (a min
 * heap). The handle, containing information about the heaps, is returned. */
mm_handle *
mm_new(const idx_t window, idx_t min_count) {
    mm_handle *mm = malloc(sizeof(mm_handle));
    mm->nodes = malloc(window * sizeof(mm_node*));
    mm->node_data = malloc(window * sizeof(mm_node));

    mm->s_heap = mm->nodes;
    mm->l_heap = &mm->nodes[window / 2 + window % 2];

    mm->window = window;
    mm->odd = window % 2;
    mm->min_count = min_count;

    mm_reset(mm);

    return mm;
}


/* Insert a new value, ai, into one of the heaps. Use this function when
 * the heaps contain less than window-1 nodes. Returns the median value.
 * Once there are window-1 nodes in the heap, switch to using mm_update. */
ai_t
mm_update_init(mm_handle *mm, ai_t ai) {
    mm_node *node;
    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;

    node = &mm->node_data[n_s + n_l];
    node->ai = ai;

    if (n_s == 0) {
        /* the first node to appear in a heap */
        mm->s_heap[0] = node;
        node->region = SH;
        node->idx = 0;
        mm->oldest = node; /* only need to set the oldest node once */
        mm->n_s = 1;
        mm->s_first_leaf = 0;
    } else {
        /* at least one node already exists in the heaps */
        mm->newest->next = node;
        if (n_s > n_l) {
            /* add new node to large heap */
            mm->l_heap[n_l] = node;
            node->region = LH;
            node->idx = n_l;
            ++mm->n_l;
            mm->l_first_leaf = FIRST_LEAF(mm->n_l);
            heapify_large_node(mm, n_l);
        } else {
            /* add new node to small heap */
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
 * when the double heap contains at least window-1 nodes. Returns the median
 * value. If there are less than window-1 nodes in the heap, use
 * mm_update_init. */
ai_t
mm_update(mm_handle *mm, ai_t ai) {
    /* node is oldest node with ai of newest node */
    mm_node *node = mm->oldest;
    node->ai = ai;

    /* update oldest, newest */
    mm->oldest = mm->oldest->next;
    mm->newest->next = node;
    mm->newest = node;

    /* adjust position of new node in heap if needed */
    if (node->region == SH) {
        heapify_small_node(mm, node->idx);
    } else {
        heapify_large_node(mm, node->idx);
    }

    /* return the median */
    if (mm->odd) {
        return mm->s_heap[0]->ai;
    } else {
        return (mm->s_heap[0]->ai + mm->l_heap[0]->ai) / 2.0;
    }
}


/*
-----------------------------------------------------------------------------
  Top-level nan functions
-----------------------------------------------------------------------------
*/

/* At the start of bn.move_median two heaps and a nan array are created. One
 * heap contains the small values (a max heap); the other heap contains the
 * large values (a min heap); the nan array contains the NaNs. The handle,
 * containing information about the heaps and the nan array is returned. */
mm_handle *
mm_new_nan(const idx_t window, idx_t min_count) {
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


/* Insert a new value, ai, into one of the heaps or the nan array. Use this
 * function when there are less than window-1 nodes. Returns the median
 * value. Once there are window-1 nodes in the heap, switch to using
 * mm_update_nan. */
ai_t
mm_update_init_nan(mm_handle *mm, ai_t ai) {
    mm_node *node;
    idx_t n_s = mm->n_s;
    idx_t n_l = mm->n_l;
    idx_t n_n = mm->n_n;

    node = &mm->node_data[n_s + n_l + n_n];
    node->ai = ai;

    if (ai != ai) {
        mm->n_array[n_n] = node;
        node->region = NA;
        node->idx = n_n;
        if (n_s + n_l + n_n == 0) {
            /* only need to set the oldest node once */
            mm->oldest = node;
        } else {
            mm->newest->next = node;
        }
        ++mm->n_n;
    } else {
        if (n_s == 0) {
            /* the first node to appear in a heap */
            mm->s_heap[0] = node;
            node->region = SH;
            node->idx = 0;
            if (n_s + n_l + n_n == 0) {
                /* only need to set the oldest node once */
                mm->oldest = node;
            } else {
                mm->newest->next = node;
            }
            mm->n_s = 1;
            mm->s_first_leaf = 0;
        } else {
            /* at least one node already exists in the heaps */
            mm->newest->next = node;
            if (n_s > n_l) {
                /* add new node to large heap */
                mm->l_heap[n_l] = node;
                node->region = LH;
                node->idx = n_l;
                ++mm->n_l;
                mm->l_first_leaf = FIRST_LEAF(mm->n_l);
                heapify_large_node(mm, n_l);
            } else {
                /* add new node to small heap */
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


/* Insert a new value, ai, into one of the heaps or the nan array. Use this
 * function when there are at least window-1 nodes. Returns the median value.
 * If there are less than window-1 nodes, use mm_update_init_nan. */
ai_t
mm_update_nan(mm_handle *mm, ai_t ai) {
    idx_t n_s, n_l, n_n;

    mm_node **l_heap;
    mm_node **s_heap;
    mm_node **n_array;
    mm_node *node2;

    /* node is oldest node with ai of newest node */
    mm_node *node = mm->oldest;
    idx_t idx = node->idx;
    node->ai = ai;

    /* update oldest, newest */
    mm->oldest = mm->oldest->next;
    mm->newest->next = node;
    mm->newest = node;

    l_heap = mm->l_heap;
    s_heap = mm->s_heap;
    n_array = mm->n_array;

    n_s = mm->n_s;
    n_l = mm->n_l;
    n_n = mm->n_n;

    if (ai != ai) {
        if (node->region == SH) {
            /* Oldest node is in the small heap and needs to be moved
             * to the nan array. Resulting hole in the small heap will be
             * filled with the rightmost leaf of the last row of the small
             * heap. */

            /* insert node into nan array */
            node->region = NA;
            node->idx = n_n;
            n_array[n_n] = node;
            ++mm->n_n;

            /* plug small heap hole */
            --mm->n_s;
            if (mm->n_s == 0) {
                mm->s_first_leaf = 0;
                if (n_l > 0) {
                    /* move head node from the large heap to the small heap */
                    node2 = mm->l_heap[0];
                    node2->region = SH;
                    s_heap[0] = node2;
                    mm->n_s = 1;
                    mm->s_first_leaf = 0;

                    /* plug hole in large heap */
                    node2 = mm->l_heap[mm->n_l - 1];
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
                if (idx != n_s - 1) {
                    s_heap[idx] = s_heap[n_s - 1];
                    s_heap[idx]->idx = idx;
                    heapify_small_node(mm, idx);
                }
                if (mm->n_s < mm->n_l) {
                    /* move head node from the large heap to the small heap */
                    node2 = mm->l_heap[0];
                    node2->idx = mm->n_s;
                    node2->region = SH;
                    s_heap[mm->n_s] = node2;
                    ++mm->n_s;
                    mm->l_first_leaf = FIRST_LEAF(mm->n_s);
                    heapify_small_node(mm, node2->idx);

                    /* plug hole in large heap */
                    node2 = mm->l_heap[mm->n_l - 1];
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

            /* insert node into nan array */
            node->region = NA;
            node->idx = n_n;
            n_array[n_n] = node;
            ++mm->n_n;

            /* plug large heap hole */
            if (idx != n_l - 1) {
                l_heap[idx] = l_heap[n_l - 1];
                l_heap[idx]->idx = idx;
                heapify_large_node(mm, idx);
            }
            --mm->n_l;
            if (mm->n_l == 0) {
                mm->l_first_leaf = 0;
            } else {
                mm->l_first_leaf = FIRST_LEAF(mm->n_l);
            }
            if (mm->n_l < mm->n_s - 1) {
                /* move head node from the small heap to the large heap */
                node2 = mm->s_heap[0];
                node2->idx = mm->n_l;
                node2->region = LH;
                l_heap[mm->n_l] = node2;
                ++mm->n_l;
                mm->l_first_leaf = FIRST_LEAF(mm->n_l);
                heapify_large_node(mm, node2->idx);

                /* plug hole in small heap */
                if (n_s != 1) {
                    node2 = mm->s_heap[mm->n_s - 1];
                    node2->idx = 0;
                    s_heap[0] = node2;
                }
                --mm->n_s;
                if (mm->n_s == 0) {
                    mm->s_first_leaf = 0;
                } else {
                    mm->s_first_leaf = FIRST_LEAF(mm->n_s);
                }
                heapify_small_node(mm, 0);
            }
            /* reorder large heap if needed */
            heapify_large_node(mm, idx);
        } else if (node->region == NA) {
            /* insert node into nan heap */
            n_array[idx] = node;
        }
    } else {
        if (node->region == SH) {
            heapify_small_node(mm, idx);
        } else if (node->region == LH) {
            heapify_large_node(mm, idx);
        } else {
            /* ai is not NaN but oldest node is in nan array */
            if (n_s > n_l) {
                /* insert into large heap */
                node->region = LH;
                node->idx = n_l;
                l_heap[n_l] = node;
                ++mm->n_l;
                mm->l_first_leaf = FIRST_LEAF(mm->n_l);
                heapify_large_node(mm, n_l);
            } else {
                /* insert into small heap */
                node->region = SH;
                node->idx = n_s;
                s_heap[n_s] = node;
                ++mm->n_s;
                mm->s_first_leaf = FIRST_LEAF(mm->n_s);
                heapify_small_node(mm, n_s);
            }
            /* plug nan array hole */
            if (idx != n_n - 1) {
                n_array[idx] = n_array[n_n - 1];
                n_array[idx]->idx = idx;
            }
            --mm->n_n;
        }
    }
    return mm_get_median(mm);
}


/*
-----------------------------------------------------------------------------
  Top-level functions common to nan and non-nan cases
-----------------------------------------------------------------------------
*/

/* At the end of each slice the double heap and nan array are reset (mm_reset)
 * to prepare for the next slice. In the 2d input array case (with axis=1),
 * each slice is a row of the input array. */
void
mm_reset(mm_handle *mm) {
    mm->n_l = 0;
    mm->n_s = 0;
    mm->n_n = 0;
    mm->s_first_leaf = 0;
    mm->l_first_leaf = 0;
}


/* After bn.move_median is done, free the memory */
void
mm_free(mm_handle *mm) {
    free(mm->node_data);
    free(mm->nodes);
    free(mm);
}


/*
-----------------------------------------------------------------------------
  Utility functions
-----------------------------------------------------------------------------
*/

/* Return the current median */
static inline ai_t
mm_get_median(mm_handle *mm) {
    idx_t n_total = mm->n_l + mm->n_s;
    if (n_total < mm->min_count)
        return MM_NAN();
    if (min(mm->window, n_total) % 2 == 1)
        return mm->s_heap[0]->ai;
    return (mm->s_heap[0]->ai + mm->l_heap[0]->ai) / 2.0;
}


static inline void
heapify_small_node(mm_handle *mm, idx_t idx) {
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

    /* Internal or leaf node */
    if (idx > 0) {
        idx2 = P_IDX(idx);
        node2 = s_heap[idx2];

        /* Move up */
        if (ai > node2->ai) {
            mm_move_up_small(s_heap, idx, node, idx2, node2);

            /* Maybe swap between heaps */
            node2 = l_heap[0];
            if (ai > node2->ai) {
                mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
            }
        } else if (idx < mm->s_first_leaf) {
            /* Move down */
            mm_move_down_small(s_heap, n_s, idx, node);
        }
    } else {
        /* Head node */
        node2 = l_heap[0];
        if (n_l > 0 && ai > node2->ai) {
            mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
        } else {
            mm_move_down_small(s_heap, n_s, idx, node);
        }
    }
}


static inline void
heapify_large_node(mm_handle *mm, idx_t idx) {
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

    /* Internal or leaf node */
    if (idx > 0) {
        idx2 = P_IDX(idx);
        node2 = l_heap[idx2];

        /* Move down */
        if (ai < node2->ai) {
            mm_move_down_large(l_heap, idx, node, idx2, node2);

            /* Maybe swap between heaps */
            node2 = s_heap[0];
            if (ai < node2->ai) {
                mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
            }
        } else if (idx < mm->l_first_leaf) {
            /* Move up */
            mm_move_up_large(l_heap, n_l, idx, node);
        }
    } else {
        /* Head node */
        node2 = s_heap[0];
        if (n_s > 0 && ai < node2->ai) {
            mm_swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
        } else {
            mm_move_up_large(l_heap, n_l, idx, node);
        }
    }

}


/* Return the index of the smallest child of the node. The pointer
 * child will also be set. */
static inline idx_t
mm_get_smallest_child(mm_node **heap, idx_t window, idx_t idx, mm_node **child) {
    idx_t i0 = FC_IDX(idx);
    idx_t i1 = i0 + NUM_CHILDREN;
    i1 = min(i1, window);

    switch (i1 - i0) {
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


/* Return the index of the largest child of the node. The pointer
 * child will also be set. */
static inline idx_t
mm_get_largest_child(mm_node **heap, idx_t window, idx_t idx, mm_node **child) {
    idx_t i0 = FC_IDX(idx);
    idx_t i1 = i0 + NUM_CHILDREN;
    i1 = min(i1, window);

    switch (i1 - i0) {
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
static inline void
mm_move_up_small(mm_node **heap, idx_t idx, mm_node *node, idx_t p_idx,
                 mm_node *parent) {
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
static inline void
mm_move_down_small(mm_node **heap, idx_t window, idx_t idx, mm_node *node) {
    mm_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = mm_get_largest_child(heap, window, idx, &child);

    while (ai < child->ai) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = mm_get_largest_child(heap, window, idx, &child);
    }
}


/* Move the given node down through the heap to the appropriate position. */
static inline void
mm_move_down_large(mm_node **heap, idx_t idx, mm_node *node, idx_t p_idx,
                   mm_node *parent) {
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
static inline void
mm_move_up_large(mm_node **heap, idx_t window, idx_t idx, mm_node *node) {
    mm_node *child;
    ai_t ai = node->ai;
    idx_t c_idx = mm_get_smallest_child(heap, window, idx, &child);

    while (ai > child->ai) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = mm_get_smallest_child(heap, window, idx, &child);
    }
}


/* Swap the heap heads. */
static inline void
mm_swap_heap_heads(mm_node **s_heap, idx_t n_s, mm_node **l_heap, idx_t n_l,
                   mm_node *s_node, mm_node *l_node) {
    s_node->region = LH;
    l_node->region = SH;
    s_heap[0] = l_node;
    l_heap[0] = s_node;
    mm_move_down_small(s_heap, n_s, 0, l_node);
    mm_move_up_large(l_heap, n_l, 0, s_node);
}
