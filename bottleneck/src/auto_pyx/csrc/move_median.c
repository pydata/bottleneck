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

typedef size_t _size_t;
typedef double value_t;


/*
 * The number of children has a maximum of 8 due to the manual loop-
 * unrolling used in the code below.
 */
const int NUM_CHILDREN = 8;


// Minimum of two numbers.
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Find indices of parent, first, and last child.
#define P_IDX(i) ((i) - 1) / NUM_CHILDREN
#define FC_IDX(i) NUM_CHILDREN * (i) + 1


struct _mm_node {
    int              small; // 1 if the node is in the small heap.
    _size_t          idx;   // The node's index in the heap array.
    value_t          val;   // The node's value.
    struct _mm_node *next;  // The next node in order of insertion.

    // double linked list for nan tracking
    struct _mm_node *next_nan;  // The next nan node in order of insertion.
    struct _mm_node *prev_nan;  // The prev nan node in order of insertion.
};

typedef struct _mm_node mm_node;

struct _mm_handle {
    _size_t   w_size;    // window size
    _size_t   n_s_nan;   // number of nans in min heap
    _size_t   n_l_nan;   // number of nans in max heap
    int       init_wnd_complete; //if atleast w_size elements have been inserted
    int       odd;       // 1 if the window size is odd, 0 otherwise.
    _size_t   n_s;       // The number of elements in the min heap.
    _size_t   n_l;       // The number of elements in the max heap.
    _size_t   min_count; // If the number of non-NaN values in a window is less
                         // than min_count, then a value of NaN is assigned to the window
    mm_node **s_heap;    // The min heap.
    mm_node **l_heap;    // The max heap.
    mm_node **nodes;     // All the nodes. s_heap and l_heap point into
    // this array.
    mm_node  *node_data; // Pointer to memory location where nodes live.
    mm_node  *first;     // The node added first to the list of nodes.
    mm_node  *last;      // The last (most recent) node added.

    // Most nodes are leaf nodes, therefore it makes sense to have a
    // quick way to check if a node is a leaf to avoid processing.
    _size_t s_first_leaf; // First leaf index in the small heap.
    _size_t l_first_leaf; // First leaf index in the large heap.

    // + and - infinity array
    mm_node  *first_nan_s;     // The node added first to the list of nodes.
    mm_node  *last_nan_s;      // The last (most recent) node added.
    mm_node  *first_nan_l;     // The node added first to the list of nodes.
    mm_node  *last_nan_l;      // The last (most recent) node added.

    _size_t max_s_heap_size;
};

typedef struct _mm_handle mm_handle;

void mm_dump(mm_handle* mm);
void mm_update_withnan(mm_handle *mm, value_t val);
void mm_update_nonan(mm_handle* mm, value_t val);

void check_asserts(mm_handle* mm)
{
    mm_dump(mm);
    assert(mm->n_s >= mm->n_s_nan);
    assert(mm->n_l >= mm->n_l_nan);
    _size_t valid_s = mm->n_s - mm->n_s_nan;
    _size_t valid_l = mm->n_l - mm->n_l_nan;

    // use valid_s and valid_l or get compiler warnings
    // these lines do nothing
    _size_t dummy = valid_l + valid_s;
    if (dummy > 100){dummy = 99;}

    assert(valid_s < 5000000000); //most likely an overflow
    assert(valid_l < 5000000000); //most likely an overflow

    assert(mm->n_s_nan < 5000000000); //most likely an overflow
    assert(mm->n_l_nan < 5000000000); //most likely an overflow

    if (mm->first_nan_l)
    {
        assert(mm->last_nan_l);
        assert(mm->last_nan_l->next_nan == NULL);
        assert(mm->first_nan_l->prev_nan == NULL);
    }
    else
        assert(mm->last_nan_l == NULL);

    if (mm->first_nan_s)
    {
        assert(mm->last_nan_s);
        assert(mm->last_nan_s->next_nan == NULL);
        assert(mm->first_nan_s->prev_nan == NULL);
    }
    else
        assert(mm->last_nan_s == NULL);

    size_t len = 0;
    mm_node* iter = mm->first_nan_l;
    while (iter!=NULL)
    {
        assert(isinf(iter->val));
        assert(len <= mm->n_l);
        if (iter->next_nan != NULL)
        {
            assert(iter->prev_nan != iter->next_nan);
            assert(iter->next_nan->prev_nan == iter);
        }
        iter = iter->next_nan;
        ++len;
    }

    len = 0;
    iter = mm->first_nan_s;
    while (iter!=NULL)
    {
        assert(isinf(iter->val));
        assert(len <= mm->n_s);
        if (iter->next_nan != NULL)
        {
            assert(iter->prev_nan != iter->next_nan);
            assert(iter->next_nan->prev_nan == iter);
        }
        iter = iter->next_nan;
        ++len;
    }

    // since valid_l and valid_s are signed, these will overflow and we don't have to check for diffs of -5, etc.
    assert(
           ((valid_l - valid_s) <= 1)
           || ((valid_s - valid_l) <= 1)
           );


    assert(mm->n_s <= mm->max_s_heap_size);
}


/*
 * Return the index of the smallest child of the node. The pointer
 * child will also be set.
 */
_size_t get_smallest_child(mm_node **heap,
                           _size_t   size,
                           _size_t   idx,
                           mm_node  *node,
                           mm_node  **child) {
    _size_t i0 = FC_IDX(idx);
    _size_t i1 = i0 + NUM_CHILDREN;
    i1 = min(i1, size);

    switch(i1 - i0) {
        case  8: if(heap[i0 + 7]->val < heap[idx]->val) { idx = i0 + 7; }
        case  7: if(heap[i0 + 6]->val < heap[idx]->val) { idx = i0 + 6; }
        case  6: if(heap[i0 + 5]->val < heap[idx]->val) { idx = i0 + 5; }
        case  5: if(heap[i0 + 4]->val < heap[idx]->val) { idx = i0 + 4; }
        case  4: if(heap[i0 + 3]->val < heap[idx]->val) { idx = i0 + 3; }
        case  3: if(heap[i0 + 2]->val < heap[idx]->val) { idx = i0 + 2; }
        case  2: if(heap[i0 + 1]->val < heap[idx]->val) { idx = i0 + 1; }
        case  1: if(heap[i0    ]->val < heap[idx]->val) { idx = i0;     }
    }

    *child = heap[idx];
    return idx;
}


/*
 * Return the index of the largest child of the node. The pointer
 * child will also be set.
 */
_size_t get_largest_child(mm_node **heap,
                          _size_t   size,
                          _size_t   idx,
                          mm_node  *node,
                          mm_node  **child) {
    _size_t i0 = FC_IDX(idx);
    _size_t i1 = i0 + NUM_CHILDREN;
    i1 = min(i1, size);

    switch(i1 - i0) {
        case  8: if(heap[i0 + 7]->val > heap[idx]->val) { idx = i0 + 7; }
        case  7: if(heap[i0 + 6]->val > heap[idx]->val) { idx = i0 + 6; }
        case  6: if(heap[i0 + 5]->val > heap[idx]->val) { idx = i0 + 5; }
        case  5: if(heap[i0 + 4]->val > heap[idx]->val) { idx = i0 + 4; }
        case  4: if(heap[i0 + 3]->val > heap[idx]->val) { idx = i0 + 3; }
        case  3: if(heap[i0 + 2]->val > heap[idx]->val) { idx = i0 + 2; }
        case  2: if(heap[i0 + 1]->val > heap[idx]->val) { idx = i0 + 1; }
        case  1: if(heap[i0    ]->val > heap[idx]->val) { idx = i0;     }
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
void move_up_small(mm_node **heap,
                   _size_t   size,
                   _size_t   idx,
                   mm_node  *node,
                   _size_t   p_idx,
                   mm_node  *parent) {
    do {
        SWAP_NODES(heap, idx, node, p_idx, parent);
        if(idx == 0) {
            break;
        }
        p_idx = P_IDX(idx);
        parent = heap[p_idx];
    } while (node->val > parent->val);
}


/*
 * Move the given node down through the heap to the appropriate position.
 */
void move_down_small(mm_node **heap,
                     _size_t   size,
                     _size_t   idx,
                     mm_node  *node) {
    mm_node *child;
    value_t val   = node->val;
    _size_t c_idx = get_largest_child(heap, size, idx, node, &child);

    while(val < child->val) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = get_largest_child(heap, size, idx, node, &child);
    }
}


/*
 * Move the given node down through the heap to the appropriate
 * position.
 */
void move_down_large(mm_node **heap,
                     _size_t   size,
                     _size_t   idx,
                     mm_node  *node,
                     _size_t   p_idx,
                     mm_node  *parent) {
    do {
        SWAP_NODES(heap, idx, node, p_idx, parent);
        if(idx == 0) {
            break;
        }
        p_idx = P_IDX(idx);
        parent = heap[p_idx];
    } while (node->val < parent->val);
}



/*
 * Move the given node up through the heap to the appropriate position.
 */
void move_up_large(mm_node **heap,
                   _size_t   size,
                   _size_t   idx,
                   mm_node  *node) {
    mm_node *child;
    value_t val   = node->val;
    _size_t c_idx = get_smallest_child(heap, size, idx, node, &child);

    while(val > child->val) {
        SWAP_NODES(heap, idx, node, c_idx, child);
        c_idx = get_smallest_child(heap, size, idx, node, &child);
    }
}


/*
 * Swap the heap heads.
 */
void swap_heap_heads(mm_node **s_heap,
                     _size_t   n_s,
                     mm_node **l_heap,
                     _size_t   n_l,
                     mm_node  *s_node,
                     mm_node  *l_node) {
    s_node->small = 0;
    l_node->small = 1;
    s_heap[0] = l_node;
    l_heap[0] = s_node;
    move_down_small(s_heap, n_s, 0, l_node);
    move_up_large(l_heap, n_l, 0, s_node);
}


void mm_update_helper( mm_handle *mm, mm_node *node, value_t val) {
    // Replace value of node
    node->val = val;

    // Local variables.
    _size_t  idx  = node->idx;

    mm_node **l_heap = mm->l_heap;
    mm_node **s_heap = mm->s_heap;
    _size_t n_s      = mm->n_s;
    _size_t n_l      = mm->n_l;

    mm_node *node2;
    _size_t  idx2;


    // In small heap.
    if(node->small) {

        // Internal or leaf node.
        if(idx > 0) {
            idx2 = P_IDX(idx);
            node2 = s_heap[idx2];

            // Move up.
            if(val > node2->val) {
                move_up_small(s_heap, n_s, idx, node, idx2, node2);

                // Maybe swap between heaps.
                node2 = (n_l>0) ? l_heap[0] : NULL; // needed because we could've only inserted nan and then a #
                if((node2 != NULL) && (val > node2->val)) {
                    swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
                }
            }

            // Move down.
            else if(idx < mm->s_first_leaf) {
                move_down_small(s_heap, n_s, idx, node);
            }
        }

        // Head node.
        else {
            node2 = (n_l>0) ? l_heap[0] : NULL; // needed because we could've only inserted nan and then a #
            if((node2 != NULL) && (val > node2->val)) {
                swap_heap_heads(s_heap, n_s, l_heap, n_l, node, node2);
            } else {
                move_down_small(s_heap, n_s, idx, node);
            }
        }
    }

    // In large heap.
    else {

        // Internal or leaf node.
        if(idx > 0) {
            idx2 = P_IDX(idx);
            node2 = l_heap[idx2];

            // Move down.
            if(val < node2->val) {
                move_down_large(l_heap, n_l, idx, node, idx2, node2);

                // Maybe swap between heaps.
                node2 = (n_s>0) ? s_heap[0] : NULL;
                if((node2 != NULL) && (val < node2->val)) {
                    swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
                }
            }

            // Move up.
            else if(idx < mm->l_first_leaf) {
                move_up_large(l_heap, n_l, idx, node);
            }
        }

        // Head node.
        else {
            node2 = (n_s>0) ? s_heap[0] : NULL;
            if((node2 != NULL) && (val < node2->val)) {
                swap_heap_heads(s_heap, n_s, l_heap, n_l, node2, node);
            } else {
                move_up_large(l_heap, n_l, idx, node);
            }
        }
    }
}

void mm_update_withnan_skipevict(mm_handle *mm, value_t val) {
    if (isinf(val))
    {
        mm_node *node = mm->first;
        // we are adding a new nan
        node->next_nan = NULL;

        if (val>0) {
            ++mm->n_l_nan;
            if (mm->first_nan_l == NULL) {
                mm->first_nan_l = node;
                mm->last_nan_l = node;
                node->prev_nan = NULL;
            } else {
                assert(node->next_nan == NULL);
                assert(node!=mm->last_nan_l);
                mm->last_nan_l->next_nan = node;
                node->prev_nan = mm->last_nan_l;
                assert(node->next_nan == NULL);
                mm->last_nan_l = node;
            }
        } else {
            ++mm->n_s_nan;
            if (mm->first_nan_s == NULL) {
                mm->first_nan_s = node;
                mm->last_nan_s = node;
                node->prev_nan = NULL;
            } else {
                mm->last_nan_s->next_nan = node;
                node->prev_nan = mm->last_nan_s;
                mm->last_nan_s = node;
            }
        }
    }

    mm_update_nonan(mm, val);
}


/*
 * insert a nan, during initialization phase.
 */
void mm_insert_nan(mm_handle *mm)
{
    value_t val = 0;

    assert(mm->init_wnd_complete == 0);

    // Local variables.
    _size_t n_s      = mm->n_s;
    _size_t n_l      = mm->n_l;
    _size_t n_s_nan  = mm->n_s_nan;
    _size_t n_l_nan  = mm->n_l_nan;

    // Nodes and indices.
    mm_node *node = &mm->node_data[n_s + n_l];
    node->next = mm->first;
    mm->first = node;

    //check_asserts(mm);

    int l_heap_full = (n_l == (mm->w_size - mm->max_s_heap_size));
    int s_heap_full = (n_s == mm->max_s_heap_size);
    if ( (s_heap_full | (n_s_nan > n_l_nan)) & (l_heap_full == 0) ) {
        // Add to the large heap.

        mm->l_heap[n_l] = node;
        node->small = 0;
        node->idx   = n_l;

        ++mm->n_l;
        mm->l_first_leaf = ceil((mm->n_l - 1) / (double)NUM_CHILDREN);

        val = INFINITY;
    } else {
        // Add to the small heap.

        mm->s_heap[n_s] = node;
        node->small = 1;
        node->idx   = n_s;

        ++mm->n_s;
        mm->s_first_leaf = ceil((mm->n_s - 1) / (double)NUM_CHILDREN);

        val = -INFINITY;
    }

    mm_update_withnan_skipevict(mm, val);
}

void move_nan_helper(mm_handle* mm, mm_node* new_last)
{
    assert(new_last != NULL);

    value_t old_val = new_last->val;
    value_t new_val = -old_val;

    assert(isinf(old_val));

    new_last->val = new_val;
    mm_update_helper(mm, new_last, new_val);
}

void move_nan_from_s_to_l(mm_handle *mm)
{
    // move nan from s to l
    assert(mm->first_nan_s != NULL);

    mm_node* new_last = mm->first_nan_s;
    assert(isinf(new_last->val));
    mm->first_nan_s = mm->first_nan_s->next_nan;

    if (mm->first_nan_s != NULL)
        mm->first_nan_s->prev_nan = NULL;
    else
        mm->last_nan_s = NULL; // that was our last nan on this side

    new_last->next_nan = NULL;

    if (mm->first_nan_l == NULL) {
        mm->first_nan_l = new_last;
        new_last->prev_nan = NULL;
    } else {
        mm->last_nan_l->next_nan = new_last;
        new_last->prev_nan = mm->last_nan_l;
    }

    mm->last_nan_l = new_last;

    --mm->n_s_nan;
    ++mm->n_l_nan;

    move_nan_helper(mm, new_last);
}

void move_nan_from_l_to_s(mm_handle *mm)
{
    // move nan from l to s
    assert(mm->first_nan_l != NULL);

    mm_node* new_last = mm->first_nan_l;
    assert(isinf(new_last->val));
    mm->first_nan_l = mm->first_nan_l->next_nan;

    if (mm->first_nan_l != NULL)
        mm->first_nan_l->prev_nan = NULL;
    else
        mm->last_nan_l = NULL; // that was our last nan on this side

    new_last->next_nan = NULL;

    if (mm->first_nan_s == NULL) {
        mm->first_nan_s = new_last;
        new_last->prev_nan = NULL;
    } else {
        mm->last_nan_s->next_nan = new_last;
        new_last->prev_nan = mm->last_nan_s;
    }

    mm->last_nan_s = new_last;

    --mm->n_l_nan;
    ++mm->n_s_nan;

    move_nan_helper(mm, new_last);
}

/*
 * Insert initial values into the double heap structure.
 *
 * Arguments:
 * mm  -- The double heap structure.
 * idx -- The index of the value running from 0 to size - 1.
 * val -- The value to insert.
 */
void mm_insert_init(mm_handle *mm, value_t val) {
    // Some local variables.
    mm_node *node = NULL;
    _size_t n_s = mm->n_s;
    _size_t n_l = mm->n_l;
    _size_t n_s_nan  = mm->n_s_nan;
    _size_t n_l_nan  = mm->n_l_nan;
    // double check these in debug, to catch overflows
    //check_asserts(mm);

    int is_nan_val = isnan(val);

    node = &mm->node_data[n_s + n_l];
    node->next_nan = NULL;

    // The first node.
    if(n_s == 0) {
        mm->n_s_nan = is_nan_val;

        mm->s_heap[0] = node;
        node->small = 1;
        node->idx   = 0;
        node->next  = mm->l_heap[0];

        mm->n_s = 1;
        mm->first = mm->last = node;
        mm->s_first_leaf = 0;

        if (is_nan_val)
        {
            node->val = -INFINITY;
            node->next_nan = NULL;
            node->prev_nan = NULL;
            mm->first_nan_s = node;
            mm->last_nan_s = node;
        }
        else
            node->val = val;
    }
    else
    {
        // Nodes after the first.

        if (is_nan_val)
        {
            mm_insert_nan(mm);
            //check_asserts(mm);
        }
        else
        {
            node->next  = mm->first;
            mm->first = node;

            _size_t nonnan_n_s = n_s - n_s_nan;
            _size_t nonnan_n_l = n_l - n_l_nan;

            if ((n_s == mm->max_s_heap_size) | (nonnan_n_s > nonnan_n_l))
            {
                // Add to the large heap.

                mm->l_heap[n_l] = node;
                node->small = 0;
                node->idx   = n_l;

                ++mm->n_l;
                mm->l_first_leaf = ceil((mm->n_l - 1) / (double)NUM_CHILDREN);
            }
            else
            {
                // Add to the small heap.

                mm->s_heap[n_s] = node;
                node->small = 1;
                node->idx   = n_s;

                ++mm->n_s;
                mm->s_first_leaf = ceil((mm->n_s - 1) / (double)NUM_CHILDREN);
            }

            mm_update_nonan(mm, val);
        }
    }

    mm->init_wnd_complete = mm->init_wnd_complete | ((n_l + n_s + 1) >= (mm->w_size));
}

void mm_update_withnan(mm_handle *mm, value_t val) {
    // Nodes and indices.
    mm_node *node = mm->first;

    if (isinf(node->val)) {
        // if we are removing a nan
        if (node->small) {
            --mm->n_s_nan;

            if (node == mm->first_nan_s) {
                mm_node* next_ptr = mm->first_nan_s->next_nan;
                mm->first_nan_s = next_ptr;
                if (next_ptr == NULL)
                    mm->last_nan_s = NULL;
                else
                    next_ptr->prev_nan = NULL; // the current nan is the first one
            } else {
                assert(node->prev_nan != NULL);
                mm_node* last_node = node->prev_nan;
                last_node->next_nan = node->next_nan;
                if (node->next_nan == NULL)
                    mm->last_nan_s = last_node;
                else
                    node->next_nan->prev_nan = last_node;
                node->next_nan = NULL;
            }
        } else {
            --mm->n_l_nan;

            if (node == mm->first_nan_l) {
                mm_node* next_ptr = mm->first_nan_l->next_nan;
                mm->first_nan_l = next_ptr;
                if (next_ptr == NULL)
                    mm->last_nan_l = NULL;
                else
                    next_ptr->prev_nan = NULL; // the current nan is the first one
            } else {
                assert(node->prev_nan != NULL);
                mm_node* last_node = node->prev_nan;
                last_node->next_nan = node->next_nan;
                if (node->next_nan == NULL)
                    mm->last_nan_l = last_node;
                else
                    node->next_nan->prev_nan = last_node;
                node->next_nan = NULL;
            }
        }
    }

    mm_update_withnan_skipevict(mm, val);
}


void mm_update_nonan(mm_handle* mm, value_t val)
{
    // Nodes and indices.
    mm_node *node = mm->first;

    // and update first, last
    mm->first = mm->first->next;
    mm->last->next = node;
    mm->last = node;

    mm_update_helper(mm, node, val);
}

void mm_update_checknan(mm_handle *mm, value_t val)
{
    _size_t n_s      = mm->n_s;
    _size_t n_l      = mm->n_l;
    _size_t n_s_nan  = mm->n_s_nan;
    _size_t n_l_nan  = mm->n_l_nan;
    _size_t nonnan_n_s = n_s - n_s_nan;
    _size_t nonnan_n_l = n_l - n_l_nan;

    if (isnan(val))
    {
        // double check these in debug, to catch overflows
        //check_asserts(mm);

        // try to keep the heaps balanced, so we can try to avoid the nan rebalancing penalty
        // makes significant difference when % of nans is large and window size is also large
        mm_node* node_to_evict = mm->first;
        value_t to_evict = node_to_evict->val;
        _size_t evict_effect_s = 0;
        _size_t evict_effect_l = 0;
        if (isinf(to_evict)) {
            if (node_to_evict->small)
                evict_effect_s = 1;
            else
                evict_effect_l = 1;
        }

        if((nonnan_n_s + evict_effect_s) > (nonnan_n_l + evict_effect_l))
            mm_update_withnan(mm, -INFINITY); // add to min heap
        else
            mm_update_withnan(mm, INFINITY); // add to max heap
    } else {
        // Note: we could still be evicting nans here, so call the nan safe function

        // I tried an if-then here to call a non nan function if
        // we are not evicting nan, but penalty to check was
        // too high.  please be careful and measure before trying
        // to optimize here
        mm_update_withnan(mm, val);
    }

    // these could've been updated, so we regrab them
    n_s_nan = mm->n_s_nan;
    n_l_nan = mm->n_l_nan;

    nonnan_n_s = n_s - n_s_nan;
    nonnan_n_l = n_l - n_l_nan;

    if ( nonnan_n_l == nonnan_n_s + 2)
        move_nan_from_s_to_l(mm); // max heap is too big...
    else if ( nonnan_n_s == nonnan_n_l + 2)
        move_nan_from_l_to_s(mm); // min heap is too big...

    // double check these in debug, to catch overflows
    //check_asserts(mm);
}


/*
 *   ----------------
 *   Public functions
 *   ----------------
 */

void mm_reset(mm_handle* mm) {
    mm->n_l = 0;
    mm->n_s = 0;
    mm->n_l_nan = 0;
    mm->n_s_nan = 0;
    mm->init_wnd_complete = 0;

    mm->first_nan_s = NULL;
    mm->last_nan_s = NULL;
    mm->first_nan_l = NULL;
    mm->last_nan_l = NULL;
    mm->first = NULL;
    mm->last = NULL;
}


/*
 * Construct the double heap with the given total number of values.
 *
 * Arguments:
 * size -- The total number of values in the double heap.
 *
 * Return: The mm_handle structure, uninitialized.
 */
mm_handle *mm_new(const _size_t size, _size_t min_count) {
    // only malloc once, this guarantees cache friendly execution
    // and easier code for cleanup
    // this change was profiled to make a 5%-10% difference in performance
    char* memory_block = malloc(sizeof(mm_handle) + size * (sizeof(mm_node*) + sizeof(mm_node)));

    if (memory_block == NULL)
        return NULL;

    char* curr_mem_ptr = memory_block;

    mm_handle *mm = (mm_handle*)curr_mem_ptr;

    curr_mem_ptr += sizeof(mm_handle);
    mm->nodes = (mm_node**) curr_mem_ptr;

    curr_mem_ptr += sizeof(mm_node*) * size;
    mm->node_data = (mm_node*) curr_mem_ptr;

    mm->max_s_heap_size = size/2 + size % 2;
    mm->w_size = size;
    mm->s_heap = mm->nodes;
    mm->l_heap = &mm->nodes[mm->max_s_heap_size];
    mm->min_count = min_count;

    mm_reset(mm);

    return mm;
}


/*
 * Update the running median with a new value.
 */
void mm_update_movemedian_possiblenan(mm_handle* mm, value_t val)
{
    int init_wind_complete = mm->init_wnd_complete;
    if (init_wind_complete == 0)
        mm_insert_init(mm, val);
    else
        mm_update_checknan(mm, val);

    //check_asserts(mm);
}

/*
 * Update the running median with a new value.
 * Only to be called if there are no nans in the *entire input array*
 */
void mm_update_movemedian_nonan(mm_handle* mm, value_t val)
{
    int init_wind_complete = mm->init_wnd_complete;
    if (init_wind_complete == 0)
        mm_insert_init(mm, val);
    else
        mm_update_nonan(mm, val);

    //check_asserts(mm);
}

/*
 * Return the current median value.
 */
value_t mm_get_median(mm_handle *mm) {
    _size_t n_s      = mm->n_s;
    _size_t n_l      = mm->n_l;
    _size_t n_s_nan  = mm->n_s_nan;
    _size_t n_l_nan  = mm->n_l_nan;

    //check_asserts(mm);

    _size_t nonnan_n_l = n_l - n_l_nan;
    _size_t nonnan_n_s = n_s - n_s_nan;
    _size_t numel_total = nonnan_n_l + nonnan_n_s;

    if (numel_total < mm->min_count)
        return NAN;

    _size_t effective_window_size = min(mm->w_size, numel_total);

    if (effective_window_size % 2 == 1) {
        if (nonnan_n_l > nonnan_n_s)
            return mm->l_heap[0]->val;
        else
            return mm->s_heap[0]->val;
    }
    else
        return (mm->s_heap[0]->val + mm->l_heap[0]->val) / 2;
}


void process_median_nans(mm_handle *mm,
                         const value_t * const input_buffer,
                         value_t* output_buffer,
                         _size_t input_size,
                         _size_t skip_num_outputs) // useful for parallel computation
{
    _size_t skip_num_outputs_actual = min(skip_num_outputs, input_size);
    const value_t * current_input = input_buffer;
    value_t* current_output = output_buffer;
    const value_t * const warmup_end = input_buffer + skip_num_outputs_actual;
    const value_t * const input_end = input_buffer + input_size;

    for ( ; current_input != warmup_end; ++current_input)
        mm_update_movemedian_possiblenan(mm, *current_input);

    for ( ; current_input != input_end; ++current_input, ++current_output) {
        mm_update_movemedian_possiblenan(mm, *current_input);
        *current_output = mm_get_median(mm);
    }
}

void process_median_nonans(mm_handle *mm,
                         const value_t * const input_buffer,
                         value_t* output_buffer,
                         _size_t input_size,
                         _size_t skip_num_outputs) // useful for parallel computation
{
    _size_t skip_num_outputs_actual = min(skip_num_outputs, input_size);
    const value_t * current_input = input_buffer;
    value_t* current_output = output_buffer;
    const value_t * const warmup_end = input_buffer + skip_num_outputs_actual;
    const value_t * const input_end = input_buffer + input_size;

    for ( ; current_input != warmup_end; ++current_input)
        mm_update_movemedian_possiblenan(mm, *current_input);

    for ( ; current_input != input_end; ++current_input, ++current_output) {
        mm_update_movemedian_nonan(mm, *current_input);
        *current_output = mm_get_median(mm);
    }
}

/*
 * Print the two heaps to the screen.
 */
void mm_dump(mm_handle *mm) {
    if (!mm) {
        printf("mm is empty");
        return;
    }
    _size_t i;
    if (mm->first)
        printf("\n\nFirst: %f\n", (double)mm->first->val);

    if (mm->last)
        printf("Last: %f\n", (double)mm->last->val);


    printf("\n\nSmall heap:\n");
    for(i = 0; i < mm->n_s; ++i) {
        printf("%i %f\n", (int)mm->s_heap[i]->idx, mm->s_heap[i]->val);
    }

    printf("\n\nLarge heap:\n");
    for(i = 0; i < mm->n_l; ++i) {
        printf("%i %f\n", (int)mm->l_heap[i]->idx, mm->l_heap[i]->val);
    }
}

/*
 * Free memory allocated in mm_new.
 */
void mm_free(mm_handle *mm) {
    free(mm);
}

