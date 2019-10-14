#include "move_median.h"

ai_t *mm_move_median(ai_t *a, idx_t length, idx_t window, idx_t min_count);
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


int main(void) {
    return mm_unit_test();
}


/* moving window median of 1d arrays returns output array */
ai_t *mm_move_median(ai_t *a, idx_t length, idx_t window, idx_t min_count) {
    mm_handle *mm;
    ai_t *out;
    idx_t i;

    out = malloc(length * sizeof(ai_t));
    mm = mm_new_nan(window, min_count);
    for (i=0; i < length; i++) {
        if (i < window) {
            out[i] = mm_update_init_nan(mm, a[i]);
        } else {
            out[i] = mm_update_nan(mm, a[i]);
        }
        if (i == window) {
            mm_print_line();
            printf("window complete; switch to mm_update\n");
        }
        mm_print_line();
        printf("inserting ai = %f\n", a[i]);
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
                 char *err_msg) {
    idx_t i;
    int failed = 0;

    mm_print_line();
    printf("%s\n", err_msg);
    mm_print_line();
    printf("input     actual    desired\n");
    for (i=0; i < length; i++)
    {
        if (isnan(actual[i]) && isnan(desired[i])) {
            printf("%9f %9f %9f\n", input[i], actual[i], desired[i]);
        } else if (actual[i] != desired[i]) {
            failed = 1;
            printf("%9f %9f %9f BUG\n", input[i], actual[i], desired[i]);
        } else {
            printf("%9f %9f %9f\n", input[i], actual[i], desired[i]);
        }
    }

    return failed;
}


int mm_unit_test(void) {
    ai_t arr_input[] = {0,   3,   7,  NAN, 1,   5,   8,   9,   2,  NAN};
    ai_t desired[] =   {0 ,  1.5, 3,  5,   4,   3,   5,   8,   8,  5.5};
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

    actual = mm_move_median(arr_input, length, window, min_count);
    failed = mm_assert_equal(actual, desired, arr_input, length, err_msg);

    free(actual);
    free(err_msg);

    return failed;
}


void mm_print_node(mm_node *node) {
    printf("\n\n%d small\n", node->region);
    printf("%d idx\n", node->idx);
    printf("%f ai\n", node->ai);
    printf("%p next\n\n", node->next);
}


void mm_print_chain(mm_handle *mm) {
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


void mm_check(mm_handle *mm) {

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
void mm_dump(mm_handle *mm) {
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
        for (i = 0; i < (int)mm->n_n; ++i) {
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
        for (i = 0; i < (int)mm->n_s; ++i) {
            printf("%i %f\n", (int)mm->s_heap[i]->idx, mm->s_heap[i]->ai);
        }
        printf("\n\nLarge heap:\n");
        for (i = 0; i < (int)mm->n_l; ++i) {
            printf("%i %f\n", (int)mm->l_heap[i]->idx, mm->l_heap[i]->ai);
        }
        printf("\n\nNaN heap:\n");
        for (i = 0; i < (int)mm->n_n; ++i) {
            printf("%i %f\n", (int)mm->n_array[i]->idx, mm->n_array[i]->ai);
        }
    }
}


/* Code to print a binary tree from http://stackoverflow.com/a/13755783
 * Code modified for bottleneck's needs. */
void
mm_print_binary_heap(mm_node **heap, idx_t n_array, idx_t oldest_idx,
                     idx_t newest_idx) {
    const int line_width = 77;
    int print_pos[n_array];
    int i, j, k, pos, x=1, level=0;

    print_pos[0] = 0;
    for (i=0,j=1; i<(int)n_array; i++,j++) {
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


void mm_print_line(void) {
    int i, width = 70;
    for (i=0; i < width; i++)
        printf("-");
    printf("\n");
}
