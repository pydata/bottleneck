from multiprocessing.pool import ThreadPool


def run_threaded(nthreads, array_list, func, **kwargs):
    unary_func = make_unary(func, **kwargs)
    try:
        threadpool = ThreadPool(nthreads)
        output_list = threadpool.map(unary_func, array_list)
    finally:
        threadpool.close()
    return output_list


def make_unary(func, **kwargs):
    def unary_func(arr):
        return func(arr, **kwargs)
    return unary_func
