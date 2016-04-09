from multiprocessing.pool import ThreadPool
from bottleneck.benchmark.autotimeit import autotimeit


class Thread(object):

    def __init__(self, nthreads, func, **kwargs):
        self.pool = ThreadPool(nthreads)
        self.func = make_unary(func, **kwargs)
        self.nthreads = nthreads
        self.func_name = func.__name__
        self.kwargs = kwargs

    def run(self, arr_list):
        return self.pool.map(self.func, arr_list)

    def timeit(self, arr_list):
        t0 = autotimeit(lambda: map(self.func, arr_list))
        t1 = autotimeit(lambda: self.pool.map(self.func, arr_list))
        return t0, t1

    def __repr__(self):
        txt = "%s(arr" % self.func_name
        for key in self.kwargs:
            txt += ", %s=%s" % (key, self.kwargs[key])
        txt += "); ThreadPool(%d)" % self.nthreads
        return txt


def make_unary(func, **kwargs):
    def unary_func(arr):
        return func(arr, **kwargs)
    return unary_func
