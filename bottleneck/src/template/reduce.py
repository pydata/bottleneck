import re
import inspect

TAB = '    '


def ndreduce(func,
             dtypes=[['float64'], ['float32'], ['int64'], ['int32']],
             var_dtypes={0: ['out', 'ai']},
             int_input_name=None):
    src = get_source(func)
    src_all = make_template_all(src, dtypes, var_dtypes)
    print src_all
    print "------------------------------------"
    return func


def get_source(func):
    src = inspect.getsource(func)
    src = re.sub(r'^@ndreduce.*\n', '', src, flags=re.M)  # rm decorators
    src = re.sub(r'^#.*\n', '', src, flags=re.M)  # rm code comments
    src = re.sub(r'^\s*\n', '', src, flags=re.M)  # rm blank lines
    return src


def get_func_name(src):
    names = re.findall(r'^def\s*(.+?)\s*\(', src, flags=re.MULTILINE)
    if len(names) > 1:
        raise ValueError('More than one function name found')
    return names[0]


def make_top_all(var_dtypes):
    top = 'cdef DTYPE0_t NAME_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,'
    top += '\n%sPy_ssize_t length, int int_input):' % (8 * TAB)
    top = top.replace('TAB', TAB)
    top = [top]
    for key, values in var_dtypes.iteritems():
        if type(key) is int:
            for value in values:
                top.append('%scdef DTYPE%d %s' % (TAB, key, value))
    return '\n'.join(top) + '\n'


def make_template_all(src, dtypes, var_dtypes):
    func_name = get_func_name(src)
    top = make_top_all(var_dtypes)
    src = re.sub(r'^def\s*.*\n', top, src, flags=re.MULTILINE)
    src = src.replace('NAME', func_name)
    ai = 'ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]'
    src = src.replace('ai = a[i]', ai)
    bot = "%sPyArray_ITER_NEXT(ita)\n%sreturn out" % (2 * TAB, TAB)
    src += bot
    return src


def make_top_one(var_dtypes): #TODO  start work here
    top = 'cdef DTYPE0_t NAME_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,'
    top += '\n%sPy_ssize_t length, int int_input):' % (8 * TAB)
    top = top.replace('TAB', TAB)
    top = [top]
    for key, values in var_dtypes.iteritems():
        if type(key) is int:
            for value in values:
                top.append('%scdef DTYPE%d %s' % (TAB, key, value))
    return '\n'.join(top) + '\n'

"""
cdef void nansum_one_float64(np.flatiter ita, np.flatiter ity,
                             Py_ssize_t stride, Py_ssize_t length,
                             int int_input):
    cdef Py_ssize_t i
    cdef float64_t asum = 0, ai
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<float64_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ity)
    else:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            for i in range(length):
                ai = (<float64_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    asum += ai
            (<float64_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
"""


@ndreduce
@ndreduce
# help
#more help
def nansum(a):

    out = 0

    for i in range(length):
        ai = a[i]
        if ai == ai:
            out += ai

@ndreduce
#help
# help
def nansum(a):
    out = 0
    for i in range(length):
        ai = a[i]
        if ai == ai:
            out += ai

@ndreduce
def nansum(a):
    out = 0
    for i in range(length):
        ai = a[i]
        if ai == ai:
            out += ai

@ndreduce
def  nansum (a) :
    out = 0
    for i in range(length):
        ai = a[i]
        if ai == ai:
            out += ai

if __name__ == "__main__":
    pass
