
def make(func, maxdim=3):
    codes = []
    codes.append(func['main'])
    select = Selector(func['name'])
    for key in func['templates']:
        f = func['templates'][key]
        code = template(name=func['name'],
                        top=f['top'],
                        loop=f['loop'],
                        axisNone=f['axisNone'],
                        dtypes=f['dtypes'],
                        select=select)
        codes.append(code)
    codes.append('\n' + select.asstring())    
    fid = open(func['pyx_file'], 'w')
    fid.write(''.join(codes))
    fid.close()

def template(name, top, loop, axisNone, dtypes, select):

    ndims = loop.keys()
    ndims.sort()
    funcs = []
    for ndim in ndims:
        if axisNone:
            axes = [None]
        else:
            axes = range(ndim)
        for dtype in dtypes:
            for axis in axes:

                # Code template
                func = top
                
                # loop
                func += loop_cdef(ndim, dtype, axis)
                func += loopy(loop[ndim], ndim, axis)

                # name, ndim, dtype, axis
                func = func.replace('NAME', name)
                func = func.replace('NDIM', str(ndim))
                func = func.replace('DTYPE', dtype)
                func = func.replace('AXIS', str(axis))

                funcs.append(func)
                select.append(ndim, dtype, axis)
    
    return ''.join(funcs)

def loopy(loop, ndim, axis):
    
    if ndim < 1:
        raise ValueError("ndim(=%d) must be and integer greater than 0" % ndim)
    if (axis < 0) and (axis is not None):
        raise ValueError("`axis` must be a non-negative integer or None")
    if axis >= ndim:
        raise ValueError("`axis` must be less then `ndim`")
   
    INDEXALL = ', '.join(['i' + str(i) for i in range(ndim)])
    code = loop.replace('INDEXALL', INDEXALL)

    idx = range(ndim)
    if axis is not None:
        idx.pop(axis)
    INDEXPOP = ', '.join(['i' + str(i) for i in idx])
    code = code.replace('INDEXPOP', INDEXPOP)

    idx = range(ndim)
    if axis is not None:
        idxpop = idx.pop(axis)
        idx.append(idxpop)
    for i, j in enumerate(idx):
        code = code.replace('INDEX%d' % i, '%d' % j)

    return code

def loop_cdef(ndim, dtype, axis):
    
    if ndim < 1:
        raise ValueError("ndim(=%d) must be and integer greater than 0" % ndim)
    if (axis < 0) and (axis is not None):
        raise ValueError("`axis` must be a non-negative integer or None")
    if axis >= ndim:
        raise ValueError("`axis` must be less then `ndim`")
   
    tab = '    '
    cdefs = []

    # cdef loop indices
    idx = ', '.join(['i'+str(i) for i in range(ndim)])
    cdefs.append(tab + 'cdef Py_ssize_t ' + idx)
    
    # cdef initialize output
    for dim in range(ndim):
        cdefs.append(tab + "cdef int n%d = a.shape[%d]" % (dim, dim))
    if (ndim > 1) and (axis is not None):
        idx = range(ndim)
        del idx[axis]
        ns = ', '.join(['n'+str(i) for i in idx])
        cdefs.append("%scdef np.npy_intp *dims = [%s]" % (tab, ns))
        y = "%scdef np.ndarray[np.%s_t, ndim=%d] y = PyArray_EMPTY(%d, dims,"
        y += "\n                                              NPY_%s, 0)"
        cdefs.append(y % (tab, dtype, ndim-1, ndim-1, dtype))
    
    return '\n'.join(cdefs) + '\n'

class Selector(object):
    
    def __init__(self, name):
        self.name = name
        self.src = []
        self.src.append("cdef dict %s_dict = {}" % name)

    def append(self, ndim, dtype, axis):
        fmt = "%s_dict[(%s, %s, %s)] = %s_%sd_%s_axis%s"
        if (ndim == 1) and (axis is None):
            tup = (self.name, str(ndim), str(dtype), str(0),
                   self.name, str(ndim), str(dtype), str(axis))
            self.src.append(fmt % tup)
        tup = 2 * (self.name, str(ndim), str(dtype), str(axis))
        self.src.append(fmt % tup)
    
    def asstring(self):
        return '\n'.join(self.src)
