
def make(func, maxdim=3):
    codes = []
    codes.append(func['main'])
    select = Selector(func['name'])
    for key in func['templates']:
        f = func['templates'][key]
        code = template(name=func['name'],
                        top=f['top'],
                        outer_loop_init=f['outer_loop_init'],
                        inner_loop_init=f['inner_loop_init'],
                        inner=f['inner'],
                        result=f['result'],
                        inarr=f['inarr'],
                        outarr=f['outarr'],
                        returns=f['returns'],
                        dtypes=f['dtype'],
                        ndims=f['ndims'],
                        axisNone=f['axisNone'],
                        select=select)
        codes.append(code)
    codes.append('\n' + select.asstring())    
    fid = open(func['pyx_file'], 'w')
    fid.write(''.join(codes))
    fid.close()

def template(name, top, outer_loop_init, inner_loop_init, inner, result, inarr,
             outarr, returns, dtypes, ndims, axisNone, select):

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
                func += loopy(ndim, axis, dtype, outer_loop_init,
                              inner_loop_init, inner, result, inarr, outarr,
                              axisNone)

                # name, ndim, dtype, axis
                func = func.replace('NAME', name)
                func = func.replace('NDIM', str(ndim))
                func = func.replace('DTYPE', dtype)
                func = func.replace('AXIS', str(axis))

                # Returns
                if returns is not None:
                    func += '\n    '
                    func += returns
                    func += '\n'

                funcs.append(func)
                select.append(ndim, dtype, axis)
    
    return ''.join(funcs)

def loopy(ndim, axis, dtype, outer_loop_init, inner_loop_init, inner, result,
          inarr, outarr, axisNone):
    
    # Check input
    if ndim < 1:
        raise ValueError("ndim(=%d) must be and integer greater than 0" % ndim)
    if (axis < 0) and (axis is not None):
        raise ValueError("`axis` must be a non-negative integer or None")
    if axis >= ndim:
        raise ValueError("`axis` must be less then `ndim`")
    
    if outer_loop_init is not None:
        outer_loop_init = outer_loop_init.strip()
    if inner_loop_init is not None:
        inner_loop_init = inner_loop_init.strip()
    if inner is not None:
        inner = inner.strip()
    if result is not None:    
        result = result.strip()
    
    # Initialization
    code = []
    tab = '    '
    tabspace = '    '    

    # cdef loop indices
    idx = ', '.join(['i'+str(i) for i in range(ndim)])
    code.append(tab + 'cdef Py_ssize_t ' + idx)
    
    # cdef input array shape
    for dim in range(ndim):
        code.append(tab + "cdef int n%d = %s.shape[%d]" % (dim, inarr, dim))

    # cdef output array
    if (ndim > 1) and (axisNone is False):
        idx = range(ndim)
        del idx[axis]
        ns = ', '.join(['n'+str(i) for i in idx])
        code.append("%scdef np.npy_intp *dims = [%s]" % (tab,ns))
        y = "%scdef np.ndarray[np.%s_t, ndim=%d] %s = PyArray_EMPTY(%d, dims,"
        y += "\n                                              NPY_%s, 0)"
        code.append(y % (tab, dtype, ndim-1, outarr, ndim-1, dtype))
    
    # Pre loop init
    if outer_loop_init is not None:
        for x in outer_loop_init.split('\n'):
            code.append(tab + x)
    
    # Make loop
    loop = "%sfor i%d in range(n%d):"
    dims = range(ndim)
    if axis is not None:
        inner_axis = dims.pop(axis)
        dims.append(inner_axis)
    for dim in dims:

        if dim == dims[-1]:
            if inner_loop_init is not None:
                for x in inner_loop_init.split('\n'):
                    code.append(tab + x)
            code.append(loop % (tab, dim, dim))
            tab += tabspace
            idx = range(ndim)
            idx = map(str, idx)
            idx = ['i' + i for i in idx]
            idx = ', '.join(idx)
            for x in inner.split('\n'):
                x = x.replace('INDEX', idx)
                code.append(tab + x)
        else:
            code.append(loop % (tab, dim, dim))
            tab += tabspace

    # Save result
    if axisNone:
        for res in result.split('\n'):
            code.append(tabspace + res)
        code.append('')
    else:    
        if (result is not None) and (ndim != 1):
            idx = range(ndim)
            idx.remove(axis)
            idx = map(str, idx)
            idx = ['i' + i for i in idx]
            idx = ', '.join(idx)
            result = result.replace("INDEX", idx)
            for res in result.split('\n'):
                code.append(tab[:-len(tabspace)] + res)

    # Done
    code = '\n'.join(code)
    return code

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
