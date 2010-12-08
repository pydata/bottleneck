
def make(func, maxdim=3):
    codes = []
    codes.append(func['main'])
    for key in func['templates']:
        f = func['templates'][key]
        for code in template(name=f['name'],
                             top=f['top'],
                             init=f['init'],
                             inner=f['inner'],
                             result=f['result'],
                             inarr=f['inarr'],
                             outarr=f['outarr'],
                             returns=f['returns'],
                             dtypes=f['dtype'],
                             ndims=f['ndims'],
                             axisNone=f['axisNone']):
            codes.append(code)   
    fid = open(func['pyx_file'], 'w')
    fid.write(''.join(codes))
    fid.close()

def template(name, top, init, inner, result, inarr, outarr, returns, dtypes,
             ndims, axisNone):

    funcs = []
    for ndim in ndims:
        if axisNone:
            axes = [None]
        else:
            axes = range(ndim)
        for dtype in dtypes:
            for axis in axes:

                # Add new function to selector
                sname = name + "_%dd_%s_axis%s" % (ndim, dtype, str(axis))

                # Code template
                func = top
                
                # loop
                func += loopy(ndim, axis, dtype, init, inner, result,
                              inarr, outarr, axisNone)

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
    
    return ''.join(funcs)

def loopy(ndim, axis, dtype, init, inner, result, inarr, outarr, axisNone):
    
    # Check input
    if ndim < 1:
        raise ValueError("ndim(=%d) must be and integer greater than 0" % ndim)
    if (axis < 0) and (axis is not None):
        raise ValueError("`axis` must be a non-negative integer or None")
    if axis >= ndim:
        raise ValueError("`axis` must be less then `ndim`")

    if init is not None:
        init = init.strip()
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
    
    # Make loop
    loop = "%sfor i%d in range(n%d):"
    for dim in range(ndim):

        if dim == ndim - 1:
            if init is not None:
                for x in init.split('\n'):
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

def selector(name, dtypes, maxdim):

    selector = [name  + "_selector = {}"]
    select = name + "_selector[(%d,%s,%d)] = %s"
    funcs = []
    for ndim in range(1, maxdim + 1):
        for dtype in dtypes:
            for axis in range(ndim):

                # Add new function to selector
                sname = name + "_%dd_%s_axis%d" % (ndim, dtype, axis)
                selector.append(select % (ndim, dtype, axis, sname))

                # Code template
                func = top
                
                # loop
                func += loopy(ndim, axis, dtype, init, inner, result)

                # name, ndim, dtype, axis
                func = func.replace('NAME', name)
                func = func.replace('NDIM', str(ndim))
                func = func.replace('DTYPE', dtype)
                func = func.replace('AXIS', str(axis))

                # Returns
                func += '\n    '
                func += returns
                func += '\n'

                funcs.append(func)
    
    return ''.join(funcs), '\n'.join(selector)

