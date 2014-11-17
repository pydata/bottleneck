import os
import re


def make_pyx():
    template('reduce.pyx')


def template(template_filename,
             dtypes=[['float64'], ['float32'], ['int64'], ['int32']]):

    dirpath = os.path.dirname(__file__)
    filename = os.path.join(dirpath, template_filename)
    with open(filename, 'r') as f:
        src_str = f.read()

    src_list = []
    lines = src_str.splitlines()
    nlines = len(lines)
    i = 0
    while i < nlines:
        line = lines[i]
        if re.match(r'^def|cdef', line):
            func_list = [line]
            i += 1
            while True:
                if i >= nlines:
                    line = '\n'.join(func_list)
                    break
                line = lines[i]
                if re.match(r'^def|cdef', line):
                    i -= 1
                    func_str = '\n'.join(func_list)
                    line = expand_dtypes(func_str, dtypes)
                    break
                else:
                    func_list.append(line)
                i += 1
        src_list.append(line)
        i += 1

    src = '\n'.join(src_list)
    print src

    filename = os.path.join(dirpath, '..', 'auto_pyx', 'reduce.pyx')
    with open(filename, 'w') as f:
        f.write(src)


def expand_dtypes(func_str, dtypes):
    DTYPE = 'DTYPE'
    if DTYPE not in func_str:
        return func_str
    func_list = []
    for dtype in dtypes:
        f = func_str[:]
        for i, dt in enumerate(dtype):
            f = f.replace('DTYPE%d' % i, dt)
            func_list.append(f)
    return '\n'.join(func_list)


if __name__ == '__main__':
    make_pyx()
