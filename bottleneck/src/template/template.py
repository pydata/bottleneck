import os
import re
import ast


def make_pyx():
    filenames = ['reduce.pyx', 'nonreduce.pyx', 'nonreduce_axis.pyx',
                 'move.pyx']
    dirpath = os.path.dirname(__file__)
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        with open(filepath, 'r') as f:
            src = f.read()
        src = expand_functions(src)
        filepath = os.path.join(dirpath, '..', 'auto_pyx', filename)
        with open(filepath, 'w') as f:
            f.write(src)


def expand_functions(src_str):
    FUNC = r'^def|cdef'
    DTYPE = r'\s*#\s*bn.dtypes\s*=\s*'
    DEC = r'^@'
    src_list = []
    dec_list = []
    lines = src_str.splitlines()
    nlines = len(lines)
    i = 0
    while i < nlines:
        line = lines[i]
        if re.match(DEC, line):
            dec_list.append(line)
            i += 1
            continue
        if re.match(FUNC, line):
            dtypes = []
            func_list = [line]
            i += 1
            while True:
                if i >= nlines:
                    line = '\n'.join(dec_list + func_list)
                    break
                line = lines[i]
                if re.match(DTYPE, line):
                    dtypes = re.sub(DTYPE, '', line)
                    dtypes = ast.literal_eval(dtypes)
                    line = None
                elif re.match(FUNC + '|' + DEC + '|^\S',line):
                    i -= 1
                    func_str = '\n'.join(dec_list + func_list)
                    dec_list = []
                    line = expand_dtypes(func_str, dtypes)
                    break
                else:
                    if line is not None:
                        func_list.append(line)
                i += 1
        src_list.append(line)
        i += 1
    src = '\n'.join(src_list)
    return src


def expand_dtypes(func_str, dtypes):
    if 'DTYPE' not in func_str:
        return func_str
    func_list = []
    for dtype in dtypes:
        f = func_str[:]
        for i, dt in enumerate(dtype):
            f = conditional_dtype(f, dtype)
            f = f.replace('DTYPE%d' % i, dt)
            if i > 0:
                f = f + '\n'
        func_list.append(f)
    return '\n'.join(func_list)


def conditional_dtype(src, dtype):
    IF_DTYPE = r'\s*if\s*DTYPE[0-9]\s*==\s*'
    BIGINT = 9999999
    ntab = 4
    lines = src.splitlines()
    nlines = len(lines)
    src_out = []
    i = 0
    n = BIGINT
    while i < nlines:
        line = lines[i]
        i += 1
        if re.match(IF_DTYPE, line):
            n = nindent(line)
            if is_target_dtype(line, dtype):
                cond = []
                while True:
                    if i >= nlines:
                        line = '\n'.join(cond)
                        break
                    line = lines[i]
                    if re.match(IF_DTYPE, line) or nindent(line) <= n:
                        i -= 1
                        line = '\n'.join(cond)
                        src_out.append(line)
                        break
                    else:
                        cond.append(line[ntab:])
                    i += 1
            else:
                i += 1

        else:
            if nindent(line) <= n:
                src_out.append(line)
                n = BIGINT
    src_out = '\n'.join(src_out)
    return src_out


def is_target_dtype(src_line, dtype):
    if_list = re.split('if\s*', src_line)
    cond = if_list[-1]
    cond = re.split(':', cond)[0]
    cond_list = re.split('\s*==\s*', cond)
    dtype_num = cond_list[0]
    num = int(dtype_num[-1])
    target_dtype = re.sub(r'\'|\"', '', cond_list[1])
    if dtype[num] == target_dtype:
        return True
    else:
        return False


def nindent(line):
    return len(line) - len(line.lstrip(' '))
