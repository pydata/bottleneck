import os
import re
import ast

DTYPE_BEGIN = r'^/\*\s*dtype\s*=\s*'
DTYPE_END = r'^/\*\s*dtype end'
COMMENT_END = r'.*\*\/.*'


def make_c_files():
    modules = ['reduce2']
    dirpath = os.path.dirname(__file__)
    for module in modules:
        filepath = os.path.join(dirpath, module + '_template.c')
        with open(filepath, 'r') as f:
            src = f.read()
        src = expand_functions(src)
        filepath = os.path.join(dirpath, module + '.c')
        with open(filepath, 'w') as f:
            f.write(src)


def expand_functions(src_str):
    src_list = src_str.splitlines()
    while True:
        idx0, idx1 = next_dtype_block(src_list)
        if idx0 is None:
            break
        func_list = src_list[idx0:idx1]
        func_list = expand_function(func_list)
        # the +1 below is to skip the /* dtype end */ line
        src_list = src_list[:idx0] + func_list + src_list[idx1+1:]
    return '\n'.join(src_list)


def next_dtype_block(lines):
    idx = None
    for i in range(len(lines)):
        line = lines[i]
        if re.match(DTYPE_BEGIN, line):
            idx = i
        elif re.match(DTYPE_END, line):
            if idx is None:
                raise ValueError("found end of function before beginning")
            return idx, i
    return None, None


def expand_function(lines):
    idx = first_occurence(COMMENT_END, lines)
    dtypes = dtype_info(lines[:idx + 1])
    lines = lines[idx + 1:]
    func_str = '\n'.join(lines)
    func_list = expand_dtypes(func_str, dtypes)
    return func_list


def first_occurence(pattern, lines):
    for i in range(len(lines)):
        if re.match(pattern, lines[i]):
            return i
    raise ValueError("end of comment not found")


def dtype_info(lines):
    line = 'n'.join(lines)
    dtypes = re.findall(r'\[.*\]', line)
    if len(dtypes) != 1:
        raise ValueError("expecting exactly one dtype specification")
    dtypes = ast.literal_eval(dtypes[0])
    return dtypes


def expand_dtypes(func_str, dtypes):
    if 'DTYPE' not in func_str:
        raise ValueError("cannot find dtype marker")
    func_list = []
    for dtype in dtypes:
        f = func_str[:]
        for i, dt in enumerate(dtype):
            f = f.replace('DTYPE%d' % i, dt)
            if i > 0:
                f = f + '\n'
        func_list.append('\n\n' + f)
    return func_list
