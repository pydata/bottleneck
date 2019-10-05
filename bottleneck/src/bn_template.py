import os
import re
import ast


def make_c_files():
    modules = ["reduce", "move", "nonreduce", "nonreduce_axis"]
    dirpath = os.path.dirname(__file__)
    for module in modules:
        template_file = os.path.join(dirpath, module + "_template.c")
        target_file = os.path.join(dirpath, module + ".c")

        if (
            os.path.exists(target_file)
            and os.stat(template_file).st_mtime < os.stat(target_file).st_mtime
        ):
            continue

        with open(template_file, "r") as f:
            src_str = f.read()
        src_str = template(src_str)
        if len(src_str) and src_str[-1] != "\n":
            src_str += "\n"
        with open(target_file, "w") as f:
            f.write(src_str)


def template(src_str):
    src_list = src_str.splitlines()
    src_list = repeat_templating(src_list)
    src_list = dtype_templating(src_list)
    src_list = string_templating(src_list)
    src_str = "\n".join(src_list)
    src_str = re.sub(r"\n\s*\n\s*\n", r"\n\n", src_str)
    return src_str


# repeat --------------------------------------------------------------------

REPEAT_BEGIN = r"^/\*\s*repeat\s*=\s*"
REPEAT_END = r"^/\*\s*repeat end"
COMMENT_END = r".*\*\/.*"


def repeat_templating(lines):
    index = 0
    while True:
        idx0, idx1 = next_block(lines, index, REPEAT_BEGIN, REPEAT_END)
        if idx0 is None:
            break
        func_list = lines[idx0:idx1]
        func_list = expand_functions_repeat(func_list)
        # the +1 below is to skip the /* repeat end */ line
        lines = lines[:idx0] + func_list + lines[idx1 + 1 :]
        index = idx0
    return lines


def expand_functions_repeat(lines):
    idx = first_occurence(COMMENT_END, lines)
    repeat_dict = repeat_info(lines[: idx + 1])
    lines = lines[idx + 1 :]
    func_str = "\n".join(lines)
    func_list = expand_repeat(func_str, repeat_dict)
    return func_list


def repeat_info(lines):
    line = "".join(lines)
    repeat = re.findall(r"\{.*\}", line)
    repeat_dict = ast.literal_eval(repeat[0])
    return repeat_dict


def expand_repeat(func_str, repeat_dict):
    nrepeats = [len(repeat_dict[key]) for key in repeat_dict]
    if len(set(nrepeats)) != 1:
        raise ValueError("All repeat lists must be the same length")
    nrepeat = nrepeats[0]
    func_list = []
    for i in range(nrepeat):
        f = func_str[:]
        for key in repeat_dict:
            f = f.replace(key, repeat_dict[key][i])
        func_list.append("\n" + f)
    func_list = ("".join(func_list)).splitlines()
    return func_list


# dtype ---------------------------------------------------------------------

DTYPE_BEGIN = r"^/\*\s*dtype\s*=\s*"
DTYPE_END = r"^/\*\s*dtype end"


def dtype_templating(lines):
    index = 0
    while True:
        idx0, idx1 = next_block(lines, index, DTYPE_BEGIN, DTYPE_END)
        if idx0 is None:
            break
        func_list = lines[idx0:idx1]
        func_list = expand_functions_dtype(func_list)
        # the +1 below is to skip the /* dtype end */ line
        lines = lines[:idx0] + func_list + lines[idx1 + 1 :]
        index = idx0
    return lines


def expand_functions_dtype(lines):
    idx = first_occurence(COMMENT_END, lines)
    dtypes = dtype_info(lines[: idx + 1])
    lines = lines[idx + 1 :]
    func_str = "\n".join(lines)
    func_list = expand_dtypes(func_str, dtypes)
    return func_list


def dtype_info(lines):
    line = "".join(lines)
    dtypes = re.findall(r"\[.*\]", line)
    if len(dtypes) != 1:
        raise ValueError("expecting exactly one dtype specification")
    dtypes = ast.literal_eval(dtypes[0])
    return dtypes


def expand_dtypes(func_str, dtypes):
    if "DTYPE" not in func_str:
        raise ValueError("cannot find dtype marker")
    func_list = []
    for dtype in dtypes:
        f = func_str[:]
        for i, dt in enumerate(dtype):
            f = f.replace("DTYPE%d" % i, dt)
            if i > 0:
                f = f + "\n"
        func_list.append("\n\n" + f)
    return func_list


# multiline strings ---------------------------------------------------------

STRING_BEGIN = r".*MULTILINE STRING BEGIN.*"
STRING_END = r".*MULTILINE STRING END.*"


def string_templating(lines):
    index = 0
    while True:
        idx0, idx1 = next_block(lines, index, STRING_BEGIN, STRING_END)
        if idx0 is None:
            break
        str_list = lines[idx0 + 1 : idx1]
        str_list = quote_string(str_list)
        lines = lines[:idx0] + str_list + lines[idx1 + 1 :]
        index = idx0
    return lines


def quote_string(lines):
    for i in range(len(lines)):
        lines[i] = '"' + lines[i] + r"\n" + '"'
    lines[-1] = lines[-1] + ";"
    return lines


# utility -------------------------------------------------------------------


def first_occurence(pattern, lines):
    for i in range(len(lines)):
        if re.match(pattern, lines[i]):
            return i
    raise ValueError("`pattern` not found")


def next_block(lines, index, begine_pattern, end_pattern):
    idx = None
    for i in range(index, len(lines)):
        line = lines[i]
        if re.match(begine_pattern, line):
            idx = i
        elif re.match(end_pattern, line):
            if idx is None:
                raise ValueError("found end of function before beginning")
            return idx, i
    return None, None
