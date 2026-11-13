#!/usr/bin/env python3
import ast
import re
from pathlib import Path
from re import Pattern


def make_c_file(
    template_file: Path,
    output_file: Path,
) -> None:
    assert template_file.suffixes == [".c", ".template"]
    target_name = template_file.stem  # trim the last suffix ('.template')
    assert target_name.endswith(".c"), template_file
    assert target_name == output_file.name, output_file

    # if (
    #     target_file.exists()
    #     and template_file.stat().st_mtime < target_file.stat().st_mtime
    # ):
    #     continue

    src_str = f'#line 1 "{template_file.name}"\n' + template(template_file.read_text())
    if len(src_str) and src_str[-1] != "\n":
        src_str += "\n"
    output_file.write_text(src_str)
    print(f"wrote {output_file}")


def template(src_str: str) -> str:
    src_list = src_str.splitlines()
    line_numbers = []
    last_empty_ind = 0
    for i, L in enumerate(src_list):
        if L.strip().endswith("{") and not L.startswith(" "):
            line_numbers.append(last_empty_ind)

        if len(L.strip()) == 0 or "*/" in L:
            last_empty_ind = i + 1

    distinct_line_numbers = set(line_numbers)
    new_src_list = []
    for i, L in enumerate(src_list):
        if i in distinct_line_numbers:
            new_src_list.append(f"#line {i + 1}")
        new_src_list.append(L)

    src_list = repeat_templating(new_src_list)
    src_list = dtype_templating(src_list)
    src_list = string_templating(src_list)
    src_str = "\n".join(src_list)
    src_str = re.sub(r"\n\s*\n\s*\n", r"\n\n", src_str)
    return src_str


# repeat --------------------------------------------------------------------

REPEAT_BEGIN = re.compile(r"^/\*\s*repeat\s*=\s*")
REPEAT_END = re.compile(r"^/\*\s*repeat end")
COMMENT_END = re.compile(r".*\*\/.*")


def repeat_templating(lines: list[str]) -> list[str]:
    index = 0
    while True:
        idx0, idx1 = next_block(lines, index, REPEAT_BEGIN, REPEAT_END)
        if idx0 is None or idx1 is None:
            break
        func_list = lines[idx0:idx1]
        func_list = expand_functions_repeat(func_list)
        # the +1 below is to skip the /* repeat end */ line
        lines = lines[:idx0] + func_list + lines[idx1 + 1 :]
        index = idx0
    return lines


def expand_functions_repeat(lines: list[str]) -> list[str]:
    idx = first_occurence(COMMENT_END, lines)
    repeat_dict = repeat_info(lines[: idx + 1])
    lines = lines[idx + 1 :]
    func_str = "\n".join(lines)
    func_list = expand_repeat(func_str, repeat_dict)
    return func_list


def repeat_info(lines: list[str]) -> dict[str, str]:
    line = "".join(lines)
    repeat = re.findall(r"\{.*\}", line)
    repeat_dict: dict[str, str] = ast.literal_eval(repeat[0])
    return repeat_dict


def expand_repeat(func_str: str, repeat_dict: dict[str, str]) -> list[str]:
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

DTYPE_BEGIN = re.compile(r"^/\*\s*dtype\s*=\s*")
DTYPE_END = re.compile(r"^/\*\s*dtype end")


def dtype_templating(lines: list[str]) -> list[str]:
    index = 0
    while True:
        idx0, idx1 = next_block(lines, index, DTYPE_BEGIN, DTYPE_END)
        if idx0 is None or idx1 is None:
            break
        func_list = lines[idx0:idx1]
        func_list = expand_functions_dtype(func_list)
        # the +1 below is to skip the /* dtype end */ line
        lines = lines[:idx0] + func_list + lines[idx1 + 1 :]
        index = idx0
    return lines


def expand_functions_dtype(lines: list[str]) -> list[str]:
    idx = first_occurence(COMMENT_END, lines)
    dtypes = dtype_info(lines[: idx + 1])
    lines = lines[idx + 1 :]
    func_str = "\n".join(lines)
    func_list = expand_dtypes(func_str, dtypes)
    return func_list


def dtype_info(lines: list[str]) -> list[str]:
    line = "".join(lines)
    dtypes = re.findall(r"\[.*\]", line)
    if len(dtypes) != 1:
        raise ValueError("expecting exactly one dtype specification")
    dtypes = ast.literal_eval(dtypes[0])
    return dtypes


def expand_dtypes(func_str: str, dtypes: list[str]) -> list[str]:
    if "DTYPE" not in func_str:
        raise ValueError("cannot find dtype marker")
    func_list = []
    for dtype in dtypes:
        f = func_str[:]
        for i, dt in enumerate(dtype):
            f = f.replace(f"DTYPE{i}", dt)
            if i > 0:
                f = f + "\n"
        func_list.append("\n\n" + f)
    return func_list


# multiline strings ---------------------------------------------------------

STRING_BEGIN = re.compile(r".*MULTILINE STRING BEGIN.*")
STRING_END = re.compile(r".*MULTILINE STRING END.*")


def string_templating(lines: list[str]) -> list[str]:
    index = 0
    while True:
        idx0, idx1 = next_block(lines, index, STRING_BEGIN, STRING_END)
        if idx0 is None or idx1 is None:
            break
        str_list = lines[idx0 + 1 : idx1]
        str_list = quote_string(str_list)
        lines = lines[:idx0] + str_list + lines[idx1 + 1 :]
        index = idx0
    return lines


def quote_string(lines: list[str]) -> list[str]:
    for i in range(len(lines)):
        lines[i] = '"' + lines[i] + r"\n" + '"'
    lines[-1] = lines[-1] + ";"
    return lines


# utility -------------------------------------------------------------------


def first_occurence(pattern: Pattern[str], lines: list[str]) -> int:
    for i in range(len(lines)):
        if re.match(pattern, lines[i]):
            return i
    raise ValueError("`pattern` not found")


def next_block(
    lines: list[str], index: int, begin_pattern: Pattern[str], end_pattern: Pattern[str]
) -> tuple[int | None, int | None]:
    idx = None
    for i in range(index, len(lines)):
        line = lines[i]
        if re.match(begin_pattern, line):
            idx = i
        elif re.match(end_pattern, line):
            if idx is None:
                raise ValueError("found end of function before beginning")
            return idx, i
    return None, None


def main(argv: list[str] | None = None) -> int:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "template_file",
        type=Path,
        help="path to template file",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        help="path to output file",
    )
    args = parser.parse_args(argv)
    make_c_file(
        template_file=args.template_file,
        output_file=args.output_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
