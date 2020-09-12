""" Based on numpy's approach to exposing compiler features via a config header.
Unfortunately that file is not exposed, so re-implement the portions we need.
"""
import os
import sys
import textwrap
from distutils.command.config import config as Config
from typing import List

OPTIONAL_FUNCTION_ATTRIBUTES = [
    ("HAVE_ATTRIBUTE_OPTIMIZE_OPT_3", '__attribute__((optimize("O3")))')
]

OPTIONAL_HEADERS = [("HAVE_SSE2", "emmintrin.h")]

OPTIONAL_INTRINSICS = [
    ("HAVE___BUILTIN_ISNAN", "__builtin_isnan", "0."),
    ("HAVE_ISNAN", "isnan", "0."),
    ("HAVE__ISNAN", "_isnan", "0."),
]


def get_python_header_include() -> List[str]:
    if sys.platform == "win32":
        suffix = ["include"]
    else:
        suffix = ["include", "python" + sys.version[:3] + sys.abiflags]

    results = []
    for prefix in [sys.prefix, sys.exec_prefix]:
        results.append(os.path.join(prefix, *suffix))

    return results


def _get_compiler_list(cmd: Config) -> str:
    """Return the compiler command as a list of strings. Distutils provides a
    wildly inconsistent API here:
      - UnixCCompiler returns a list
      - MSVCCompiler intentionally doesn't set this variable
      - CygwinCompiler returns a string

    As we are focused on identifying gcc vs clang right now, we ignore MSVC's
    bad result and convert all results into lists of strings
    """
    compiler = getattr(cmd.compiler, "compiler", "")
    if isinstance(compiler, str):
        compiler = compiler.split()
    return compiler


def is_gcc(cmd: Config) -> bool:
    return any("gcc" in x for x in _get_compiler_list(cmd))


def is_clang(cmd: Config) -> bool:
    return any("clang" in x for x in _get_compiler_list(cmd))


def check_inline(cmd: Config) -> str:
    """Return the inline identifier (may be empty)."""
    body = textwrap.dedent(
        """
        #ifndef __cplusplus
        static %(inline)s int static_func (void)
        {
            return 0;
        }
        %(inline)s int nostatic_func (void)
        {
            return 0;
        }
        #endif
        int main(void) {
            int r1 = static_func();
            int r2 = nostatic_func();
            return r1 + r2;
        }
        """
    )

    for kw in ["inline", "__inline__", "__inline"]:
        st = cmd.try_compile(body % {"inline": kw}, None, None)
        if st:
            return kw

    return ""


def check_gcc_function_attribute(cmd: Config, attribute: str, name: str) -> bool:
    """Return True if the given function attribute is supported."""
    if is_gcc(cmd):
        pragma = '#pragma GCC diagnostic error "-Wattributes"'
    elif is_clang(cmd):
        pragma = '#pragma clang diagnostic error "-Wattributes"'
    else:
        pragma = ""

    body = (
        textwrap.dedent(
            """
        %s

        int %s %s(void*);

        int main(void)
        {
            return 0;
        }
        """
        )
        % (pragma, attribute, name)
    )
    if cmd.try_compile(body, None, None):
        return True
    else:
        return False


def check_gcc_header(cmd: Config, header: str) -> bool:
    return cmd.check_header(
        header,
        include_dirs=get_python_header_include(),
    )


def check_gcc_intrinsic(cmd: Config, intrinsic: str, value: str) -> bool:
    """Return True if the given intrinsic is supported."""
    body = (
        textwrap.dedent(
            """
        int check(void) {
            return %s(%s);
        }

        int main(void)
        {
            return check();
        }
        """
        )
        % (intrinsic, value)
    )
    if cmd.try_link(body, headers=["math.h"]):
        return True
    else:
        return False


def create_config_h(config: Config) -> None:
    dirname = os.path.dirname(__file__)
    config_h = os.path.join(dirname, "bn_config.h")

    if (
        os.path.exists(config_h)
        and os.stat(__file__).st_mtime < os.stat(config_h).st_mtime
    ):
        return

    if not check_gcc_header(config, "Python.h"):
        raise ValueError(
            """Cannot compile a trivial program with Python.h! Please check the following:
 - A supported compiler is installed
 - Python development libraries are installed

 For detailed installation instructions, please see:
 https://bottleneck.readthedocs.io/en/latest/installing.html
"""
        )

    output = []

    for config_attr, func_attr in OPTIONAL_FUNCTION_ATTRIBUTES:
        if check_gcc_function_attribute(config, func_attr, config_attr.lower()):
            output.append((config_attr, "1"))
        else:
            output.append((config_attr, "0"))

    for config_attr, header in OPTIONAL_HEADERS:
        if check_gcc_header(config, header):
            output.append((config_attr, "1"))
        else:
            output.append((config_attr, "0"))

    for config_attr, intrinsic, value in OPTIONAL_INTRINSICS:
        if check_gcc_intrinsic(config, intrinsic, value):
            output.append((config_attr, "1"))
        else:
            output.append((config_attr, "0"))

    inline_alias = check_inline(config)

    with open(config_h, "w") as f:
        for setting in output:
            f.write("#define {} {}\n".format(*setting))

        if inline_alias == "inline":
            f.write("/* undef inline */\n")
        else:
            f.write("#define inline {}\n".format(inline_alias))

        # ISO C requires every translation unit to have 1+ declarations
        f.write("typedef int _make_iso_compilers_happy;\n")
