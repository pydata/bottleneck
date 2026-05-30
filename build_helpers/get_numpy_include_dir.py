# adapted from scipy's meson.build
import os
from contextlib import suppress

import numpy as np


def main(argv: list[str] | None = None) -> int:
    assert not argv
    incdir = np.get_include()

    with suppress(Exception):
        # when things are split across drives on Windows,
        # there is no relative path and an exception gets raised.
        incdir = os.path.relpath(np.get_include())
    print(incdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
