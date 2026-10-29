from pathlib import Path

import pytest

from ..template import main

SRC_DIR = Path(__file__).parents[2] / "bottleneck" / "src"


@pytest.fixture(params=sorted(SRC_DIR.glob("*.c.template")))
def template_file(request):
    return request.param


def test_make_c_file(template_file, tmp_path) -> None:
    target = tmp_path / template_file.stem
    retcode = main([str(template_file), "-o", str(target)])
    assert retcode == 0
    assert target.is_file()


DATA_DIR = Path(__file__).parent / "data" / "template_test"


def test_known_result(tmp_path) -> None:
    ref = DATA_DIR / "truth.c"
    target = tmp_path / "test.c"
    retcode = main([str(DATA_DIR / "test.c.template"), "-o", str(target)])
    assert retcode == 0
    assert target.read_text() == ref.read_text()
