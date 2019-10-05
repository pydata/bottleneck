from cStringIO import StringIO
import sys
import os

import bottleneck as bn


def update_readme():

    # run benchmark suite while capturing output; indent
    with Capturing() as bench_list:
        bn.bench()
    bench_list = ["    " + b for b in bench_list]

    # read readme
    cwd = os.path.dirname(__file__)
    readme_path = os.path.join(cwd, "../README.rst")
    with open(readme_path) as f:
        readme_list = f.readlines()
    readme_list = [r.strip("\n") for r in readme_list]

    # remove old benchmark result from readme
    idx1 = readme_list.index("    Bottleneck performance benchmark")
    idx2 = [i for i, line in enumerate(readme_list) if line == ""]
    idx2 = [i for i in idx2 if i > idx1]
    idx2 = idx2[1]
    del readme_list[idx1:idx2]

    # insert new benchmark result into readme; remove trailing whitespace
    readme_list = readme_list[:idx1] + bench_list + readme_list[idx1:]
    readme_list = [r.rstrip() for r in readme_list]

    # replace readme file
    os.remove(readme_path)
    with open(readme_path, "w") as f:
        f.write("\n".join(readme_list))


# ---------------------------------------------------------------------------
# Capturing class taken from
# http://stackoverflow.com/questions/16571150/
# how-to-capture-stdout-output-from-a-python-function-call


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


if __name__ == "__main__":
    update_readme()
