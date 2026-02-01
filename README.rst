.. image:: https://github.com/pydata/bottleneck/actions/workflows/ci.yml/badge.svg?branch=master
    :target: https://github.com/pydata/bottleneck/actions/workflows/ci.yml


==========
Bottleneck
==========

Bottleneck is a collection of fast NumPy array functions written in C.

Let's give it a try. Create a NumPy array:

.. code-block:: pycon

    >>> import numpy as np
    >>> a = np.array([1, 2, np.nan, 4, 5])

Find the nanmean:

.. code-block:: pycon

    >>> import bottleneck as bn
    >>> bn.nanmean(a)
    3.0

Moving window mean:

.. code-block:: pycon

    >>> bn.move_mean(a, window=2, min_count=1)
    array([ 1. ,  1.5,  2. ,  4. ,  4.5])

Benchmark
=========

Bottleneck comes with a benchmark suite:

.. code-block:: pycon

    >>> bn.bench()
    Bottleneck performance benchmark
        Bottleneck 1.6.0.post0.dev32; Numpy 2.4.2
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-fifth NaNs; float64 used

                no NaN     no NaN      NaN       no NaN      NaN
                (100,)  (1000,1000)(1000,1000)(1000,1000)(1000,1000)
                axis=0     axis=0     axis=0     axis=1     axis=1
    nansum         12.2        0.4        2.0        0.4        2.0
    nanmean        29.8        0.8        2.3        0.5        2.2
    nanstd         34.2        0.8        2.2        0.7        2.1
    nanvar         32.9        0.8        2.2        0.7        2.1
    nanmin         12.7        0.1        0.1        0.1        0.1
    nanmax         12.8        0.1        0.1        0.1        0.1
    median         38.7        1.1        6.7        1.0        6.5
    nanmedian      38.4        2.1        2.2        1.9        2.1
    ss              5.2        0.3        0.3        0.3        0.3
    nanargmin      25.9        1.2        3.2        0.9        2.8
    nanargmax      26.0        1.2        3.2        0.9        2.8
    anynan          8.1        0.3       42.1        0.3       35.7
    allnan         11.6       58.4       58.6       47.1       47.5
    rankdata       14.9        1.4        1.4        1.5        1.5
    nanrankdata    16.4        1.5        1.4        1.6        1.5
    partition       2.0        1.1        1.6        1.0        1.5
    argpartition    2.4        1.3        1.8        1.2        1.8
    replace         7.4        2.9        2.9        2.9        2.9
    push         1453.8       16.2        8.8       24.1       10.3
    move_sum     1159.7       89.4      143.3      168.6      192.1
    move_mean    2575.8      182.0      171.7      214.2      202.4
    move_std     2863.9      137.4      274.5      145.1      310.7
    move_var     2792.3      137.9      279.7      154.1      325.8
    move_min      690.7        4.1        4.2        5.2        5.2
    move_max      659.9        4.2        4.2        5.2        5.2
    move_argmin  1369.1       33.7       77.5       35.7       83.5
    move_argmax  1344.7       32.8       78.2       35.9       83.3
    move_median   686.6      153.5      156.9      156.0      159.8
    move_rank     502.0        1.9        2.0        1.8        2.1

You can also run a detailed benchmark for a single function using, for
example, the command:

.. code-block:: pycon

    >>> bn.bench_detailed("move_median", fraction_nan=0.3)

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtypes result in calls to slower, unaccelerated
functions. In the rare case of a byte-swapped input array (e.g. a big-endian
array on a little-endian operating system) the function will not be
accelerated regardless of dtype.

Where
=====

===================   ========================================================
 download             https://pypi.python.org/pypi/Bottleneck
 docs                 https://bottleneck.readthedocs.io
 code                 https://github.com/pydata/bottleneck
 mailing list         https://groups.google.com/group/bottle-neck
===================   ========================================================

License
=======

Bottleneck is distributed under a Simplified BSD license. See the LICENSE file
and LICENSES directory for details.

Install
=======

Bottleneck provides binary wheels on PyPI for all the most common platforms.
Binary packages are also available in conda-forge. We recommend installing binaries
with ``pip``, ``uv``, ``conda`` or similar - it's faster and easier than building
from source.

Installing from source
----------------------

Requirements:

======================== ============================================================================
Bottleneck               Python >3.9; NumPy 1.16.0+
Compile                  gcc, clang, MinGW or MSVC
Unit tests               pytest
Documentation            sphinx, numpydoc
======================== ============================================================================

To install Bottleneck on Linux, Mac OS X, et al.:

.. code-block:: console

    $ pip install .

To install bottleneck on Windows, first install MinGW and add it to your
system path. Then install Bottleneck with the command:

.. code-block:: console

    $ python setup.py install --compiler=mingw32

Unit tests
==========

After you have installed Bottleneck, run the suite of unit tests:

.. code-block:: pycon

  In [1]: import bottleneck as bn

  In [2]: bn.test()
  ============================= test session starts =============================
  platform linux -- Python 3.7.4, pytest-4.3.1, py-1.8.0, pluggy-0.12.0
  hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/home/chris/code/bottleneck/.hypothesis/examples')
  rootdir: /home/chris/code/bottleneck, inifile: setup.cfg
  plugins: openfiles-0.3.2, remotedata-0.3.2, doctestplus-0.3.0, mock-1.10.4, forked-1.0.2, cov-2.7.1, hypothesis-4.32.2, xdist-1.26.1, arraydiff-0.3
  collected 190 items

  bottleneck/tests/input_modification_test.py ........................... [ 14%]
  ..                                                                      [ 15%]
  bottleneck/tests/list_input_test.py .............................       [ 30%]
  bottleneck/tests/move_test.py .................................         [ 47%]
  bottleneck/tests/nonreduce_axis_test.py ....................            [ 58%]
  bottleneck/tests/nonreduce_test.py ..........                           [ 63%]
  bottleneck/tests/reduce_test.py ....................................... [ 84%]
  ............                                                            [ 90%]
  bottleneck/tests/scalar_input_test.py ..................                [100%]

  ========================= 190 passed in 46.42 seconds =========================
  Out[2]: True

If developing in the git repo, simply run ``py.test``
