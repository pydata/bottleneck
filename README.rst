**July 26, 2019: Due to health problems, I can no longer maintain bottleneck. I hope that bottleneck is forked and a dominant fork emerges. Thanks to all who have reported issues and made PRs or just stopped by to say hello.**

.. image:: https://travis-ci.org/kwgoodman/bottleneck.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/bottleneck
.. image:: https://ci.appveyor.com/api/projects/status/github/kwgoodman/bottleneck?svg=true&passingText=passing&failingText=failing&pendingText=pending
    :target: https://ci.appveyor.com/project/kwgoodman/bottleneck
==========
Bottleneck
==========

Bottleneck is a collection of fast NumPy array functions written in C.

Let's give it a try. Create a NumPy array::

    >>> import numpy as np
    >>> a = np.array([1, 2, np.nan, 4, 5])

Find the nanmean::

    >>> import bottleneck as bn
    >>> bn.nanmean(a)
    3.0

Moving window mean::

    >>> bn.move_mean(a, window=2, min_count=1)
    array([ 1. ,  1.5,  2. ,  4. ,  4.5])

Benchmark
=========

Bottleneck comes with a benchmark suite::

    >>> bn.bench()
    Bottleneck performance benchmark
        Bottleneck 1.3.0; Numpy 1.16.0
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-fifth NaNs; float64 used

                  no NaN     no NaN      NaN       no NaN      NaN
                   (100,)  (1000,1000)(1000,1000)(1000,1000)(1000,1000)
                   axis=0     axis=0     axis=0     axis=1     axis=1
    nansum         83.7        1.5        1.9        2.3        2.4
    nanmean       245.3        2.3        2.4        3.4        2.9
    nanstd        316.8        2.0        2.1        2.6        2.5
    nanvar        298.9        1.8        2.1        2.7        2.5
    nanmin         63.8        0.7        1.9        1.0        3.3
    nanmax         53.8        0.7        1.8        0.8        2.6
    median        155.1        1.3        6.0        1.1        6.2
    nanmedian     166.7        7.3        8.4        7.2        8.5
    ss             35.4        1.2        1.2        1.6        1.6
    nanargmin     105.2       12.2        3.0        2.3        5.8
    nanargmax     123.5        3.1        6.6        2.3        5.7
    anynan         24.0        0.3       41.5        0.5       28.5
    allnan         51.6      147.5      129.4      119.3       57.8
    rankdata       67.5        2.5        2.4        2.9        2.9
    nanrankdata    71.1        2.7        2.6        3.2        3.1
    partition       4.1        1.2        2.7        1.0        1.4
    argpartition    3.7        1.1        1.4        1.1        1.6
    replace        15.1        1.5        1.5        1.5        1.5
    push         3199.6        3.1        9.9       21.2       16.2
    move_sum     4901.4       75.1      116.5      201.3      201.3
    move_mean   11550.7      109.8      182.6      243.5      252.4
    move_std    11910.9       98.0      195.0      140.5      248.8
    move_var    14669.2      129.3      237.2      211.7      316.8
    move_min     2484.2       19.5       35.6       25.0       43.8
    move_max     2401.6       20.0       36.0       25.6       44.1
    move_argmin  4531.0       41.8       74.8       42.9       83.8
    move_argmax  4339.2       42.7       74.8       43.5       82.2
    move_median  2752.2      150.2      151.0      153.2      154.7
    move_rank    1585.6        1.9        2.0        2.5        2.7

You can also run a detailed benchmark for a single function using, for
example, the command::

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
 docs                 https://kwgoodman.github.io/bottleneck-doc
 code                 https://github.com/kwgoodman/bottleneck
 mailing list         https://groups.google.com/group/bottle-neck
===================   ========================================================

License
=======

Bottleneck is distributed under a Simplified BSD license. See the LICENSE file
for details.

Install
=======

Requirements:

======================== ====================================================
Bottleneck               Python 2.7, 3.5, 3.6; NumPy 1.16.0
Compile                  gcc, clang, MinGW or MSVC
Unit tests               nose
Documentation            sphinx, numpydoc
======================== ====================================================

To install Bottleneck on GNU/Linux, Mac OS X, et al.::

    $ sudo python setup.py install

To install bottleneck on Windows, first install MinGW and add it to your
system path. Then install Bottleneck with the command::

    python setup.py install --compiler=mingw32

Alternatively, you can use the Windows binaries created by Christoph Gohlke:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck

Unit tests
==========

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 169 tests in 57.205s
    OK
    <nose.result.TextTestResult run=169 errors=0 failures=0>
