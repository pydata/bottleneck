.. image:: https://travis-ci.org/kwgoodman/bottleneck.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/bottleneck
==========
Bottleneck
==========

Bottleneck is a collection of fast NumPy array functions written in Cython.

Let's give it a try. Create a NumPy array::

    >>> import numpy as np
    >>> arr = np.array([1, 2, np.nan, 4, 5])

Find the nanmean::

    >>> import bottleneck as bn
    >>> bn.nanmean(arr)
    3.0

Moving window mean::

    >>> bn.move_mean(arr, window=2, min_count=1)
    array([ 1. ,  1.5,  2. ,  4. ,  4.5])

Benchmark
=========

Bottleneck comes with a benchmark suite::

    >>> bn.bench()
    Bottleneck performance benchmark
        Bottleneck 1.1.0; Numpy 1.11.0
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum         61.5        2.3       61.1        7.4
        nanmean       219.4        3.5      219.3        8.3
        nanstd        350.5        2.7      355.3        7.1
        nanvar        363.6        2.7      357.3        7.1
        nanmin         51.0        1.0       51.1        1.4
        nanmax         51.7        1.0       51.6        2.1
        median         92.4        0.8      110.5        5.3
        nanmedian      90.1        3.9      109.8       12.0
        ss             29.5        1.6       29.1        1.6
        nanargmin      91.5        3.0       93.0        7.5
        nanargmax      92.9        2.9       93.8        8.3
        anynan         24.4        0.7       26.2       55.2
        allnan         26.4       62.3       26.2       59.0
        rankdata       44.2        1.5       44.9        2.2
        nanrankdata    57.0       26.6       52.5       41.9
        partsort        5.9        0.9        5.9        1.1
        argpartsort     2.8        0.9        3.0        0.7
        replace        10.0        1.1       10.0        1.1
        push          207.6       20.2      204.6       25.7
        move_sum      307.0      154.0      307.6      407.8
        move_mean     750.2      149.7      760.0      539.2
        move_std     1070.9       77.8     1093.5      370.4
        move_var     1035.1      107.2     1059.8      367.6
        move_min      218.6       23.1      219.6       51.4
        move_max      239.7       23.0      243.8       92.3
        move_argmin   391.5       40.6      405.9      243.6
        move_argmax   386.9       41.3      404.5      279.0
        move_median   508.7       42.1      501.8      160.2
        move_rank     575.7        2.9      630.2       11.1

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtypes result in calls to slower, unaccelerated
functions. In the rare case of a byte-swapped input array (e.g. a big-endian
array on a little-endian operating system) the function will not be
accelerated regardless of dtype.

Where
=====

===================   ========================================================
 download             http://pypi.python.org/pypi/Bottleneck
 docs                 http://berkeleyanalytics.com/bottleneck
 code                 http://github.com/kwgoodman/bottleneck
 mailing list         http://groups.google.com/group/bottle-neck
===================   ========================================================

License
=======

Bottleneck is distributed under a Simplified BSD license. See the LICENSE file
for details.

Install
=======

Requirements:

======================== ====================================================
Bottleneck               Python 2.7, 3.4, 3.5; NumPy 1.11.0
Compile                  gcc or clang or MinGW
Unit tests               nose
Cython                   Optional for released version of Bottleneck
======================== ====================================================

If Cython is installed on your computer then you can install either a released
version of Bottleneck (PyPI) or a development version (GitHub).

If Cython is not installed on your computer then you can only install a
released version of Bottleneck (PyPI). Cython is not required because the
Cython files have already been converted to C source files in Bottleneck
releases.

To install Bottleneck on GNU/Linux, Mac OS X, et al.::

    $ sudo python setup.py install

To install bottleneck on Windows, first install MinGW and add it to your
system path. Then install Bottleneck with the commands::

    python setup.py install --compiler=mingw32

Alternatively, you can use the Windows binaries created by Christoph Gohlke:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck

Unit tests
==========

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 127 tests in 18.978s
    OK
    <nose.result.TextTestResult run=127 errors=0 failures=0>
