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
        Bottleneck 1.1.0dev; Numpy 1.11.0
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum         40.9        2.3       40.2        7.5
        nanmean       146.9        3.3      146.4        8.1
        nanstd        256.0        2.6      255.5        6.8
        nanvar        245.1        2.6      242.7        6.8
        nanmin         32.4        1.0       32.2        1.4
        nanmax         34.1        1.0       34.0        2.1
        median         77.1        0.8       95.3        5.5
        nanmedian      70.9        3.8       80.2       11.8
        ss             18.2        1.6       17.9        1.6
        nanargmin      57.4        3.1       58.2        7.7
        nanargmax      58.4        3.1       58.5        8.6
        anynan         13.7        0.7       14.3       53.4
        allnan         15.3       63.3       15.3       59.3
        rankdata       45.0        1.5       46.4        2.2
        nanrankdata    55.6       25.2       51.6       38.7
        partsort        5.9        0.9        5.9        1.1
        argpartsort     3.0        0.9        3.2        0.7
        replace        10.1        1.1       10.1        1.1
        push          210.4       19.6      214.7       25.5
        move_sum      258.4      155.6      259.1      409.9
        move_mean     647.1      141.8      635.2      525.3
        move_std     1012.0       75.3     1024.6      364.8
        move_var      939.7      104.0      982.6      363.8
        move_min      191.7       23.1      200.2       51.3
        move_max      179.8       22.9      183.3       92.1
        move_argmin   317.3       40.9      323.6      243.9
        move_argmax   303.2       41.7      313.0      280.8
        move_median   419.4       42.2      415.7      160.4
        move_rank     508.7        2.7      549.4       10.5

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
