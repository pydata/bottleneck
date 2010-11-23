#!/usr/bin/env python

import os
from distutils.core import setup
from distutils.extension import Extension
import numpy


CLASSIFIERS = ["Development Status :: 2 - Pre-Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Cython",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "Fast, NaN-aware descriptive statistics of NumPy arrays."
long_description = """
DSNA uses the magic of Cython to give you fast, NaN-aware descriptive
statistics of NumPy arrays.

The functions in dsna fall into three categories:

===============  ===============================
 General          sum(arr, axis=None)
 Moving window    move_sum(arr, window, axis=0)
 Group by         group_sum(arr, label, axis=0)
===============  ===============================

For example, create a NumPy array::
    
    >>> import numpy as np
    >>> arr = np.array([1, 2, np.nan, 4, 5])

Then find the sum::

    >>> import dsna as ds
    >>> ds.sum(arr)
    12.0

Moving window sum::

    >>> ds.move_sum(arr, window=2)
    array([ nan,   3.,   2.,   4.,   9.])

Group sum::   

    >>> label = ['a', 'a', 'b', 'b', 'a']
    >>> a, lab = ds.group_sum(arr, label)
    >>> a
    array([ 8.,  4.])
    >>> lab
    ['a', 'b']

Fast
====

DNSA is fast::

    >>> import dsna as ds
    >>> import numpy as np
    >>> arr = np.random.rand(100, 100)
    
    >>> timeit np.nansum(arr)
    10000 loops, best of 3: 68.4 us per loop
    >>> timeit ds.sum(arr)
    100000 loops, best of 3: 17.7 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nansum(arr)
    1000 loops, best of 3: 417 us per loop
    >>> timeit ds.sum(arr)
    10000 loops, best of 3: 64.8 us per loop

Faster
======

Under the hood dsna uses a separate Cython function for each combination of
ndim, dtype, and axis. A lot of the overhead in ds.max, for example, is
in checking that your axis is within range, converting non-array data to an
array, and selecting the function to use to calculate the maximum.

You can get rid of the overhead by doing all this before you, say, enter
an inner loop::

    >>> arr = np.random.rand(10,10)
    >>> func, a = ds.func.max_selector(arr, axis=0)
    >>> func
    <built-in function max_2d_float64_axis0> 

Let's see how much faster than runs::    
    
    >> timeit np.nanmax(arr, axis=0)
    10000 loops, best of 3: 25.7 us per loop
    >> timeit ds.max(arr, axis=0)
    100000 loops, best of 3: 5.25 us per loop
    >> timeit func(a)
    100000 loops, best of 3: 2.5 us per loop

Note that ``func`` is faster than the Numpy's non-nan version of max::
    
    >> timeit arr.max(axis=0)
    100000 loops, best of 3: 3.28 us per loop

So adding NaN protection to your inner loops has a negative cost!           

Functions
=========

DSNA is in the prototype stage.

DSNA contains the following functions (an asterisk means not yet complete): 

=========    ==============   ===============
sum*         move_sum*        group_sum*
mean         move_mean*       group_mean*
var          move_var*        group_var*
std          move_std*        group_std*
min          move_min*        group_min*
max          move_max*        group_max*
median*      move_median*     group_median*
zscore*      move_zscore*     group_zscore*
ranking*     move_ranking*    group_ranking*
quantile*    move_quantile*   group_quantile*
count*       move_count*      group_count*
=========    ==============   ===============

Currently only 1d, 2d, and 3d NumPy arrays with dtype int32, int64, and
float64 are supported.
No long description yet.

"""

# Get la version
ver_file = os.path.join('dsna', 'version.py')
fid = file(ver_file, 'r')
VER = fid.read()
fid.close()
VER = VER.split("= ")
VER = VER[1].strip()
VER = VER.strip("\"")
VER = VER.split('.')

NAME                = 'dsna'
MAINTAINER          = "Keith Goodman"
MAINTAINER_EMAIL    = ""
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = ""
DOWNLOAD_URL        = ""
LICENSE             = "Simplified BSD"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Archipel Asset Management AB"
AUTHOR_EMAIL        = "kwgoodman@gmail.com"
PLATFORMS           = "OS Independent"
MAJOR               = VER[0]
MINOR               = VER[1]
MICRO               = VER[2]
ISRELEASED          = False
VERSION             = '%s.%s.%s' % (MAJOR, MINOR, MICRO)
PACKAGES            = ["dsna", "dsna/src", "dsna/src/func", "dsna/tests",
                       "dsna/bench"]
PACKAGE_DATA        = {'dsna': ['LICENSE']}
REQUIRES            = ["numpy"]


setup(name=NAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      classifiers=CLASSIFIERS,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      platforms=PLATFORMS,
      version=VERSION,
      packages=PACKAGES,
      package_data=PACKAGE_DATA,
      requires=REQUIRES,
      ext_modules = [Extension("dsna.func",
                               sources=["dsna/src/func/func.c"],
                               include_dirs=[numpy.get_include()])]
     )
