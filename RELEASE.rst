
=============
Release Notes
=============

These are the major changes made in each release. For details of the changes
see the commit log at http://github.com/kwgoodman/bottleneck

Bottleneck 0.7.0
================

*Release date: 2013-09-10*

**Enhancements**

- bn.rankdata() is twice as fast (with input a = np.random.rand(1000000))
- C files now included in github repo; cython not needed to try latest
- C files are now generated with Cython 0.19.1 instead of 0.16
- Test bottleneck across multiple python/numpy versions using tox
- Source tarball size cut in half

**Bug fixes**

- #50 move_std, move_nanstd return inappropriate NaNs (sqrt of negative #)
- #52 `make test` fails on some computers
- #57 scipy optional yet some unit tests depend on scipy
- #49, #55 now works on Mac OS X 10.8 using clang compiler
- #60 nanstd([1.0], ddof=1) and nanvar([1.0], ddof=1) crash

Older versions
==============

Release notes from past releases.

Bottleneck 0.6.0
----------------

*Release date: 2012-06-04*

Thanks to Dougal Sutherland, Bottleneck now runs on Python 3.2.

**New functions**

- replace(arr, old, new), e.g, replace(arr, np.nan, 0)
- nn(arr, arr0, axis) nearest neighbor and its index of 1d arr0 in 2d arr
- anynan(arr, axis) faster alternative to np.isnan(arr).any(axis) 
- allnan(arr, axis) faster alternative to np.isnan(arr).all(axis) 

**Enhancements**

- Python 3.2 support (may work on earlier versions of Python 3)
- C files are now generated with Cython 0.16 instead of 0.14.1
- Upgrade numpydoc from 0.3.1 to 0.4 to support Sphinx 1.0.1

**Breaks from 0.5.0**

- Support for Python 2.5 dropped
- Default axis for benchmark suite is now axis=1 (was 0)

**Bug fixes**

- #31 Confusing error message in partsort and argpartsort
- #32 Update path in MANIFEST.in
- #35 Wrong output for very large (2**31) input arrays

Bottleneck 0.5.0
----------------

*Release date: 2011-06-13*

The fifth release of bottleneck adds four new functions, comes in a single
source distribution instead of separate 32 and 64 bit versions, and contains
bug fixes.

J. David Lee wrote the C-code implementation of the double heap moving
window median.

**New functions**

- move_median(), moving window median
- partsort(), partial sort
- argpartsort()
- ss(), sum of squares, faster version of scipy.stats.ss

**Changes**

- Single source distribution instead of separate 32 and 64 bit versions
- nanmax and nanmin now follow Numpy 1.6 (not 1.5.1) when input is all NaN

**Bug fixes**

- #14 Support python 2.5 by importing `with` statement
- #22 nanmedian wrong for particular ordering of NaN and non-NaN elements
- #26 argpartsort, nanargmin, nanargmax returned wrong dtype on 64-bit Windows
- #29 rankdata and nanrankdata crashed on 64-bit Windows

Bottleneck 0.4.3
----------------

*Release date: 2011-03-17*

This is a bug fix release.

**Bug fixes**

- #11 median and nanmedian modified (partial sort) input array
- #12 nanmedian wrong when odd number of elements with all but last a NaN

**Enhancement**

- Lazy import of SciPy (rarely used) speeds Bottleneck import 3x

Bottleneck 0.4.2
----------------

*Release date: 2011-03-08*

This is a bug fix release.

Same bug fixed in Bottleneck 0.4.1 for nanstd() was fixed for nanvar() in
this release. Thanks again to Christoph Gohlke for finding the bug.

Bottleneck 0.4.1
----------------

*Release date: 2011-03-08*

This is a bug fix release.

The low-level functions nanstd_3d_int32_axis1 and nanstd_3d_int64_axis1,
called by bottleneck.nanstd(), wrote beyond the memory owned by the output
array if arr.shape[1] == 0 and arr.shape[0] > arr.shape[2], where arr is
the input array.

Thanks to Christoph Gohlke for finding an example to demonstrate the bug.

Bottleneck 0.4.0
----------------

*Release date: 2011-03-08*

The fourth release of Bottleneck contains new functions and bug fixes.
Separate source code distributions are now made for 32 bit and 64 bit
operating systems.

**New functions**

- rankdata()
- nanrankdata()

**Enhancements**

- Optionally specify the shapes of the arrays used in benchmark
- Can specify which input arrays to fill with one-third NaNs in benchmark

**Breaks from 0.3.0**

- Removed group_nanmean() function
- Bump dependency from NumPy 1.4.1 to NumPy 1.5.1
- C files are now generated with Cython 0.14.1 instead of 0.13

**Bug fixes**

- #6 Some functions gave wrong output dtype for some input dtypes on 32 bit OS
- #7 Some functions choked on size zero input arrays
- #8 Segmentation fault with Cython 0.14.1 (but not 0.13)

Bottleneck 0.3.0
----------------

*Release date: 2010-01-19*

The third release of Bottleneck is twice as fast for small input arrays and
contains 10 new functions.

**Faster**

- All functions are faster (less overhead in selector functions) 

**New functions**

- nansum()
- move_sum()
- move_nansum()  
- move_mean()  
- move_std()
- move_nanstd()  
- move_min()
- move_nanmin()
- move_max()
- move_nanmax()

**Enhancements**

- You can now specify the dtype and axis to use in the benchmark timings
- Improved documentation and more unit tests  

**Breaks from 0.2.0**

- Moving window functions now default to axis=-1 instead of axis=0
- Low-level moving window selector functions no longer take window as input 

**Bug fix**

- int input array resulted in call to slow, non-cython version of move_nanmean

Bottleneck 0.2.0
----------------

*Release date: 2010-12-27*

The second release of Bottleneck is faster, contains more functions, and
supports more dtypes.

**Faster**

- All functions faster (less overhead) when output is not a scalar
- Faster nanmean() for 2d, 3d arrays containing NaNs when axis is not None

**New functions**

- nanargmin()
- nanargmax()
- nanmedian()

**Enhancements**

- Added support for float32
- Fallback to slower, non-Cython functions for unaccelerated ndim/dtype  
- Scipy is no longer a dependency
- Added support for older versions of NumPy (1.4.1)
- All functions are now templated for dtype and axis  
- Added a sandbox for prototyping of new Bottleneck functions
- Rewrote benchmarking code

Bottleneck 0.1.0
----------------

*Release date: 2010-12-10*

Initial release. The three categories of Bottleneck functions:

- Faster replacement for NumPy and SciPy functions
- Moving window functions
- Group functions that bin calculations by like-labeled elements
