
=============
Release Notes
=============

These are the major changes made in each release. For details of the changes
see the commit log at https://github.com/pydata/bottleneck

Bottleneck 1.4.1
================

*Release date: 2024-10-12

Enhancements
~~~~~~~~~~~~
- Add python 3.13 build

Bottleneck 1.4.0
================

*Release date: 2024-06-17

Enhancements
~~~~~~~~~~~~
- Building against numpy 2.0, which ships new backwards compatible ABI
- Update CI config

Bottleneck 1.3.8
================

*Release date: 2024-02-04*

Enhancements
~~~~~~~~~~~~
- Python 3.12 wheel available
- Update CI config

Bottleneck 1.3.7
================

*Release date: 2023-01-20*

Enhancements
~~~~~~~~~~~~
- Python 3.11 wheel available


Bottleneck 1.3.6
================

*Release date: 2023-01-19*

Bug Fixes
~~~~~~~~~
- Fix ValueError: cannot convert float NaN to integer with new numpy version

Enhancements
~~~~~~~~~~~~
- Python 3.11 available in CI tests

Cleanup
~~~~~~~~
- Python 3.6 won't be tested anymore because of the deprecation in the associated
  Python Github action 


Bottleneck 1.3.5
================

*Release date: 2022-07-02*

Bug Fixes
~~~~~~~~~
- Fix numpy deprecation of non-tuple indices


Enhancements
~~~~~~~~~~~~
- Switch build to manylinux_2_24_x86_64 using cibuildwheel

Bottleneck 1.3.4
================

*Release date: 2022-02-22*

Bug Fixes
~~~~~~~~~
- Fix Memory leak with big-endian data

Bottleneck 1.3.3
================

*Release date: 2022-02-21*

Bug Fixes
~~~~~~~~~
- Fix Python 3.10 build

Enhancements
~~~~~~~~~~~~
- Provide pre-compiled wheels for most x86_64 architectures

Project Updates
~~~~~~~~~~~~~~~
- The project has two new maintainers: Ruben Di Battista (``@rdbisme`` on Github) and
  Steven Troxler (``@stroxler`` on Github)

Contributors
~~~~~~~~~~~~

.. contributors:: v1.3.2..v1.3.3


Bottleneck 1.3.2
================

*Release date: 2020-02-20*

Bug Fixes
~~~~~~~~~
- Explicitly declare numpy version dependency in ``pyproject.toml`` for Python 3.8, fixing
  certain cases where ``pip install`` would fail. Thanks to ``@goggle``, ``@astrofrog``,
  and ``@0xb0b`` for reporting. (:issue:`277`)

Contributors
~~~~~~~~~~~~

.. contributors:: v1.3.1..v1.3.2

Older Releases
~~~~~~~~~~~~~~

Bottleneck 1.3.1
----------------

*Release date: 2019-11-18*

Bug Fixes
~~~~~~~~~
- Fix memory leak in :func:`bottleneck.nanmedian` with the default argument of ``axis=None``. Thanks to ``@jsmodic`` for reporting! (:issue:`276`, :issue:`278`)
- Add regression test for memory leak case (:issue:`279`)

Contributors
~~~~~~~~~~~~

.. contributors:: v1.3.0..v1.3.1


Bottleneck 1.3.0
----------------

*Release date: 2019-11-12*

Project Updates
~~~~~~~~~~~~~~~
- Bottleneck has a new maintainer, Christopher Whelan (``@qwhelan`` on GitHub).
- Documentation now hosted at https://bottleneck.readthedocs.io
- 1.3.x will be the last release to support Python 2.7
- Bottleneck now supports and is tested against Python 3.7 and 3.8. (:issue:`211`, :issue:`268`)
- The ``LICENSE`` file has been restructured to only include the license for the Bottleneck project to aid license audit tools. There has been no change to the licensing of Bottleneck.

  - Licenses for other projects incorporated by Bottleneck are now reproduced in full in separate files in the ``LICENSES/`` directory (eg, ``LICENSES/NUMPY_LICENSE``)
  - All licenses have been updated. Notably, setuptools is now MIT licensed and no longer under the ambiguous dual PSF/Zope license.
- Bottleneck now uses :pep:`518` for specifying build dependencies, with per Python version specifications (:issue:`247`)


Enhancements
~~~~~~~~~~~~
- Remove ``numpydoc`` package from Bottleneck source distribution
- :func:`bottleneck.slow.reduce.nansum` and :func:`bottleneck.slow.reduce.ss` now longer coerce output to have the same dtype as input
- Test (tox, travis, appveyor) against latest ``numpy`` (in conda)
- Performance benchmarking also available via ``asv``
- ``versioneer`` now used for versioning (:issue:`213`)
- Test suite now uses ``pytest`` as ``nose`` is deprecated (:issue:`222`)
- ``python setup.py build_ext --inplace`` is now incremental (:issue:`224`)
- ``python setup.py clean`` now cleans all artifacts (:issue:`226`)
- Compiler feature support now identified by testing rather than hardcoding (:issue:`227`)
- The ``BN_OPT_3`` macro allows selective use of ``-O3`` at the function level (:issue:`223`)
- Contributors are now automatically cited in the release notes (:issue:`244`)

Performance
~~~~~~~~~~~
- Speed up :func:`bottleneck.reduce.anynan` and :func:`bottleneck.reduce.allnan` by 2x via ``BN_OPT_3`` (:issue:`223`)
- All functions covered by ``asv`` benchmarks
- :func:`bottleneck.nonreduce.replace` speedup of 4x via more explicit typing (:issue:`239`)
- :func:`bottleneck.reduce.median` up to 2x faster for Fortran-ordered arrays (:issue:`248`)


Bug Fixes
~~~~~~~~~

- Documentation fails to build on Python 3 (:issue:`170`)
- :func:`bottleneck.benchmark.bench` crashes on python 3.6.3, numpy 1.13.3 (:issue:`175`)
- :func:`bottleneck.nonreduce_axis.push` raises when :code:`n=None` is explicitly passed (:issue:`178`)
- :func:`bottleneck.reduce.nansum` wrong output when :code:`a = np.ones((2, 2))[..., np.newaxis]`
  same issue of other reduce functions (:issue:`183`)
- Silenced FutureWarning from NumPy in the slow version of move functions (:issue:`194`)
- Installing bottleneck onto a system that does not already have Numpy (:issue:`195`)
- Memory leaked when input was not a NumPy array (:issue:`201`)
- Tautological comparison in :func:`bottleneck.move.move_rank` removed (:issue:`207`, :issue:`212`)

Cleanup
~~~~~~~

- The ``ez_setup.py`` module is no longer packaged (:issue:`211`)
- Building documentation is now self-contained in ``make doc`` (:issue:`214`)
- Codebase now ``flake8`` compliant and run on every commit
- Codebase now uses ``black`` for autoformatting (:issue:`253`)

Contributors
~~~~~~~~~~~~

.. contributors:: v1.2.1..v1.3.0


Bottleneck 1.2.1
----------------

*Release date: 2017-05-15*

This release adds support for NumPy's relaxed strides checking and
fixes a few bugs.

**Bug Fixes**

- Installing bottleneck when two versions of NumPy are present (:issue:`156`)
- Compiling on Ubuntu 14.04 inside a Windows 7 WMware (:issue:`157`)
- Occasional segmentation fault in :func:`bn.nanargmin`, :func:`nanargmax`, :func:`median`,
  and :func:`nanmedian` when all of the following conditions are met:
  axis is None, input array is 2d or greater, and input array is not C
  contiguous. (:issue:`159`)
- Reducing np.array([2**31], dtype=np.int64) overflows on Windows (:issue:`163`)

**Contributors**

.. contributors:: v1.2.0..v1.2.1

Bottleneck 1.2.0
----------------

*Release date: 2016-10-20*

This release is a complete rewrite of Bottleneck.

**Port to C**

- Bottleneck is now written in C
- Cython is no longer a dependency
- Source tarball size reduced by 80%
- Build time reduced by 66%
- Install size reduced by 45%

**Redesign**

- Besides porting to C, much of bottleneck has been redesigned to be
  simpler and faster. For example, bottleneck now uses its own N-dimensional
  array iterators, reducing function call overhead.

**New features**

- The new function bench_detailed runs a detailed performance benchmark on
  a single bottleneck function.
- Bottleneck can be installed on systems that do not yet have NumPy
  installed. Previously that only worked on some systems.

**Beware**

- Functions partsort and argpartsort have been renamed to partition and
  argpartition to match NumPy. Additionally the meaning of the input
  arguments have changed: :func:`bn.partsort(a, n)` is now equivalent to
  :func:`bn.partition(a, kth=n-1)`. Similarly for bn.argpartition.
- The keyword for array input has been changed from `arr` to `a` in all
  functions. It now matches NumPy.

**Thanks**

- Moritz E. Beber: continuous integration with AppVeyor
- Christoph Gohlke: Windows compatibility
- Jennifer Olsen: comments and suggestions
- A special thanks to the Cython developers. The quickest way to appreciate
  their work is to remove Cython from your project. It is not easy.

**Contributors**

.. contributors:: v1.1.0..v1.2.0

Bottleneck 1.1.0
----------------

*Release date: 2016-06-22*

This release makes Bottleneck more robust, releases GIL, adds new functions.

**More Robust**

- :func:`bn.move_median` can now handle NaNs and `min_count` parameter
- :func:`bn.move_std` is slower but numerically more stable
- Bottleneck no longer crashes on byte-swapped input arrays

**Faster**

- All Bottleneck functions release the GIL
- median is faster if the input array contains NaN
- move_median is faster for input arrays that contain lots of NaNs
- No speed penalty for median, nanmedian, nanargmin, nanargmax for Fortran
  ordered input arrays when axis is None
- Function call overhead cut in half for reduction along all axes (axis=None)
  if the input array satisfies at least one of the following properties: 1d,
  C contiguous, F contiguous
- Reduction along all axes (axis=None) is more than twice as fast for long,
  narrow input arrays such as a (1000000, 2) C contiguous array and a
  (2, 1000000) F contiguous array

**New Functions**

- move_var
- move_argmin
- move_argmax
- move_rank
- push

**Beware**

- :func:`bn.median` now returns NaN for a slice that contains one or more NaNs
- Instead of using the distutils default, the '-O2' C compiler flag is forced
- :func:`bn.move_std` output changed when mean is large compared to standard deviation
- Fixed: Non-accelerated moving window functions used min_count incorrectly
- :func:`bn.move_median` is a bit slower for float input arrays that do not contain NaN

**Thanks**

Alphabeticaly by last name

- Alessandro Amici worked on setup.py
- Pietro Battiston modernized bottleneck installation
- Moritz E. Beber set up continuous integration with Travis CI
- Jaime Frio improved the numerical stability of move_std
- Christoph Gohlke revived Windows compatibility
- Jennifer Olsen added NaN support to move_median

**Contributors**

.. contributors:: v1.0.0..v1.1.0

Bottleneck 1.0.0
----------------

*Release date: 2015-02-06*

This release is a complete rewrite of Bottleneck.

**Faster**

- "python setup.py build" is 18.7 times faster
- Function-call overhead cut in half---a big speed up for small input arrays
- Arbitrary ndim input arrays accelerated; previously only 1d, 2d, and 3d
- bn.nanrankdata is twice as fast for float input arrays
- bn.move_max, bn.move_min are faster for int input arrays
- No speed penalty for reducing along all axes when input is Fortran ordered

**Smaller**

- Compiled binaries 14.1 times smaller
- Source tarball 4.7 times smaller
- 9.8 times less C code
- 4.3 times less Cython code
- 3.7 times less Python code

**Beware**

- Requires numpy 1.9.1
- Single API, e.g.: bn.nansum instead of bn.nansum and nansum_2d_float64_axis0
- On 64-bit systems bn.nansum(int32) returns int32 instead of int64
- bn.nansum now returns 0 for all NaN slices (as does numpy 1.9.1)
- Reducing over all axes returns, e.g., 6.0; previously np.float64(6.0)
- bn.ss() now has default axis=None instead of axis=0
- bn.nn() is no longer in bottleneck

**min_count**

- Previous releases had moving window function pairs: move_sum, move_nansum
- This release only has half of the pairs: move_sum
- Instead a new input parameter, min_count, has been added
- min_count=None same as old move_sum; min_count=1 same as old move_nansum
- If # non-NaN values in window < min_count, then NaN assigned to the window
- Exception: move_median does not take min_count as input

**Bug Fixes**

- Can now install bottleneck with pip even if numpy is not already installed
- bn.move_max, bn.move_min now return float32 for float32 input

**Contributors**

.. contributors:: v0.8.0..v1.0.0

Bottleneck 0.8.0
----------------

*Release date: 2014-01-21*

This version of Bottleneck requires NumPy 1.8.

**Breaks from 0.7.0**

- This version of Bottleneck requires NumPy 1.8
- nanargmin and nanargmax behave like the corresponding functions in NumPy 1.8

**Bug fixes**

- nanargmax/nanargmin wrong for redundant max/min values in 1d int arrays

**Contributors**

.. contributors:: v0.7.0..v0.8.0

Bottleneck 0.7.0
----------------

*Release date: 2013-09-10*

**Enhancements**

- bn.rankdata() is twice as fast (with input a = np.random.rand(1000000))
- C files now included in github repo; cython not needed to try latest
- C files are now generated with Cython 0.19.1 instead of 0.16
- Test bottleneck across multiple python/numpy versions using tox
- Source tarball size cut in half

**Bug fixes**

- move_std, move_nanstd return inappropriate NaNs (sqrt of negative #) (:issue:`50`)
- `make test` fails on some computers (:issue:`52`)
- scipy optional yet some unit tests depend on scipy (:issue:`57`)
- now works on Mac OS X 10.8 using clang compiler (:issue:`49`, :issue:`55`)
- nanstd([1.0], ddof=1) and nanvar([1.0], ddof=1) crash (:issue:`60`)

**Contributors**

.. contributors:: v0.6.0..v0.7.0

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

- Confusing error message in partsort and argpartsort (:issue:`31`)
- Update path in MANIFEST.in (:issue:`32`)
- Wrong output for very large (2**31) input arrays (:issue:`35`)

**Contributors**

.. contributors:: v0.5.0..v0.6.0

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

- Support python 2.5 by importing `with` statement (:issue:`14`)
- nanmedian wrong for particular ordering of NaN and non-NaN elements (:issue:`22`)
- argpartsort, nanargmin, nanargmax returned wrong dtype on 64-bit Windows (:issue:`26`)
- rankdata and nanrankdata crashed on 64-bit Windows (:issue:`29`)

Bottleneck 0.4.3
----------------

*Release date: 2011-03-17*

This is a bug fix release.

**Bug fixes**

- median and nanmedian modified (partial sort) input array (:issue:`11`)
- nanmedian wrong when odd number of elements with all but last a NaN (:issue:`12`)

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

- Some functions gave wrong output dtype for some input dtypes on 32 bit OS (:issue:`6`)
- Some functions choked on size zero input arrays (:issue:`7`)
- Segmentation fault with Cython 0.14.1 (but not 0.13) (:issue:`8`)

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
