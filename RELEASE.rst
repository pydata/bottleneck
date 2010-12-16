
=============
Release Notes
=============

These are the major changes made in each release. For details of the changes
see the commit log at http://github.com/kwgoodman/bottleneck

Bottleneck 0.2.0
================

*Release date: Not yet released, in development*

**Enhancements**

- Added support for float32
- Fallback to slower, non-Cython functions for unaccelerated ndim/dtype  
- All functions faster (less overhead) when output is not a scalar
- Faster nanmean() for 2d, 3d arrays with NaNs when axis is not None
- Scipy is no longer a dependency
- Added support for older versions of NumPy (1.4.1)
- All functions are now templated for dtype and axis  
- Added a sandbox for prototyping of new Bottleneck functions
- Rewrote benchmarking code  

**Breaks from 0.2**

- To run benchmark use bn.bench() instead of bn.benchit()

Older versions
==============

Release notes from past releases.

Bottleneck 0.1.0
----------------

*Release date: 2010-12-01*

Preview release of Bottleneck.
