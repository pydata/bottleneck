===========
Development
===========

There are many ways to help improve Bottleneck:

- Use it and report bugs, typos, suggestions
- Write a prototype for a new function in the sandbox 
- Work on bits from the roadmap below

Step #1: Get the code at https://github.com/kwgoodman/bottleneck 

Roadmap
-------

**0.1**

- Initial, preview release

**0.2**

- Template the code to make maintance and the expansion to more dtypes
  easier.
- Fall back to slower non-Cython functions for unaccelerated ndim/dtype

**0.3**

- Add more functions
- What's a good way to find a moving window maximum? Is
  `this <http://home.tiac.net/~cri/2001/slidingmin.html>`_ a good way to go?
- What other functions would fit in well with the rest of Bottleneck?  

Sandbox
-------

A good place to try out an idea for a new Bootleneck function is in the
sandbox directory.

You do not need to make a function that handles every possible dtype and
an arbitrary number of dimensions. A prototype function need only handle a 1d
float64 input array. The idea is to concentrate on the algorithm, not dtypes
or ndim.

The sandbox comes with an example of a nanmean cython function that takes a
1d float64 NumPy array as input.

To convert nanmean.pyx to a C file and compile it::

    $ cd bottleneck/sandbox
    bottleneck/sandbox$ python setup.py build_ext --inplace
    running build_ext
    cythoning nanmean.pyx to nanmean.c
    <snip>

To use the the function::

    >>> from nanmean import nanmean
    >>> import numpy as np
    >>> a = np.array([1.0, 2.0, 4.0])
    >>> nanmean(a)
    1.1666666666666667
