===========
Development
===========

There are many ways to help improve Bottleneck:

- Use it and report bugs, typos, suggestions
- Write a prototype for a new function (that takes a 1d array as input)
- Work on bits from the roadmap below
- Participate in a sprint!

Step #1: Get the code at https://github.com/kwgoodman/bottleneck 

Roadmap
"""""""

**0.1**

- Initial, preview release

**0.2**

- Add a Cython version of NumPy's apply_along_axis function. Input is a Cython
  function that reduces a 1d array to a scalar (such as sum, std, or max).
  Data will probably be passed to the 1d reducing function as pointers
  to a buffer using strides. See
  `here <http://projects.scipy.org/numpy/attachment/ticket/1213/_selectmodule.pyx>`_ for a (non-reducing) example.
- Template the code to make maintance and the expansion to more dtypes
  easier. Two possible approaches, which is better:
  `one <http://mail.scipy.org/pipermail/scipy-user/2010-November/027645.html>`_
  or `two <http://projects.scipy.org/numpy/attachment/ticket/1213/generate_qselect.py>`_?

**0.3**

- Add more functions
- What's a good way to find a moving window maximum? Is
  `this <http://home.tiac.net/~cri/2001/slidingmin.html>`_ a good way to go?
- What other functions would fit in well with the rest of Bottleneck?  
