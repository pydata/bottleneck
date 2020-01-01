==========
Bottleneck
==========

Bottleneck is a collection of fast, NaN-aware NumPy array functions written in C.

As one example, to check if a ``np.array`` has any NaNs using numpy, one must call ``np.any(np.isnan(array))``. The :meth:`bottleneck.anynan` function interleaves the :meth:`np.isnan` check with :meth:`np.any` pre-exit, enabling up to an ``O(N)`` speedup relative to numpy.

Bottleneck strives to be a drop-in accelerator for NumPy functions. When using the following libraries, Bottleneck support is automatically enabled and utilized:
 * `pandas <https://pandas.pydata.org/pandas-docs/stable/install.html#recommended-dependencies>`_
 * `xarray <http://xarray.pydata.org/en/stable/installing.html#instructions>`_
 * `astropy <https://docs.astropy.org/en/stable/install.html>`_

Details on the performance benefits can be found in :ref:`benchmarking`

Table of Contents
=================
.. toctree::
   :maxdepth: 3
   
   intro
   reference
   release
   license

Indices and tables
==================
.. toctree::
   :maxdepth: 1

   bottleneck

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
