==================
Function reference
==================

Bottleneck provides the following functions:

=================================  ==============================================================================================
reduce                             :meth:`nansum <bottleneck.nansum>`, :meth:`nanmean <bottleneck.nanmean>`,
                                   :meth:`nanstd <bottleneck.nanstd>`, :meth:`nanvar <bottleneck.nanvar>`,
                                   :meth:`nanmin <bottleneck.nanmin>`, :meth:`nanmax <bottleneck.nanmax>`,
                                   :meth:`median <bottleneck.median>`, :meth:`nanmedian <bottleneck.nanmedian>`,
                                   :meth:`ss <bottleneck.ss>`, :meth:`nanargmin <bottleneck.nanargmin>`,
                                   :meth:`nanargmax <bottleneck.nanargmax>`, :meth:`anynan <bottleneck.anynan>`,
                                   :meth:`allnan <bottleneck.allnan>`

non-reduce                         :meth:`replace <bottleneck.replace>`

non-reduce with axis               :meth:`rankdata <bottleneck.rankdata>`, :meth:`nanrankdata <bottleneck.nanrankdata>`,
                                   :meth:`partsort <bottleneck.partsort>`, :meth:`argpartsort <bottleneck.argpartsort>`,

moving window                      :meth:`move_sum <bottleneck.move_sum>`, :meth:`move_mean <bottleneck.move_mean>`,
                                   :meth:`move_std <bottleneck.move_std>`, :meth:`move_min <bottleneck.move_min>`,
                                   :meth:`move_max <bottleneck.move_max>`

moving window without `min_count`  :meth:`move_median <bottleneck.move_median>`

=================================  ==============================================================================================


Reduce
------

Functions the reduce the input array along the specified axis.

------------

.. autofunction:: bottleneck.nansum

------------

.. autofunction:: bottleneck.nanmean

------------

.. autofunction:: bottleneck.nanstd

------------

.. autofunction:: bottleneck.nanvar

------------

.. autofunction:: bottleneck.nanmin

------------

.. autofunction:: bottleneck.nanmax

------------

.. autofunction:: bottleneck.median

------------

.. autofunction:: bottleneck.nanmedian

------------

.. autofunction:: bottleneck.ss

------------

.. autofunction:: bottleneck.nanargmin

------------

.. autofunction:: bottleneck.nanargmax


------------

.. autofunction:: bottleneck.anynan

------------

.. autofunction:: bottleneck.allnan


Non-reduce
----------

Functions that do not reduce the input array.

------------

.. autofunction:: bottleneck.replace


Non-reduce with axis
--------------------

Functions that do not reduce the input array but operate along a specified
axis.

------------

.. autofunction:: bottleneck.rankdata

------------

.. autofunction:: bottleneck.nanrankdata

------------

.. autofunction:: bottleneck.partsort

------------

.. autofunction:: bottleneck.argpartsort


Moving window functions
-----------------------

Moving window functions (with a 1d window) that take `min_count` as an optional
input.

------------

.. autofunction:: bottleneck.move_sum

------------

.. autofunction:: bottleneck.move_mean

------------

.. autofunction:: bottleneck.move_std

------------

.. autofunction:: bottleneck.move_min

------------

.. autofunction:: bottleneck.move_max


Moving window functions without `min_count`
-------------------------------------------

Moving window functions (with a 1d window) that do NOT take `min_count` as an
optional input. In other works, NaNs in will results in NaNs in the output.

------------

.. autofunction:: bottleneck.move_median

