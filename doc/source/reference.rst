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
                                   :meth:`partition <bottleneck.partition>`, :meth:`argpartition <bottleneck.argpartition>`,
                                   :meth:`push <bottleneck.push>`

moving window                      :meth:`move_sum <bottleneck.move_sum>`, :meth:`move_mean <bottleneck.move_mean>`,
                                   :meth:`move_std <bottleneck.move_std>`, :meth:`move_var <bottleneck.move_var>`,
                                   :meth:`move_min <bottleneck.move_min>`, :meth:`move_max <bottleneck.move_max>`,
                                   :meth:`move_argmin <bottleneck.move_argmin>`, :meth:`move_argmax <bottleneck.move_argmax>`,
                                   :meth:`move_median <bottleneck.move_median>`, :meth:`move_rank <bottleneck.move_rank>`

=================================  ==============================================================================================


Reduce
------

Functions that reduce the input array along the specified axis.

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

Functions that do not reduce the input array and do not take `axis` as input.

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

.. autofunction:: bottleneck.partition

------------

.. autofunction:: bottleneck.argpartition

------------

.. autofunction:: bottleneck.push


Moving window functions
-----------------------

Functions that operate along a (1d) moving window.

------------

.. autofunction:: bottleneck.move_sum

------------

.. autofunction:: bottleneck.move_mean

------------

.. autofunction:: bottleneck.move_std

------------

.. autofunction:: bottleneck.move_var

------------

.. autofunction:: bottleneck.move_min

------------

.. autofunction:: bottleneck.move_max

------------

.. autofunction:: bottleneck.move_argmin

------------

.. autofunction:: bottleneck.move_argmax

------------

.. autofunction:: bottleneck.move_median

------------

.. autofunction:: bottleneck.move_rank

