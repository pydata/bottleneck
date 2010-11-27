"""
Alternative (non-Cython) group function.

These function are slow but useful as a reference for unit testing.
"""

import numpy as np

def group_func(func, arr, label, axis=0):
    """
    Slow, generic group function. Only use for unit testing.
    
    Parameters
    ----------
    func : function
        Reducing function such as np.nansum that takes an array and axis as
        input.
    arr : array_like
        Values to be grouped.
    label : array_like
        List of group membership of each element along the axis.
    axis : int, optional
        `axis` along which `func` is calculated.
        
    Returns
    -------
    garr : ndarray
        The group values of `func` evaluated for each group along `axis`. 
        If `arr` has shape (n,m), `axis` is 0, and there are q unique labels,
        then the output array will have shape (q,m). The dtype of `garr` is
        float64.
    ulabel : list
        A list of the unique values in `label` in the order of the results
        in `garr`.

    """
    arr = np.array(arr, copy=False)
    label = np.asarray(label)    
    ulabels = np.unique(label)
    shape = list(arr.shape)
    shape[axis] = len(ulabels)
    garr = np.zeros(shape, dtype=np.float64)  
    idx1 = [slice(None)] * arr.ndim
    idx2 = [slice(None)] * arr.ndim
    for i, ulabel in enumerate(ulabels):
        idx1[axis] = i
        idx2[axis] = label == ulabel
        garr[idx1] = func(arr[idx2], axis=axis)       
    return garr, ulabels.tolist()
