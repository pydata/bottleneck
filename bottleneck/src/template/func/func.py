
from bottleneck.src.template.template import template

from median import median
from nanmedian import nanmedian
from nansum import nansum
from nanmean import nanmean
from nanvar import nanvar
from nanstd import nanstd
from nanmin import nanmin
from nanmax import nanmax
from nanargmin import nanargmin
from nanargmax import nanargmax
from rankdata import rankdata
from nanrankdata import nanrankdata

funcs = {}
funcs['median'] = median
funcs['nanmedian'] = nanmedian
funcs['nansum'] = nansum
funcs['nanmean'] = nanmean
funcs['nanvar'] = nanvar
funcs['nanstd'] = nanstd
funcs['nanmin'] = nanmin
funcs['nanmax'] = nanmax
funcs['nanargmin'] = nanargmin
funcs['nanargmax'] = nanargmax
funcs['rankdata'] = rankdata
funcs['nanrankdata'] = nanrankdata

def funcpyx(funcs=funcs, bits=None):
    for func in funcs:
        template(funcs[func], bits)
