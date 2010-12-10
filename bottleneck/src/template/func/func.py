
from bottleneck.src.template.template import template

from median import median
from nanmean import nanmean
from nanvar import nanvar
from nanstd import nanstd
from nanmin import nanmin
from nanmax import nanmax

funcs = {}
funcs['median'] = median
funcs['nanmean'] = nanmean
funcs['nanvar'] = nanvar
funcs['nanstd'] = nanstd
funcs['nanmin'] = nanmin
funcs['nanmax'] = nanmax

def funcpyx(funcs=funcs):
    for func in funcs:
        template(funcs[func])
