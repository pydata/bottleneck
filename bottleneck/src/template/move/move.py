
from bottleneck.src.template.template import template

from move_sum import move_sum
from move_nansum import move_nansum
from move_mean import move_mean
from move_nanmean import move_nanmean
from move_std import move_std
from move_nanstd import move_nanstd
from move_min import move_min
from move_max import move_max
from move_nanmin import move_nanmin
from move_nanmax import move_nanmax

funcs = {}
funcs['move_sum'] = move_sum
funcs['move_nansum'] = move_nansum
funcs['move_mean'] = move_mean
funcs['move_nanmean'] = move_nanmean
funcs['move_std'] = move_std
funcs['move_nanstd'] = move_nanstd
funcs['move_min'] = move_min
funcs['move_max'] = move_max
funcs['move_nanmin'] = move_nanmin
funcs['move_nanmax'] = move_nanmax

def movepyx(funcs=funcs, bits=None):
    for func in funcs:
        template(funcs[func], bits)
