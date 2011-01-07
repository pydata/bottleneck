
from bottleneck.src.template.template import template

from move_nanmean import move_nanmean
from move_min import move_min
from move_max import move_max
from move_nanmin import move_nanmin
from move_nanmax import move_nanmax

funcs = {}
funcs['move_nanmean'] = move_nanmean
funcs['move_min'] = move_min
funcs['move_max'] = move_max
funcs['move_nanmin'] = move_nanmin
funcs['move_nanmax'] = move_nanmax

def movepyx(funcs=funcs):
    for func in funcs:
        template(funcs[func])
