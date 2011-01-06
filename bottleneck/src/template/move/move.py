
from bottleneck.src.template.template import template

from move_nanmean import move_nanmean
from move_min import move_min

funcs = {}
funcs['move_nanmean'] = move_nanmean
funcs['move_min'] = move_min

def movepyx(funcs=funcs):
    for func in funcs:
        template(funcs[func])
