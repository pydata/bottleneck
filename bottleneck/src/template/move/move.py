
from bottleneck.src.template.template import template

from move_nanmean import move_nanmean

funcs = {}
funcs['move_nanmean'] = move_nanmean

def movepyx(funcs=funcs):
    for func in funcs:
        template(funcs[func])
