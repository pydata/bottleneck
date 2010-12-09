
from bottleneck.src.template.template import make

from move_nanmean import move_nanmean

funcs = {}
funcs['move_nanmean'] = move_nanmean

def build(funcs=funcs):
    for func in funcs:
        make(funcs[func])
