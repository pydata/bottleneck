
from bottleneck.src.template.template import make

from nanmin import nanmin

funcs = {}
funcs['nanmin'] = nanmin

def build(funcs=funcs, maxdim=3):
    for func in funcs:
        make(funcs[func])
