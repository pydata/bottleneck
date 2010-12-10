
from bottleneck.src.template.template import template

from nanmin import nanmin
from median import median

funcs = {}
funcs['nanmin'] = nanmin
funcs['median'] = median

def funcpyx(funcs=funcs):
    for func in funcs:
        template(funcs[func])
