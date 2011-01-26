
from bottleneck.src.template.template import template

from group_nanmean import group_nanmean

funcs = {}
funcs['group_nanmean'] = group_nanmean

def grouppyx(funcs=funcs, bits=None):
    for func in funcs:
        template(funcs[func], bits)
