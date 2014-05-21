"Generate all pyx files from templates"

from bottleneck.src.template.func.func import funcpyx
from bottleneck.src.template.move.move import movepyx


def makepyx(ndim_max=3):
    funcpyx(ndim_max=ndim_max)
    movepyx(ndim_max=ndim_max)
