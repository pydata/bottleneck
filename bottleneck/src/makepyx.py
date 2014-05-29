"Generate all pyx files from templates"

from bottleneck.src.template.func.func import funcpyx
from bottleneck.src.template.move.move import movepyx


def makepyx():
    funcpyx()
    movepyx()
