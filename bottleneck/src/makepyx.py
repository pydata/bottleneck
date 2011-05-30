"Generate all pyx files from templates"

from bottleneck.src.template.func.func import funcpyx
from bottleneck.src.template.move.move import movepyx

def makepyx():
    funcpyx(bits=32)
    movepyx(bits=32)
    funcpyx(bits=64)
    movepyx(bits=64)
