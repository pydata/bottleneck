from ..move import move_max, move_min
from ..move import move_quantile as move_quantile_c

all = ["move_quantile"]

def move_quantile(*args, **kwargs):
    if ('q' not in kwargs) or ((kwargs['q'] > 0.) and (kwargs['q'] < 1.)):
        return move_quantile_c(*args, **kwargs)
    elif (kwargs['q'] == 1.):
        del kwargs['q']
        return move_max(*args, **kwargs)
    elif (kwargs['q'] == 0.):
        del kwargs['q']
        return move_min(*args, **kwargs)
    
move_quantile.__doc__ = move_quantile_c.__doc__