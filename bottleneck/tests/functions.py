import bottleneck as bn


def reduce_functions():
    return func_dict()['reduce']


def move_functions():
    return func_dict()['move']


def nonreduce_functions():
    return func_dict()['nonreduce']


def nonreduce_axis_functions():
    return func_dict()['nonreduce_axis']


def all_functions():
    a = []
    funcs = func_dict()
    for key in funcs:
        for func in funcs[key]:
            a.append(func)
    return a


def func_dict():
    d = {}
    d['reduce'] = [bn.nansum,
                   bn.nanmean,
                   bn.nanstd,
                   bn.nanvar,
                   bn.nanmin,
                   bn.nanmax,
                   bn.median,
                   bn.nanmedian,
                   bn.ss,
                   bn.nanargmin,
                   bn.nanargmax,
                   bn.anynan,
                   bn.allnan,
                   ]
    d['move'] = [bn.move_sum,
                 bn.move_mean,
                 bn.move_std,
                 bn.move_var,
                 bn.move_min,
                 bn.move_max,
                 bn.move_argmin,
                 bn.move_argmax,
                 bn.move_median,
                 bn.move_rank,
                 ]
    d['nonreduce'] = [bn.replace]
    d['nonreduce_axis'] = [bn.partsort,
                           bn.argpartsort,
                           bn.rankdata,
                           bn.nanrankdata,
                           bn.push,
                           ]
    return d
