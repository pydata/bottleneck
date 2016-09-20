import bottleneck as bn


def get_functions(module_name, as_string=False):
    "Returns a list of functions, optionally as string function names"
    if module_name == 'all':
        funcs = []
        funcs_in_dict = func_dict()
        for key in funcs_in_dict:
            for func in funcs_in_dict[key]:
                funcs.append(func)
    else:
        funcs = func_dict()[module_name]
    if as_string:
        funcs = [f.__name__ for f in funcs]
    return funcs


def func_dict():
    d = {}
    d['reduce'] = [bn.nansum,
                   bn.nansum2,
                   bn.nanmean,
                   bn.nanmean2,
                   bn.nanstd,
                   bn.nanstd2,
                   bn.nanvar,
                   bn.nanvar2,
                   bn.nanmin,
                   bn.nanmin2,
                   bn.nanmax,
                   bn.nanmax2,
                   bn.median,
                   bn.median2,
                   bn.nanmedian,
                   bn.nanmedian2,
                   bn.ss,
                   bn.ss2,
                   bn.nanargmin,
                   bn.nanargmin2,
                   bn.nanargmax,
                   bn.nanargmax2,
                   bn.anynan,
                   bn.anynan2,
                   bn.allnan,
                   bn.allnan2,
                   ]
    d['move'] = [bn.move_sum,
                 bn.move_sum2,
                 bn.move_mean,
                 bn.move_mean2,
                 bn.move_std,
                 bn.move_std2,
                 bn.move_var,
                 bn.move_var2,
                 bn.move_min,
                 bn.move_min2,
                 bn.move_max,
                 bn.move_max2,
                 bn.move_argmin,
                 bn.move_argmin2,
                 bn.move_argmax,
                 bn.move_argmax2,
                 bn.move_median,
                 bn.move_median2,
                 bn.move_rank,
                 bn.move_rank2,
                 ]
    d['nonreduce'] = [bn.replace, bn.replace2]
    d['nonreduce_axis'] = [bn.partsort,
                           bn.partsort2,
                           bn.argpartsort,
                           bn.argpartsort2,
                           bn.rankdata,
                           bn.rankdata2,
                           bn.nanrankdata,
                           bn.nanrankdata2,
                           bn.push,
                           bn.push2,
                           ]
    return d
