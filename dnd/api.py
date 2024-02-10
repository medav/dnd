import torch
import torch.fx
from functorch.compile import aot_module

import time

from .common import *
from . import env
from . import nvtools
from . import tracer
from . import bench

from .tracer import trace_region

__all__ = ['trace_region', 'instrument', 'profile']

def instrument(fn_or_module : 'callable', use_fx : bool = False):
    if env.mode != env.Mode.TRACE: return fn_or_module

    if use_fx:
        traced = torch.fx.symbolic_trace(fn_or_module)
        return aot_module(traced, tracer.trace_compile_fn)

    else:
        return torch.compile(fn_or_module, backend=tracer.aot_compile_fn)

def _benchmark(roi, *args, no_compile : bool = False, **kwargs):
    print(f'Warmup with {env.bench_nw} Iters')
    for i in range(env.bench_nw): roi(*args, **kwargs)

    print(f'Running {env.bench_ni} Iters')
    tt0 = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(env.bench_ni): roi(*args, **kwargs)
    torch.cuda.synchronize()
    tt1 = time.perf_counter()

    print(f'Elapsed Time: {tt1 - tt0:.3f} s')

    if bench.benchfile is not None:
        with open(bench.benchfile, 'a') as f:
            print(f'total_time: {tt1 - tt0}', file=f)
            print(f'avg_time: {(tt1 - tt0) / env.bench_ni}', file=f)

def _trace(roi, *args, no_compile : bool = False, **kwargs):
    tracer.op_id = 0

    #
    # Some applications don't yet work with torch.compile. For these
    # applications, we assume the user has worked out an alternative approach
    # to instrumenting their application (E.g. use torch.fx and aot_module).
    # In this case, we simply run the roi() as-is.
    #

    if no_compile:
        roi(*args, **kwargs)
    else:
        instrumented = instrument(roi)
        instrumented(*args, **kwargs)

def profile(roi : 'callable', *args, **kwargs):
    env.print_config()
    if env.mode == env.Mode.BENCH:
        _benchmark(roi, *args, **kwargs)
    elif env.mode == env.Mode.TRACE:
        _trace(roi, *args, **kwargs)
    else:
        raise ValueError(f'Unknown mode: {env.mode}')
