
import os
import subprocess
import tempfile
import yaml
import sys
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.fx
from functorch.compile import aot_module
from functorch.compile import make_boxed_func

from .common import *
from . import prof

tracefile = get_optional_env('LOOM_TRACEFILE')

ignore_aten_ops = {
    'aten.t.default',
    'aten.view.default',
    '<built-in function getitem>',
    'aten.randn.default',
    'aten.slice.Tensor',
    'aten.new_zeros.default'
}

kern_id = 0
op_id = 0

class KernelTracer(torch.fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)

    def call_function(self, target, args, kwargs):
        global tracefile
        global op_id
        op_uid = f'{op_id}:{target}'
        op_id += 1
        with torch.cuda.profiler.profile():
            with torch.cuda.nvtx.range(op_uid):
                outs = super().call_function(target, args, kwargs)


        if tracefile is not None:
            with open(tracefile, 'a') as f:
                print(f'  - uid: {op_uid}', file=f)

        return outs

def trace_compile_fn(gm : torch.fx.GraphModule, args):
    def wrapper(*args, **kwargs):
        return KernelTracer(gm).run(*args, **kwargs)

    return make_boxed_func(wrapper)

def aot_compile_fn(gm : torch.fx.GraphModule, args):
    return aot_module(gm, trace_compile_fn)

region_active = False

@contextmanager
def trace_region(name):
    global tracefile
    global region_active

    assert not region_active, 'Nested trace_region\'s not supported'

    region_active = True
    if tracefile is not None:
        with open(tracefile, 'a') as f:
            print(f'{name}:', file=f)

    yield
    region_active = False

def trace(fn_or_module : 'callable', *args, **kwargs):
    global op_id
    op_id = 0

    compiled = torch.compile(fn_or_module, backend=aot_compile_fn)
    compiled(*args, **kwargs)

def instrument(fn_or_module : 'callable', use_fx : bool = False):
    global op_id
    op_id = 0

    if use_fx:
        traced = torch.fx.symbolic_trace(fn_or_module)
        return aot_module(traced, trace_compile_fn)

    else:
        compiled = torch.compile(fn_or_module, backend=aot_compile_fn)
        return compiled(*args, **kwargs)

def run_kernel_trace(prog_args, kernel_trace_file):
    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['LOOM_TRACEFILE'] = kernel_trace_file
    subprocess.check_call(prog_args, env=env)
    return prof.run_prof(prog_args, ncu_use_nvtx=True)

def lookup_kerns(kerns : 'list[prof.Kernel]', uid : str):
    return list(filter(lambda k: k.nvtx_range == uid, kerns))

def process_region(name, yd, kerns : 'list[prof.Kernel]', outfile):
    if yd is None: yd = []

    ops = []
    for opd in yd:
        ops.append(Operator(
            uid=opd['uid'],
            kerns=lookup_kerns(kerns, opd['uid'])
        ))

    print(f'{name}:', file=outfile)
    op : Operator
    for op in ops:
        opname = op.uid.split(':')[1]
        if opname in ignore_aten_ops: continue
        op.print_yaml(file=outfile)


def run_tracing(prog_args, kerns_file):
    with tempfile.NamedTemporaryFile(suffix='.yaml') as temp_kern_trace_file:
        kerns = run_kernel_trace(prog_args, temp_kern_trace_file.name)
        yd = yaml.safe_load(open(temp_kern_trace_file.name, 'r'))

        with open(kerns_file, 'w') as f:
            for rname in yd.keys():
                process_region(rname, yd[rname], kerns, f)

