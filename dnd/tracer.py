
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
from . import nvtools

tracefile = get_optional_env('DND_TRACEFILE')

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


def run_kernel_trace(prog_args, kernel_trace_file):
    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['DND_MODE'] = 'trace'

    ncu_env = env.copy()
    if kernel_trace_file is not None:
        env['DND_TRACEFILE'] = kernel_trace_file

    return nvtools.run_ncu_nsys(
        prog_args,
        ncu_config=nvtools.NcuConfig(
            replay_mode=dnd_config.ncu_replay_mode,
            use_nvtx=True,
            env=ncu_env
        ),
        nsys_config=nvtools.NsysConfig(
            num_samples=dnd_config.nsys_num_samples,
            env=env
        )
    )

def lookup_kerns(kerns : 'list[Kernel]', uid : str):
    return list(filter(lambda k: k.nvtx_range == uid, kerns))

def stitch_ops_kerns(
    ops : 'list[dict]',
    kerns : 'list[Kernel]'
) -> 'list[Operator]':
    return [
        Operator(uid=op['uid'], kerns=lookup_kerns(kerns, op['uid']))
        for op in ops
    ]

def stitch_and_print_region(
    rname : str,
    orig_ops : 'list[dict]',
    kerns : 'list[Kernel]',
    outfile,
    indent : int = 0
):
    if orig_ops is None: orig_ops = []

    print(f'{"  " * indent}{rname}:', file=outfile)
    op : Operator
    for op in stitch_ops_kerns(orig_ops, kerns):
        op.print_yaml(file=outfile, indent=indent + 1)


def run_tracing(prog_args, kerns_file):
    with temp_file(suffix='.yaml') as temp_kern_trace_file:
        kerns = run_kernel_trace(prog_args, temp_kern_trace_file)
        yd = yaml.safe_load(open(temp_kern_trace_file, 'r'))

        with open(kerns_file, 'w') as f:
            for rname in yd.keys():
                stitch_and_print_region(rname, yd[rname], kerns, f, indent=0)

