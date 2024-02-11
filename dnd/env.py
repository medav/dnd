import os
import sys
import torch
import enum
from .common import *

class Mode(enum.Enum):
    BENCH = 0
    TRACE = 1

    @staticmethod
    def from_str(s : str):
        return {
            'bench': Mode.BENCH,
            'trace': Mode.TRACE,
        }[s]

dev = torch.device(get_optional_env('DND_DEV', 'cuda:0'))
bs = int(get_optional_env('DND_BS', '1'))

dtype = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}[get_optional_env('DND_DTYPE', 'float32')]

mode = Mode.from_str(get_optional_env('DND_MODE', 'bench'))
bench_nw = int(get_optional_env('DND_NW', '10'))
bench_ni = int(get_optional_env('DND_NI', '100'))

def print_config():
    print(f'==================================================================')
    print(f'Benchmark Parameters:')
    print(f'  Device: {dev}')
    print(f'  (Bench) Num Warmup Iters: {bench_nw}')
    print(f'  (Bench) Num Bench Iters: {bench_ni}')
    print(f'  Batch Size: {bs}')
    print(f'  Data Type: {dtype}')
    print(f'==================================================================')

def dump_yaml(f, indent=0):
    print(f'{"  " * indent}params:', file=f)
    print(f'{"  " * indent}  Device: {dev}', file=f)
    print(f'{"  " * indent}  Num Warmup Iters: {bench_nw}', file=f)
    print(f'{"  " * indent}  Num Bench Iters: {bench_ni}', file=f)
    print(f'{"  " * indent}  Batch Size: {bs}', file=f)
    print(f'{"  " * indent}  Data Type: {dtype}', file=f)
