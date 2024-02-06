import os
import sys
import torch
from .common import *

appname = get_optional_env('APP', None)
dev = torch.device(get_optional_env('DEV', 'cuda:0'))
bs = int(get_optional_env('BS', '1'))

dtype = {
    'FP16': torch.float16,
    'FP32': torch.float32,
    'FP64': torch.float64,
}[get_optional_env('DTYPE', 'FP32')]


def print_config():
    print(f'==================================================================')
    print(f'Benchmark Parameters:')
    print(f'  Application Name: {appname}')
    print(f'  Batch Size: {bs}')
    print(f'  Data Type: {dtype}')
    # print(f'  Num Warmup Passes: {nw}')
    # print(f'  Num Iters: {ni}')
    print(f'==================================================================')

