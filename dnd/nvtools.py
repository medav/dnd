import os
import sys
import functools
import tempfile
import pandas as pd
from dataclasses import dataclass, field
import subprocess
import json

from .common import *

default_metrics = [
    'gpu__time_duration.sum',
    'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    # 'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed'
]

NCU_PATH = get_optional_env('NCU_PATH', 'ncu')
NSYS_PATH = get_optional_env('NSYS_PATH', 'nsys')

if not command_exists(NCU_PATH):
    print(f'NCU not found (NCU_PATH={NCU_PATH}). Please set NCU_PATH to valid ncu binary.', file=sys.stderr)
    sys.exit(1)

if not command_exists(NSYS_PATH):
    print(f'NSYS not found (NSYS_PATH={NSYS_PATH}). Please set NSYS_PATH to valid nsys binary.', file=sys.stderr)
    sys.exit(1)

@functools.lru_cache
def get_ncu_version(path : str):
    return subprocess.check_output([path, '--version']) \
        .decode().strip().split('\n')[-1].split(' ')[1]


@functools.lru_cache
def get_nsys_version(path : str):
    return subprocess.check_output([path, '--version']) \
        .decode().strip().split(' ')[-1].split('-')[0]

class Reader(object):
    def __init__(self, g):
        self.g = g
    def read(self, n=0):
        try: return next(self.g)
        except StopIteration: return ''

def read_ncu_output(output):
    it = iter(output.split('\n'))
    line = ''
    while not line.startswith('"ID","Process ID","Process Name",'):
        line = next(it)

    yield line + '\n'

    for line in it: yield line + '\n'

def read_nsys_output(output):
    it = iter(output.split('\n'))
    line = ''
    while not line.startswith('Start (ns)'):
        line = next(it)

    yield line + '\n'

    for line in it: yield line + '\n'

@dataclass(frozen=True)
class NcuConfig:
    replay_mode : str = 'application'
    metrics : 'list[str]' = field(default_factory=lambda: default_metrics)
    use_nvtx : bool = False
    env : 'dict[str, str]' = field(default_factory=lambda: os.environ.copy())


@dataclass(frozen=True)
class NsysConfig:
    env : 'dict[str, str]' = field(default_factory=lambda: os.environ.copy())

def run_ncu_nsys(
    prog_args : 'list[str]',
    use_cuda_profiler_api : bool = False,
    ncu_config : NcuConfig = NcuConfig(),
    nsys_config : NsysConfig = NsysConfig(),
    quiet : bool = False
) -> 'list[Kernel]':

    if not quiet: print('>>> Running NCU...')

    ncu_env = ncu_config.env.copy()
    ncu_env['CUDA_LAUNCH_BLOCKING'] = '1'

    if ncu_config.use_nvtx:
        nvtx_args = [
            '--nvtx',
            '--print-nvtx-rename', 'kernel'
        ]
    else:
        nvtx_args = []

    cmdline = [
        NCU_PATH,
        '--csv',
        *nvtx_args,
        '--target-processes', 'all',
        '--profile-from-start', 'no' if use_cuda_profiler_api else 'yes',
        '--replay-mode', ncu_config.replay_mode,
        '--metrics', ','.join(ncu_config.metrics)
    ] + prog_args

    with check_subprocess():
        ncu_output = subprocess.check_output(cmdline, env=ncu_env).decode()

    if not quiet:print('>>> Done!')


    ncu_df = pd.read_csv(
        Reader(read_ncu_output(ncu_output)),
        low_memory=False,
        thousands=r',')

    ncu_names = dict()
    ncu_metrics = dict()

    for row in ncu_df.iterrows():
        row = row[1]
        ncu_names[row['ID']] = row['Kernel Name']
        if row['ID'] not in ncu_metrics: ncu_metrics[row['ID']] = dict()
        ncu_metrics[row['ID']][row['Metric Name']] = row['Metric Value']


    with tempfile.NamedTemporaryFile(suffix='.nsys-rep') as temp_nsys_rep:
        if not quiet: print('>>> Running NSYS...')
        nsys_env = nsys_config.env.copy()

        cmdline = [
            NSYS_PATH,
            'profile',
            '-t', 'cuda,cudnn,cublas',
            '-o', temp_nsys_rep
        ] + prog_args

        if use_cuda_profiler_api:
            cmdline += [
                '--capture-range=cudaProfilerApi',
                '--capture-range-end=stop'
            ]

        with check_subprocess():
            subprocess.check_output(cmdline, env=nsys_env).decode()

        if not quiet: print('>>> Done!')

        report_name = {
            '2022.1.3.3': 'gputrace',
        }.get(get_nsys_version(NSYS_PATH), 'cuda_gpu_trace')

        stats_cmdline = [
            NSYS_PATH,
            'stats',
            '-r', report_name,
            '-f', 'csv',
            temp_nsys_rep
        ]

        with check_subprocess():
            stats_output = subprocess.check_output(stats_cmdline).decode()

    nsys_df = pd.read_csv(
        Reader(read_nsys_output(stats_output)),
        low_memory=False,
        thousands=r',')

    ordered_ids = sorted(ncu_names.keys())
    ordered_names = [ncu_names[i] for i in ordered_ids]

    if ncu_config.use_nvtx:
        nvtx_ranges = []
        new_names = []

        for n in ordered_names:
            if '/' in n:
                nvtx_range, name = n.split('/')
                nvtx_ranges.append(nvtx_range)
                new_names.append(name)
            else:
                nvtx_ranges.append('')
                new_names.append(n)

        ordered_names = new_names

    else: nvtx_ranges = [''] * len(ordered_names)

    kerns = []

    kid = 0
    for row in nsys_df.iterrows():
        row = row[1]
        if kid >= len(ordered_ids): break
        if row['Name'] != ordered_names[kid]: continue

        kerns.append(Kernel(
            kid=kid,
            name=row['Name'],
            nvtx_range=nvtx_ranges[kid],
            grid= (int(row['GrdX']), int(row['GrdY']), int(row['GrdZ'])),
            block=(int(row['BlkX']), int(row['BlkY']), int(row['BlkZ'])),
            reg_per_thread=int(row['Reg/Trd']),
            static_smem=int(float(row['StcSMem (MB)']) * 2**20),
            dynamic_smem=int(float(row['DymSMem (MB)']) * 2**20),
            metrics=ncu_metrics[ordered_ids[kid]]
        ))

        kid += 1

    assert kid == len(ordered_ids)
    return kerns

