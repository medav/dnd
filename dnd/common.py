import os
import sys
from dataclasses import dataclass
from distutils.spawn import find_executable
import functools
import json
import yaml

def get_optional_env(name : str, default : str = None):
    return os.environ.get(name, default)

def get_required_env(name : str):
    if name in os.environ: return os.environ[name]
    raise RuntimeError(f'Environment variable {name} is required')

def shortstr(s : str, maxlen : int = 20):
    if len(s) <= maxlen: return s
    return s[:maxlen - 3] + '...'

def command_exists(name):
    return find_executable(name) is not None

def parse_int_tuple(s : str):
    return tuple(map(int, s.replace('(', '').replace(')', '').split(',')))

@dataclass
class Kernel:
    kid : int
    name : str
    nvtx_range : str
    grid : tuple
    block : tuple
    reg_per_thread : int
    static_smem : int
    dynamic_smem : int
    metrics : dict

    @functools.cached_property
    def sanitized_name(self):
        sn = self.name \
            .replace('void ', '') \
            .replace('at::native::', '') \
            .replace('<unnamed>::', '') \
            .replace('cutlass::', '')

        if '<' in sn: sn = sn[:sn.index('<')]
        if '(' in sn: sn = sn[:sn.index('(')]
        return sn

    @property
    def threads_per_block(self): return self.block[0] * self.block[1] * self.block[2]

    @property
    def reg_per_block(self): return self.reg_per_thread * self.threads_per_block

    @property
    def tot_smem(self): return self.static_smem + self.dynamic_smem

    @property
    def lat(self): return self.metrics['gpu__time_duration.sum']

    @property
    def dram_util(self): return self.metrics['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'] / 100

    @property
    def sm_util(self): return self.metrics['sm__throughput.avg.pct_of_peak_sustained_elapsed'] / 100

    @property
    def tensor_util(self): return self.metrics['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'] / 100

    @property
    def yaml_repr(self):
        return '{' + ', '.join([
            f'kid: {self.kid}',
            f'name: "{self.name}"',
            f'nvtx_range: "{self.nvtx_range}"',
            f'grid: "{self.grid}"',
            f'block: "{self.block}"',
            f'reg_per_thread: {self.reg_per_thread}',
            f'static_smem: {self.static_smem}',
            f'dynamic_smem: {self.dynamic_smem}',
            f'metrics: {json.dumps(self.metrics)}'
        ]) + '}'

    @staticmethod
    def from_yaml(yd):
        return Kernel(
            kid=yd['kid'],
            name=yd['name'],
            nvtx_range=yd['nvtx_range'],
            grid=parse_int_tuple(yd['grid']),
            block=parse_int_tuple(yd['block']),
            reg_per_thread=yd['reg_per_thread'],
            static_smem=yd['static_smem'],
            dynamic_smem=yd['dynamic_smem'],
            metrics=yd['metrics']
        )

@dataclass
class Operator:
    uid : str
    kerns : 'list[Kernel]'

    def print_yaml(self, file=sys.stdout):
        print(f'  - uid: {self.uid}', file=file)
        print(f'    kerns:', file=file)
        for k in self.kerns:
            print(f'      - {k.yaml_repr}', file=file)

    @staticmethod
    def from_yaml(yd):
        return Operator(
            uid=yd['uid'],
            kerns=[Kernel.from_yaml(kd) for kd in yd['kerns']]
        )

def read_kerns_yaml(filename : str) -> 'dict[str, list[Operator]]':
    with open(filename) as f:
        yd = yaml.safe_load(f)

    return {k: [Operator.from_yaml(opd) for opd in yd[k]] for k in yd}

