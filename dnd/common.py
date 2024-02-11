import os
import sys
from dataclasses import dataclass
from distutils.spawn import find_executable
import functools
import json
import yaml
import subprocess
import tempfile
from contextlib import contextmanager

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

@contextmanager
def check_subprocess():
    try:
        yield
    except subprocess.CalledProcessError as e:
        print('Error: subprocess returned with an error!')
        print(e)
        if e.stdout is not None:
            print('==== STDOUT ====')
            print(e.stdout.decode())
        if e.stderr is not None:
            print('==== STDERR ====')
            print(e.stderr.decode())
        print('================')
        raise

@contextmanager
def temp_file(suffix : str, delete : bool = True):
    fname = tempfile.mktemp(suffix=suffix)
    yield fname
    if delete: os.remove(fname)

dnd_config_file = get_optional_env('DND_CONFIG', '~/.dnd-config.yaml')

@dataclass
class GlobalConfig:
    ncu_replay_mode : str = 'application'
    nsys_num_samples : int = 10

    @staticmethod
    def from_file(fname : str):
        with open(fname, 'r') as f:
            yd = yaml.safe_load(f)

        defconfig = GlobalConfig()

        return GlobalConfig(
            ncu_replay_mode=yd.get('ncu_replay_mode', defconfig.ncu_replay_mode),
            nsys_num_samples=yd.get('nsys_num_samples', defconfig.nsys_num_samples)
        )

dnd_config_file = os.path.expanduser(dnd_config_file)
if os.path.exists(dnd_config_file):
    dnd_config = GlobalConfig.from_file(dnd_config_file)
else:
    dnd_config = GlobalConfig()

@dataclass
class Kernel:
    kid : int
    name : str
    nsys_avg_lat_ns : float
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
    def ncu_lat_ns(self): return self.metrics['gpu__time_duration.sum']

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
            f'nsys_avg_lat_ns: {self.nsys_avg_lat_ns}',
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
            nsys_avg_lat_ns=yd['nsys_avg_lat_ns'],
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

    def print_yaml(self, file=sys.stdout, indent=0):
        print(f'{"  " * indent}- uid: {self.uid}', file=file)
        print(f'{"  " * indent}  kerns:', file=file)
        for k in self.kerns:
            print(f'{"  " * indent}    - {k.yaml_repr}', file=file)

    @staticmethod
    def from_yaml(yd):
        return Operator(
            uid=yd['uid'],
            kerns=[Kernel.from_yaml(kd) for kd in yd['kerns']] \
                if yd['kerns'] is not None else []
        )

def read_kerns_yaml(filename : str) -> 'dict[str, list[Operator]]':
    with open(filename) as f:
        yd = yaml.safe_load(f)

    return {k: [Operator.from_yaml(opd) for opd in yd[k]] for k in yd}


@dataclass
class Profile:
    params : dict
    bench : dict
    trace : dict

    @staticmethod
    def from_file(filename : str) -> 'Profile':
        with open(filename) as f:
            yd = yaml.safe_load(f)

        return Profile(
            params=yd['params'],
            bench=yd['bench'],
            trace={
                k: [Operator.from_yaml(opd) for opd in v] \
                    if v is not None else []
                for k, v in yd['trace'].items()
            }
        )
