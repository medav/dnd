
import os
import subprocess
import tempfile
import yaml
import sys
from dataclasses import dataclass
from contextlib import contextmanager

from .common import *

benchfile = get_optional_env('DND_BENCHFILE')


def run_bench(prog_args, nw=10, ni=100):
    with tempfile.NamedTemporaryFile(suffix='.yaml') as bench_result_file:
        env = os.environ.copy()
        env['DND_MODE'] = 'bench'
        env['DND_BENCHFILE'] = bench_result_file.name
        env['DND_NW'] = str(nw)
        env['DND_NI'] = str(ni)

        subprocess.check_call(prog_args, env=env)

        with open(bench_result_file.name, 'r') as f:
            bench_result = yaml.safe_load(f)

        return bench_result

