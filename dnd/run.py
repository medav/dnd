import argparse
import sys
import functools
import itertools
import numpy as np
import random
from enum import Enum

from .common import *
from . import prof
from . import tracer
from . import reports

def main():
    idx = sys.argv.index('--')
    tool_args = sys.argv[1:idx]
    prog_args = sys.argv[idx + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--out-kern-file', type=str, default='kerns.yaml', help='Output kernel trace to file')
    args = parser.parse_args(tool_args)

    tracer.run_tracing(prog_args, kerns_file=args.out_kern_file)

def stat():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kern-file', type=str, default='kerns.yaml', help='Input kernel trace file')

    args = parser.parse_args(sys.argv[1:])
    kdata = read_kerns_yaml(args.kern_file)

    for k, v in kdata.items():
        reports.generate_timeline(v, make_plt=False, title=k)



if __name__ == '__main__':
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == 'main': main()

