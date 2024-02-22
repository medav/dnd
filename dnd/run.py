import argparse
import tempfile
import sys

from .common import *
from . import nvtools
from . import tracer
from . import reports
from . import bench
from . import env

def comma_separated_ints(s): return [int(x) for x in s.split(',')]

def run_bare(args, prog_args : 'list[str]'):
    kerns = tracer.run_kernel_trace(prog_args, None)

    with open(args.outfile, 'w') as f:
        env.dump_yaml(f)

        print(f'bench: None', file=f)

        print(f'trace:', file=f)
        print(f'  region:', file=f)
        for i, k in enumerate(kerns):
            op = Operator(uid=f'unknown:{i}', kerns=[k])
            op.print_yaml(file=f, indent=2)


def run_full(args, prog_args : 'list[str]'):
    with temp_file(suffix='.yaml') as temp_kern_trace_file:
        kerns = tracer.run_kernel_trace(prog_args, temp_kern_trace_file)
        oplists = yaml.safe_load(open(temp_kern_trace_file, 'r'))

    bench_result = bench.run_bench(prog_args)

    with open(args.outfile, 'w') as f:
        env.dump_yaml(f)

        print(f'bench:', file=f)
        for k, v in bench_result.items():
            print(f'  {k}: {v}', file=f)

        print(f'trace:', file=f)
        for rname in oplists.keys():
            tracer.stitch_and_print_region(
                rname, oplists[rname], kerns, f, indent=1)

def main():
    idx = sys.argv.index('--')
    tool_args = sys.argv[1:idx]
    prog_args = sys.argv[idx + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, default='profile.yaml', help='Output profile data to file')
    parser.add_argument('-b', '--bare', action='store_true', help='Run without framework tracing')
    args = parser.parse_args(tool_args)

    if args.bare: run_bare(args, prog_args)
    else: run_full(args, prog_args)

def stat():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kern-file', type=str, default='kerns.yaml', help='Input kernel trace file')

    args = parser.parse_args(sys.argv[1:])
    kdata = read_kerns_yaml(args.kern_file)

    for k, v in kdata.items():
        reports.generate_timeline(v, make_plt=False, title=k)


def overhead():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Profile datafile')
    args = parser.parse_args(sys.argv[1:])

    prof = Profile.from_file(args.filename)

    avg_time_ms = prof.bench['avg_time_ms']
    cuda_avg_time_ms = prof.bench['cuda_avg_time_ms']

    print(f'Kernel Latency: {cuda_avg_time_ms:.3f} ms')
    print(f'Total (Avg) Latency: {avg_time_ms:.3f} ms')
    print(f'Overhead: {(avg_time_ms - cuda_avg_time_ms) / avg_time_ms * 100:.2f}%')


if __name__ == '__main__':
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == 'main': main()

