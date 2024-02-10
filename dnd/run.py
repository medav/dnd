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

def main():
    idx = sys.argv.index('--')
    tool_args = sys.argv[1:idx]
    prog_args = sys.argv[idx + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, default='profile.yaml', help='Output profile data to file')
    args = parser.parse_args(tool_args)

    with tempfile.NamedTemporaryFile(suffix='.yaml') as temp_kern_trace_file:
        kerns = tracer.run_kernel_trace(prog_args, temp_kern_trace_file.name)
        oplists = yaml.safe_load(open(temp_kern_trace_file.name, 'r'))

    avg_time = bench.run_bench(prog_args)

    with open(args.outfile, 'w') as f:
        env.dump_yaml(f)

        print(f'bench:', file=f)
        print(f'  avg_time: {avg_time}', file=f)

        print(f'trace:', file=f)
        for rname in oplists.keys():
            tracer.stitch_and_print_region(
                rname, oplists[rname], kerns, f, indent=1)


def stat():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kern-file', type=str, default='kerns.yaml', help='Input kernel trace file')

    args = parser.parse_args(sys.argv[1:])
    kdata = read_kerns_yaml(args.kern_file)

    for k, v in kdata.items():
        reports.generate_timeline(v, make_plt=False, title=k)


def overhead():
    idx = sys.argv.index('--')
    tool_args = sys.argv[1:idx]
    prog_args = sys.argv[idx + 1:]

    env = os.environ.copy()
    env['DND_MODE'] = 'trace'

    kerns = nvtools.run_ncu_nsys(
        prog_args,
        ncu_config=nvtools.NcuConfig(env=env),
        nsys_config=nvtools.NsysConfig(env=env)
    )

    avg_time_ms = bench.run_bench(prog_args) * 1e3

    klat_ms = sum(k.lat for k in kerns) / 1e6
    print(f'Kernel Latency: {klat_ms:.3f} ms')
    print(f'Total (Avg) Latency: {avg_time_ms:.3f} ms')
    print(f'Overhead: {(avg_time_ms - klat_ms) / avg_time_ms * 100:.2f}%')


if __name__ == '__main__':
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == 'main': main()

