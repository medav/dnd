import numpy as np
from .common import *

try:
    import matplotlib.pyplot as plt
    has_plt = True
except ImportError:
    print(f'ERROR: matplotlib appears to not be installed.')
    print(f'ERROR: Will not generate a plot for this app.')
    print(f'ERROR: run `pip install matplotlib` to install')
    has_plt = False

def generate_timeline(ops : 'list[Operator]', sortby='sm', make_plt=True, title='timeline'):
    kerns : 'list[Kernel]'
    kerns = sum((op.kerns for op in ops), [])
    global has_plt
    tot_time = sum(k.lat for k in kerns)
    frac_times = np.zeros(len(kerns))
    sm_util = np.zeros(len(kerns))
    tc_util = np.zeros(len(kerns))
    dram_util = np.zeros(len(kerns))

    k : Kernel
    for i, k in enumerate(kerns):
        frac_times[i] = k.lat / tot_time
        sm_util[i] = k.sm_util
        tc_util[i] = k.tensor_util
        dram_util[i] = k.dram_util

    sort_idx = sm_util.argsort()

    sort_idx = {
        'sm': sm_util.argsort(),
        'tc': tc_util.argsort(),
        'dram': dram_util.argsort()
    }[sortby]

    frac_times = frac_times[sort_idx]
    sm_util = sm_util[sort_idx] * 100
    tc_util = tc_util[sort_idx] * 100
    dram_util = dram_util[sort_idx] * 100

    cuml_sm_util = np.array([
        np.average(sm_util[:i + 1], weights=frac_times[:i + 1]) for i in range(len(frac_times))
    ])

    cuml_tc_util = np.array([
        np.average(tc_util[:i + 1], weights=frac_times[:i + 1]) for i in range(len(frac_times))
    ])

    cuml_dram_util = np.array([
        np.average(dram_util[:i + 1], weights=frac_times[:i + 1]) for i in range(len(frac_times))
    ])

    if has_plt and make_plt:
        fig = plt.figure(figsize=(3.375, 2))
        xs = np.cumsum(frac_times)
        if sortby == 'sm':
            plt.plot(xs, sm_util, label='SM Util.', linewidth=1)
            plt.plot(xs, cuml_dram_util, label='Avg. DRAM Util.', linewidth=1, linestyle='--')
            plt.plot(xs, cuml_tc_util, label='Avg. Tensor Util.', linewidth=1, linestyle='--')
        elif sortby == 'tc':
            plt.plot(xs, tc_util, label='Tensor Util.', linewidth=1)
            plt.plot(xs, cuml_dram_util, label='Avg. DRAM Util.', linewidth=1, linestyle='--')
            plt.plot(xs, cuml_sm_util, label='Avg. SM Util.', linewidth=1, linestyle='--')
        else:
            plt.plot(xs, dram_util, label='DRAM Util.', linewidth=1)
            plt.plot(xs, cuml_sm_util, label='Avg. SM Util.', linewidth=1, linestyle='--')
            plt.plot(xs, cuml_tc_util, label='Avg. Tensor Util.', linewidth=1, linestyle='--')
        plt.legend(loc='upper left', fontsize=6)
        plt.xlabel('Fraction of time', fontsize=8)
        plt.ylabel('% Utilization', fontsize=8)
        plt.title(title, fontsize=8)
        plt.tick_params(axis='both', labelsize=6)
        plt.xlim([0.0, 1.0])
        plt.tight_layout(pad=0.2)
        plt.savefig(f'{title}.pdf')

    print('=' * 10, title, '=' * 10)
    print(f'{"Kernel".ljust(20)}\tFrac T\tSM\tA.DRAM\tA.TC')
    for i in range(len(kerns)):
        print(f'{shortstr(kerns[i].sanitized_name).ljust(20)}\t{frac_times[i]:.2f}\t{sm_util[i]:.2f}%\t{cuml_dram_util[i]:.2f}%\t{cuml_tc_util[i]:.2f}%')

    print('=' * (22 + len(title)))
