# DND: Dnd's Not Dlprof!

[![PyPI version](https://badge.fury.io/py/dndlprof.svg)](https://badge.fury.io/py/dndlprof)

DND is a simple tool for profiling and tracing PyTorch programs on NVIDIA GPUs.
If uses NVIDIA's NSIGHT Systems and NSIGHT Compute along-side PyTorch 2.0's
Dynamo graph capturing system to provide framework operator and CUDA kernel
level breakdowns of a Deep Learning application. Check out the
[Sample App](./dnd/sample_app.py) for an example of an instrumented application.


## Install
```bash
$ pip install dndlprof
```

## Quick Start
```bash
$ dnd -- python -m dnd.sample_app
```
