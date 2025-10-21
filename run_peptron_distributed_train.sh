#!/bin/bash
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=<path-to-peptron-repo>
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8  -m peptron.train
