#!/bin/bash
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export CUDA_LAUNCH_BLOCKING=1

PYTHONPATH=. python -m peptron.train