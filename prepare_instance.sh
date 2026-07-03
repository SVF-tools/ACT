#!/bin/bash
# VNN-COMP 2026 prepare_instance.sh for ACT.
# Args: $1="v1", $2=benchmark name, $3=onnx path, $4=vnnlib path.
# ACT loads the onnx+vnnlib lazily at run time, so there is nothing to
# precompute per instance; this only performs a sanity check and exits 0.

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "prepare_instance.sh: expected first argument '$VERSION_STRING', got '$1'"
    exit 1
fi

BENCHMARK="$2"
ONNX="$3"
VNNLIB="$4"

echo "ACT prepare: benchmark='$BENCHMARK' onnx='$ONNX' vnnlib='$VNNLIB'"

# Kill any ACT runner left over from a previous instance: the harness kills
# 'conda run' on timeout but the signal does not reliably reach the python
# child, and a leaked process pins the GPU at 100% and starves this instance.
pkill -f act_run_instance.py 2>/dev/null || true
sleep 1

nvidia-smi || true
exit 0
