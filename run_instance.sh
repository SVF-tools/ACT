#!/bin/bash
# VNN-COMP 2026 run_instance.sh for ACT.
# Args: $1="v1", $2=benchmark, $3=onnx, $4=vnnlib, $5=results file, $6=timeout(s).
# Runs ACT on the single instance and writes the result token (+counterexample
# for sat) to $5 via act_run_instance.py. The Python self-limits to
# timeout-margin so it returns before the harness kill (timeout+60s). Uses the
# offline 'gain' config, or the LEAPS 'gain+llm' search when an LLM key is present.

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "run_instance.sh: expected first argument '$VERSION_STRING', got '$1'"
    exit 1
fi

BENCHMARK="$2"
ONNX="$3"
VNNLIB="$4"
RESULTS="$5"
TIMEOUT="$6"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "ACT run: benchmark='$BENCHMARK' onnx='$ONNX' vnnlib='$VNNLIB' results='$RESULTS' timeout=$TIMEOUT"
nvidia-smi || true

# Locate and initialise conda. The VNN-COMP harness runs this in a non-interactive,
# non-login shell where ~/.bashrc (and thus conda's PATH hook) is NOT sourced, so
# 'conda' is usually absent from PATH even though install_tool.sh already created the
# act-py312 env. Discover the base the same way install_tool.sh installs it
# ($HOME/miniconda3) and source conda.sh; without this the 'conda run' below fails
# with exit code 127 (conda: command not found).
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniconda" "$HOME/anaconda" \
                /opt/conda /opt/miniconda3 /usr/local/miniconda3 \
                /home/ubuntu/miniconda3 /root/miniconda3; do
        if [ -x "$base/bin/conda" ]; then
            CONDA_BASE="$base"
            break
        fi
    done
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "run_instance.sh: could not locate a conda installation (looked for \$HOME/miniconda3, /opt/conda, ...); run install_tool.sh first" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Optional post-install secrets (e.g. the LLM API key) kept out of this public repo;
# create it on the run machine, it is gitignored (see act_llm_secrets.sh.example).
[ -f "$SCRIPT_DIR/act_llm_secrets.sh" ] && source "$SCRIPT_DIR/act_llm_secrets.sh"

# 'gain' is the offline dual_alpha_eta+gain config (no network). With an LLM API key
# present, switch to the LEAPS closed-loop 'gain+llm' search under a short per-call
# cap so a slow reply falls back to the sound baseline within the instance timeout.
CONFIG_ARGS=(--config gain)
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    CONFIG_ARGS=(--config gain+llm --llm-backend openrouter \
                 --llm-model google/gemini-2.5-flash-lite --llm-timeout 5)
fi

conda run --no-capture-output -n act-py312 python "$SCRIPT_DIR/act_run_instance.py" \
    "$ONNX" "$VNNLIB" "$RESULTS" "$TIMEOUT" \
    "${CONFIG_ARGS[@]}" --max-batch-size auto --fuzzing-seconds 5
