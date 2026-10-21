#!/usr/bin/env bash
# Runs ON the pod, once (push.sh calls it).  Safe to rerun.
#
# Train environment: the pod image's own CUDA PyTorch plus transformers and the fast Qwen3.5
# kernels (flash-linear-attention, causal-conv1d), installed into the image's python3.  Using the
# image's PyTorch avoids pulling a wheel built for a newer CUDA driver than the host has.
#
# vLLM environment (.venv-vllm): only for `ask`, which is most of the work.  Optional: if it
# does not install or does not start, run.sh generates with transformers instead.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/hf}" PATH="$HOME/.local/bin:$PATH"
export PIP_BREAK_SYSTEM_PACKAGES=1  # Ubuntu 24.04 images mark the system python "externally managed"; this pod is disposable

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('image torch', torch.__version__, torch.version.cuda)"

python3 -m pip install --quiet --upgrade "transformers @ git+https://github.com/huggingface/transformers.git" safetensors huggingface_hub flash-linear-attention pyarrow
python3 -m pip install --quiet --no-build-isolation causal-conv1d \
  || echo "causal-conv1d did not build; the teacher uses the slower reference convolution (still correct)"
python3 -c "import torch, transformers; print('train env: torch', torch.__version__, 'transformers', transformers.__version__, 'on', torch.cuda.get_device_name())"

if [ "${SKIP_VLLM:-0}" != 1 ] && [ ! -x .venv-vllm/bin/python ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  if uv venv --python 3.12 .venv-vllm && uv pip install --python .venv-vllm/bin/python vllm \
      && .venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"; then
    :
  else
    echo "vLLM did not install; ask will use transformers"
    rm -rf .venv-vllm
  fi
fi

python3 -c "from huggingface_hub import snapshot_download; print('teacher at', snapshot_download('Qwen/Qwen3.5-0.8B'))"
echo "setup done"
