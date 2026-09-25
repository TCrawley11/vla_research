#!/usr/bin/env bash
# Serve Qwen3.5-9B Q6_K + mmproj for carla_data_pipeline.annotate.
#
# Qwen3.5 GGUF is architecture `qwen35` (hybrid GDN). The generic vLLM-GGUF
# mapper (historically weights_adapter/default.py, now transformers.py) cannot
# load it. This repo pins vllm-gguf-plugin from git so Qwen35GGUFAdapter in
# weights_adapter/qwen3_5.py is registered. --hf-config-path points at Qwen's
# Hugging Face config, not GGUF metadata alone.
#
# The vision tower is a sibling mmproj-BF16.gguf; the plugin picks up
# *mmproj*.gguf next to the backbone. Both files must live in MODEL_DIR.
#
# Usage:
#   scripts/serve_annotator.sh
#   PORT=8001 scripts/serve_annotator.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_DIR:-models}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-9B-GGUF}"
BACKBONE="${BACKBONE:-Qwen3.5-9B-Q6_K.gguf}"
MMPROJ="${MMPROJ:-mmproj-BF16.gguf}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-9B}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-9b-q6_k}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEM="${GPU_MEM:-0.85}"
# Git pin that registers Qwen35GGUFAdapter (weights_adapter/qwen3_5.py).
PLUGIN_REV="${PLUGIN_REV:-fb973ad784f38b98b054e136bec3414b7cd8494d}"
PLUGIN_URL="git+https://github.com/vllm-project/vllm-gguf-plugin@${PLUGIN_REV}"

if ! uv run --group infer python -c "import vllm, vllm_gguf_plugin" >/dev/null 2>&1; then
  echo "installing vLLM + vllm-gguf-plugin@${PLUGIN_REV}"
  uv sync --group infer
  # Must share vLLM's torch. Isolated builds pick a different torch and
  # then fail to load Qwen3.5's non-standard GGUF mapping.
  uv pip install "${PLUGIN_URL}" --no-build-isolation
fi

download() {
  local file="$1"
  if [[ -f "${MODEL_DIR}/${file}" ]]; then
    echo "have ${MODEL_DIR}/${file}"
    return
  fi
  echo "downloading ${MODEL_REPO} ${file} -> ${MODEL_DIR}/"
  mkdir -p "${MODEL_DIR}"
  uv run hf download "${MODEL_REPO}" "${file}" --local-dir "${MODEL_DIR}"
}

download "${BACKBONE}"
download "${MMPROJ}"

if [[ ! -f "${MODEL_DIR}/${MMPROJ}" ]]; then
  echo "missing ${MODEL_DIR}/${MMPROJ}; 6-camera vision will not load" >&2
  exit 1
fi

exec uv run --group infer vllm serve "${MODEL_DIR}/${BACKBONE}" \
  --tokenizer "${TOKENIZER}" \
  --hf-config-path "${TOKENIZER}" \
  --served-model-name "${SERVED_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --limit-mm-per-prompt '{"image": 6}' \
  --reasoning-parser qwen3 \
  --port "${PORT}"
