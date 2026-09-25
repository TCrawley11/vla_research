#!/usr/bin/env bash
# Serve the cached quantized 27B model through llama.cpp's local API.
# No downloads or dependency changes. MODEL and MMPROJ can point at other files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-27B-MTP-GGUF}"
MODEL_REVISION="${MODEL_REVISION:-5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace}"

cached_file() {
  "$PYTHON" - "$MODEL_REPO" "$1" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import try_to_load_from_cache
path = try_to_load_from_cache(sys.argv[1], sys.argv[2], revision=sys.argv[3])
if not isinstance(path, str):
    raise SystemExit(f"Not cached: {sys.argv[1]}/{sys.argv[2]}. Set MODEL and MMPROJ to local files.")
print(path)
PY
}

MODEL="${MODEL:-$(cached_file Qwen3.6-27B-Q3_K_M.gguf)}"
MMPROJ="${MMPROJ:-$(cached_file mmproj-BF16.gguf)}"
[[ -x "$LLAMA_SERVER" ]] || { echo "Set LLAMA_SERVER to your llama-server executable." >&2; exit 2; }
[[ -f "$MODEL" && -f "$MMPROJ" ]] || { echo "MODEL and MMPROJ must exist." >&2; exit 2; }

exec "$LLAMA_SERVER" \
  --model "$MODEL" --mmproj "$MMPROJ" \
  --alias "${SERVED_NAME:-qwen3.6-27b-q3_k_m}" \
  --host 127.0.0.1 --port "${PORT:-8001}" \
  --ctx-size "${MAX_MODEL_LEN:-32768}" --parallel 1 \
  --threads "${CPU_THREADS:-8}" --no-mmproj-offload \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --fit-target "${FIT_TARGET:-768}" --reasoning-format deepseek
