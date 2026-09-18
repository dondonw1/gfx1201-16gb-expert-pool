#!/usr/bin/env bash
#
# Example launcher for gfx1201 (RX 9070 XT, 16 GB) built natively against host ROCm.
#
# Build the server first:
#   cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 && cmake --build build -j"$(nproc)"
#
# Usage:
#   MODEL_DIR=/path/to/models ./scripts/run-gfx1201-example.sh
#
# Read the README before changing the pool knobs: the per-tensor slot size and
# the rail decide whether the pool fits next to the KV cache and compute buffers.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="${SERVER_BIN:-$REPO_DIR/build/bin/llama-server}"
ROCM_LIB_DIR="${ROCM_LIB_DIR:-/opt/rocm/lib}"
MODEL_DIR="${MODEL_DIR:-/home/ub/llm/models}"
MODEL="${MODEL:-Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf}"
MMPROJ="${MMPROJ:-next-mmproj-F16.gguf}"

PORT="${PORT:-1234}"
CONTEXT="${CONTEXT:-131072}"
N_CPU_MOE="${N_CPU_MOE:-48}"
THREADS="${THREADS:-8}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
CACHE_RAM="${CACHE_RAM:-4096}"
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-2}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

# Slots are per offloaded expert weight tensor (144 tensors here), not tokens.
# One slot measured 128 MiB for this model against 76.5 MiB on the gfx906
# validation model, so 40 slots is the comparable budget, not the gfx906 value 66.
EXPERT_CACHE="${EXPERT_CACHE:-40}"
POOL_RAIL_MIB="${POOL_RAIL_MIB:-2048}"
POOL_CAP_MIB="${POOL_CAP_MIB:-}"
POOL_PROFILE="${POOL_PROFILE:-}"

LOG_FILE="${LOG_FILE:-/tmp/llama-server-gfx1201.log}"
PID_FILE="${PID_FILE:-/tmp/llama-server-gfx1201.pid}"

MODEL_PATH="$MODEL_DIR/$MODEL"
MMPROJ_PATH="$MODEL_DIR/$MMPROJ"

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "[ERROR] Server not found: $SERVER_BIN"
  echo "[ERROR] Build it with -DGGML_HIP=ON -DGPU_TARGETS=gfx1201"
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[ERROR] Model not found: $MODEL_PATH"
  exit 1
fi
if [[ ! -f "$MMPROJ_PATH" ]]; then
  echo "[ERROR] Multimodal projector not found: $MMPROJ_PATH"
  exit 1
fi

# True bind test, so a listener is detected without parsing `ss` output.
port_free() {
  python3 - "$PORT" <<'PY' 2>/dev/null
import errno, socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError as e:
    print("in-use" if e.errno == errno.EADDRINUSE else f"cannot-bind:{e.strerror}")
else:
    print("free")
finally:
    s.close()
PY
}

PORT_STATE="$(port_free)"
if [[ "$PORT_STATE" != "free" ]]; then
  echo "[ERROR] Port $PORT is not usable: $PORT_STATE"
  exit 1
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[+] Stopping previous instance (pid $(cat "$PID_FILE"))"
  kill "$(cat "$PID_FILE")" || true
  sleep 2
fi

# gfx1201 is native in ROCm 10, so no HSA_OVERRIDE_GFX_VERSION is needed.
POOL_ENV=("LLAMA_MOE_POOL_RAIL_MIB=$POOL_RAIL_MIB" "GGML_MOE_POOL_STATS=1")
if [[ -n "$POOL_CAP_MIB" ]]; then
  POOL_ENV+=("LLAMA_MOE_POOL_CAP_MIB=$POOL_CAP_MIB")
fi
if [[ -n "$POOL_PROFILE" ]]; then
  POOL_ENV+=("LLAMA_MOE_POOL_PROFILE=$POOL_PROFILE")
fi

echo "[+] Server:  $SERVER_BIN"
echo "[+] Model:   $MODEL_PATH"
echo "[+] Pool:    $EXPERT_CACHE slots, rail ${POOL_RAIL_MIB} MiB"
echo "[+] Context: $CONTEXT, batch $BATCH / ubatch $UBATCH, threads $THREADS"
echo "[+] Log:     $LOG_FILE"

env "LD_LIBRARY_PATH=$ROCM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "${POOL_ENV[@]}" \
  "$SERVER_BIN" \
  -m "$MODEL_PATH" \
  --mmproj "$MMPROJ_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  -ngl 99 \
  --n-cpu-moe "$N_CPU_MOE" \
  -c "$CONTEXT" \
  -b "$BATCH" \
  -ub "$UBATCH" \
  --threads "$THREADS" \
  --threads-batch "$THREADS" \
  --cache-ram "$CACHE_RAM" \
  --ctx-checkpoints "$CTX_CHECKPOINTS" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  --flash-attn on \
  --load-mode none \
  --metrics \
  --jinja \
  --moe-expert-cache "$EXPERT_CACHE" \
  </dev/null >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "[+] Server pid $SERVER_PID"

# Model load takes minutes for a 92 GB multi-shard GGUF.
for ((i = 1; i <= 180; i++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ERROR] Server exited, last log lines:"
    tail -n 30 "$LOG_FILE"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[+] Healthy after ~$((i * 5))s"
    break
  fi
  sleep 5
done

# Admission may fall back to fewer slots or none at all, so read the markers.
grep -E 'expert pool (status|first-use)=' "$LOG_FILE" | tail -n 2 || true
echo "[i] Check limited_by=slots|rail|cap and copy_bytes, not only the hit rate."
echo "[i] Stop with: kill \$(cat $PID_FILE)"
