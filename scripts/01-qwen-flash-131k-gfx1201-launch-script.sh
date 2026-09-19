#!/usr/bin/env bash
#
# run_Qwen3.8-Flash-Next_Vision_Uncensored_IQ4_XS_128K_gfx1201_expert_pool.sh
#
# gfx1201 (RX 9070 XT, 16 GB) adaptation of the MI50 expert-pool launcher.
# Runs the freshly built expert-pool fork of llama.cpp natively (no Docker):
#   <repo>/build/bin/llama-server   (cmake -DGGML_HIP=ON -DGPU_TARGETS=gfx1201,
#   built from the gfx1201-16gb-expert-pool clone, ROCm 10 on the host).
#
# The fork keeps a bounded pool of routed MoE experts in VRAM and remaps expert
# IDs so decode-shaped MoE matmuls run on the accelerator instead of the CPU
# (the stock `--n-cpu-moe 48` path).
#
# Profile carried over from the MI50 validation (same 16 GB VRAM class):
#   * rail-only admission holds POOL_RAIL_MIB free instead of a fixed cap.
#   * daily profile: EXPERT_CACHE=66 with POOL_RAIL_MIB=2048.
#   * quantized KV: q8_0/q8_0 with flash-attn on; batch 2048 / ubatch 512.
#   * all measured throughput numbers on the MI50 (11.76 t/s cache-off,
#     17.87/18.71 t/s hot-cache) are from the OTHER model build (AD-3.84bpw,
#     28 shards). They do NOT transfer to this model/host; re-measure here.
#
# gfx1201 first-run findings (2026, this card + this model):
#   * pool planned/enabled fine, then context creation OOM'd: 66 slots =
#     8,434,851,840 B left ~3.08 GiB free and the pp compute buffer asked for
#     3952.28 MiB. Cause: this model's slots are 128 MiB vs the MI50's 76.5 MiB,
#     plus ubatch 2048 instead of 512. EXPERT_CACHE now defaults to 40.
#   * planner ledger from that run: ceiling=14,948,499,456 (total 15.92 GiB -
#     rail 2048) and cap/budget=9,596,764,160 => used_now+future_reserve =
#     5.35 GB of weights already on device BEFORE the pool was sized.
#   * model geometry (qwen4exp): 48 blocks, head_count=24, head_count_kv=2,
#     key/value_length=256, expert_count=512, expert_used_count=10,
#     full_attention_interval=4 (hybrid attention) with per-layer compress
#     ratios — so KV at 128K is NOT a full 48-layer cache. Measure, don't guess.
#   * to measure the cache-off footprint first: EXPERT_CACHE=0 <this script>
#     then read the 'expert pool' lines for the real headroom before sizing.
#
# Caveats (inherited from the MI50 notes, re-verify on this box):
#   * 128K is the only profile the pool budget was validated at. At longer
#     contexts the startup budget gate can admit zero pools and the server
#     silently runs the stock host-copy path — always check the startup marker.
#   * Output is NOT bit-identical to cache-off: cache-off runs MoE on the CPU
#     backend, cache-on on the accelerator, and their arithmetic differs.
#     Treat as "no measured degradation", not an improvement.
#   * Throughput gain depends on routing locality: a churning single-pass
#     workload ran ~1.8x SLOWER with the pool. Cold caches can regress.
#   * Only decode-shaped graphs use the pool; prompt/prefill batches bypass it.
#   * Speculative decoding / MTP is unsupported with the pool and fails closed.
#     Keep SPEC_TYPE=none.
#   * HOST RAM IS TIGHT on this box: the IQ4_XS model is ~92 GB (3 shards) and
#     the host has ~91 GB total. `--n-cpu-moe 48` keeps the MoE experts as
#     mmap'd host tensors, so expect page-cache pressure / disk refaults.
#     Lower CACHE_RAM (and watch for an n-gram mmap if you add one) if the
#     host starts swapping.
#
# HSA_OVERRIDE_GFX_VERSION is intentionally NOT set: ROCm 10 supports gfx1201
# natively (the fatbin was built for gfx1201). ROCR_VISIBLE_DEVICES defaults to
# 0 = the RX 9070 XT; the Raphael iGPU is not ROCm-capable on this stack, but
# verify device order with `rocminfo | grep gfx` if the ever picks the wrong one.
#
# This script touches only its own pid file / process; it never stops another
# llama.cpp instance. It defaults to PORT=8886 so it can run beside a service
# on 8885.
#
# Usage:
#   ./run_Qwen3.8-Flash-Next_Vision_Uncensored_IQ4_XS_128K_gfx1201_expert_pool.sh
#
# Overrides:
#   PORT=8887 ...                      # for A/B comparisons
#   EXPERT_CACHE=0 ...                 # pool disabled, same profile
#   N_CPU_MOE=44 ...                   # also keep 4 whole MoE layers on GPU
#   CONTEXT=40960 ...                  # small context leaves more pool budget
#   THREADS=12 ...                     # host has 16 cores; default 14
#
# Pool limits (LLAMA_MOE_POOL_CAP_MIB / LLAMA_MOE_POOL_RAIL_MIB):
#   On the MI50 build, 1 pool slot was about 76.5 MiB at the previous model
#   profile. Slot size depends on the expert tensor sizes/quant, so the byte
#   numbers below are indicative only — the admission gate is what matters:
#   admitted = min(EXPERT_CACHE, the largest uniform count that fits while
#   holding POOL_RAIL_MIB free). EXPERT_CACHE is the size knob, the rail is a
#   backstop. POOL_CAP_MIB is optional and unset by default, so only physical
#   memory and the rail bound the pool.
#   Keep total VRAM comfortably under the card's ~16 GiB and the rail at or
#   above 1024 MiB (the fork's floor; it clamps lower values with a warning).
#   The status line reports limited_by=slots|rail|cap: rail means the rail
#   bound the slot count, slots means the requested EXPERT_CACHE was fully
#   admitted, cap means an explicit LLAMA_MOE_POOL_CAP_MIB bound it.
#   POOL_PROFILE=/path/to/routing.profile optionally selects a per-layer
#   routing prior for the weighted planner (format: "blk.<layer> <weight>"
#   lines); unset means the uniform default.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# CONFIG
# =========================
LLAMACPP_DIR="${LLAMACPP_DIR:-$SCRIPT_DIR/./gfx1201-16gb-expert-pool}"
SERVER_BIN="${SERVER_BIN:-$LLAMACPP_DIR/build/bin/llama-server}"
ROCM_LIB_DIR="${ROCM_LIB_DIR:-/opt/rocm/lib}"

MODELS_DIR="${MODELS_DIR:-/home/ub/llm/models}"
MODEL="${MODEL:-Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf}"
MMPROJ="${MMPROJ:-next-mmproj-F16.gguf}"

# Empty = use the chat template embedded in the model (the uncensored build
# ships its own). Point at a .jinja file to override.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

# Validated 128K profile (carried over; re-verify markers on this host)
CONTEXT="${CONTEXT:-131072}"
PORT="${PORT:-1234}"
NGL="${NGL:-99}"                       # all layers; --n-cpu-moe controls experts
THREADS="${THREADS:-8}"               # 16-core host; leave a couple for the OS
THREADS_BATCH="${THREADS_BATCH:-8}"
PARALLEL="${PARALLEL:-1}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
CACHE_RAM="${CACHE_RAM:-4096}"         # watch host RAM: model is ~92 GB of mmap
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-2}"

FLASH_ATTN="${FLASH_ATTN:-on}"         # required for quantized KV
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"   # validated profile used q8_0/q8_0
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

# Expert cache: EXPERT_CACHE is the size knob, in slots PER offloaded expert
# weight tensor (144 tensors for this model), NOT a global count and not tokens.
# Each slot holds one routed expert weight slice.
# MEASURED ON THIS CARD + MODEL (gfx1201, Uncensored IQ4_XS, 128K):
#   * 1 slot = 128 MiB (8,434,851,840 B / 66 slots) — the MI50 validation model
#     used 76.5 MiB/slot, so its 66-slot budget (4.93 GiB) is ~2.9 GiB SMALLER
#     than 66 slots here (7.86 GiB).
#   * 66 slots here left only ~3.08 GiB free, and context creation then failed:
#     "allocating 3952.28 MiB on device 0: cudaMalloc failed: out of memory"
#     (the pp compute buffer, inflated by ubatch 2048).
# 40 slots = 5.0 GiB is the budget-comparable starting point. Raise it only
# while the 'expert pool status=' line and rocm-smi show real headroom.
# Keep it modest when the vision path is used heavily — image buffers come out
# of the same budget.
EXPERT_CACHE="${EXPERT_CACHE:-66}"

# VRAM limits for the pool, in MiB. Per the fork README, the rail is "VRAM held
# back from the pool for compute buffers and later allocations" (the KV cache is
# created after the planner runs). ceiling = total_VRAM - rail, and
# budget = ceiling - used_now(weights already on device) - future_reserve.
# So the rail MUST cover the KV cache + the compute buffers, or context creation
# OOMs even when admission reported limited_by=slots. The pp compute buffer at
# ubatch 2048 measured 3.86 GiB here, which a 2048 MiB rail cannot cover;
# ubatch 512 (the validated profile) keeps it near 1 GiB, which is why rail 2048
# worked on the MI50. Raise the rail to ~4096 if you insist on ubatch 2048.
# Fork floor is 1024 MiB; leaving POOL_CAP_MIB empty means no cap.
POOL_CAP_MIB="${POOL_CAP_MIB:-}"
POOL_RAIL_MIB="${POOL_RAIL_MIB:-2048}"
# optional per-layer routing prior for the weighted planner; empty = uniform
POOL_PROFILE="${POOL_PROFILE:-}"

CPU_MOE="${CPU_MOE:-0}"
N_CPU_MOE="${N_CPU_MOE:-48}"

# MTP / draft speculation is forced off in code while the pool is enabled.
SPEC_TYPE="${SPEC_TYPE:-none}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-64}"
SPEC_NGRAM_MAP_K_SIZE_N="${SPEC_NGRAM_MAP_K_SIZE_N:-32}"
SPEC_NGRAM_MAP_K_SIZE_M="${SPEC_NGRAM_MAP_K_SIZE_M:-32}"

# Server log verbosity. Empty = llama.cpp default, which suppresses the
# INFO-level per-pool hit/miss lines. Set to 4 for a diagnostic run to expose
# the per-pool distribution (rate-limited to one line per 512 pool events).
LOG_LEVEL="${LOG_LEVEL:-}"

# Instance bookkeeping (own instance only, like the Docker launcher's
# own-container rule)
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/run}"
PID_FILE="${PID_FILE:-$RUN_DIR/server.pid}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/server.log}"

# =========================
# MODEL / ASSET CHECKS
# =========================
if [[ ! -x "$SERVER_BIN" ]]; then
  echo "[ERROR] llama-server binary not found: $SERVER_BIN"
  echo "[ERROR] Build it with:"
  echo "[ERROR]   cd $LLAMACPP_DIR"
  echo "[ERROR]   cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1201"
  echo "[ERROR]   cmake --build build -j\$(nproc)"
  exit 1
fi

MODEL_PATH="$MODELS_DIR/$MODEL"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[ERROR] Model not found: $MODEL_PATH"
  exit 1
fi

SHARD_DIR="$(dirname "$MODEL_PATH")"
# prefix used only to count shards (mmproj excluded from the count)
MODEL_PREFIX="${MODEL_PREFIX:-${MODEL%%-0*}}"  # strips the -NNNNN-of-NNNNN shard suffix, excludes mmproj
# `|| true` keeps a no-match `ls` from tripping `set -e`/`pipefail`
# (which would abort silently with rc=2).
SHARD_COUNT=$(ls -1 "$SHARD_DIR"/"$MODEL_PREFIX"*.gguf 2>/dev/null | wc -l) || true
if [[ "$SHARD_COUNT" -lt 1 ]]; then
  echo "[WARN] No .gguf files found in $SHARD_DIR. llama.cpp may fail to load."
fi

if [[ -n "$CHAT_TEMPLATE" ]]; then
  if [[ -f "$CHAT_TEMPLATE" ]]; then
    echo "[+] Chat template: $CHAT_TEMPLATE"
  else
    echo "[WARN] Chat template not found: $CHAT_TEMPLATE — using the model default"
    CHAT_TEMPLATE=""
  fi
fi

MMPROJ_PATH="$MODELS_DIR/$MMPROJ"
if [[ -f "$MMPROJ_PATH" ]]; then
  echo "[+] Multimodal projector: $MMPROJ ($(du -h "$MMPROJ_PATH" | cut -f1))"
else
  echo "[ERROR] Multimodal projector not found: $MMPROJ_PATH"
  exit 1
fi

# =========================
# GPU / RUNTIME CHECKS
# =========================
if [[ ! -e /dev/kfd || ! -e /dev/dri ]]; then
  echo "[WARN] /dev/kfd or /dev/dri not visible from this shell — the server"
  echo "[WARN] will not see the GPU if launched from here (container?). Run on"
  echo "[WARN] the host, or pass the devices into the container."
fi
if ! ls /dev/dri/renderD* >/dev/null 2>&1; then
  echo "[WARN] No /dev/dri/renderD* node visible; ROCm needs it even with a"
  echo "[WARN] render-capable /dev/kfd path."
fi

echo "============================================"
echo "  Qwen3.8-Flash-Next Vision Uncensored IQ4_XS"
echo "  gfx1201 (RX 9070 XT 16 GB) native ROCm + expert pool"
echo "============================================"
echo "[+] Binary:        $SERVER_BIN"
echo "[+] ROCm libs:   $ROCM_LIB_DIR (LD_LIBRARY_PATH; no GFX override — native gfx1201)"
echo "[+] Model:         $MODEL ($SHARD_COUNT shards)"
echo "[+] Vision:        mmproj=$MMPROJ"
echo "[+] Context:       $CONTEXT ($((CONTEXT / 1024))K tokens)"
echo "[+] Target KV:     K=$CACHE_TYPE_K / V=$CACHE_TYPE_V"
echo "[+] Flash attn:    $FLASH_ATTN"
echo "[+] Expert cache:  $EXPERT_CACHE (0 = disabled; pool cap ${POOL_CAP_MIB:-none} MiB, rail ${POOL_RAIL_MIB} MiB)"
echo "[i] Admission = min(EXPERT_CACHE, slots that fit above the rail) — MI50 numbers do not transfer (1 slot = 128 MiB here)"
echo "[+] MoE placement: $((48 - N_CPU_MOE)) layers GPU / $N_CPU_MOE layers CPU"
echo "[+] Batch/Ubatch:  $BATCH / $UBATCH, threads $THREADS / $THREADS_BATCH"
echo "[+] Spec decode:   $SPEC_TYPE"
echo "[+] Port:          $PORT"
echo "[+] Log/pid:       $LOG_FILE / $PID_FILE"
echo "============================================"
echo "[i] Output is not bit-identical to cache-off; throughput gain depends on"
echo "[i] routing locality. Verify the 'expert pool status=' marker after startup."
echo "[i] All quoted byte/t/s numbers are MI50-era and indicative only here."
echo "============================================"

# =========================
# SERVER ARGS
# =========================
SERVER_ARGS=(
  -m "$MODEL_PATH"
  --host 0.0.0.0
  --port "$PORT"
  -np "$PARALLEL"
  -ngl "$NGL"
  -c "$CONTEXT"
  -b "$BATCH"
  -ub "$UBATCH"
  --threads "$THREADS"
  --threads-batch "$THREADS_BATCH"
  --cache-ram "$CACHE_RAM"
  --ctx-checkpoints "$CTX_CHECKPOINTS"
  --jinja
  --mmproj "$MMPROJ_PATH"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --flash-attn "$FLASH_ATTN"
  --load-mode mmap
  --lazy-mode on
  --metrics
  --moe-expert-cache "$EXPERT_CACHE"
)
if [[ -n "$LOG_LEVEL" ]]; then
  SERVER_ARGS+=(-lv "$LOG_LEVEL")
fi

if [[ -n "$CHAT_TEMPLATE" ]]; then
  SERVER_ARGS+=(--chat-template-file "$CHAT_TEMPLATE")
fi

if [[ -n "$N_CPU_MOE" ]]; then
  SERVER_ARGS+=(--n-cpu-moe "$N_CPU_MOE")
elif [[ "$CPU_MOE" == "1" ]]; then
  SERVER_ARGS+=(--cpu-moe)
fi

if [[ "$SPEC_TYPE" != "none" ]]; then
  echo "[WARN] SPEC_TYPE=$SPEC_TYPE: the pool is force-disabled for draft/MTP types."
  SERVER_ARGS+=(--spec-type "$SPEC_TYPE")
  SERVER_ARGS+=(--spec-draft-n-max "$SPEC_DRAFT_N_MAX")
  SERVER_ARGS+=(--spec-ngram-map-k-size-n "$SPEC_NGRAM_MAP_K_SIZE_N")
  SERVER_ARGS+=(--spec-ngram-map-k-size-m "$SPEC_NGRAM_MAP_K_SIZE_M")
fi

# =========================
# LIFECYCLE (own instance only)
# =========================
mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[+] Stopping existing instance from $PID_FILE (pid $(cat "$PID_FILE"))..."
  kill "$(cat "$PID_FILE")"
  for ((i = 1; i <= 30; i++)); do
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break
    sleep 1
  done
fi
rm -f "$PID_FILE"

# Authoritative bind test: it fails exactly when another listener holds the
# port (SO_REUSEADDR, like llama-server sets, still forbids two listeners on the
# same addr:port — only SO_REUSEPORT would share it). This avoids parsing `ss`
# output entirely; note `ss -ltn` prints its header even when nothing matches,
# which is what made an earlier `ss ... | grep -q .` check fire on every port.
# Prints exactly one of: free | in-use | cannot-bind:<reason>
port_state() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$PORT" <<'PY' 2>/dev/null
import errno, socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError as e:
    if e.errno == errno.EADDRINUSE:
        print("in-use")
    else:
        print(f"cannot-bind:{e.strerror}")
else:
    print("free")
finally:
    s.close()
PY
    return
  fi
  local out
  out="$(ss -ltnH "sport = :$PORT" 2>/dev/null)" \
    || out="$(ss -ltn "sport = :$PORT" 2>/dev/null | tail -n +2)"
  if [[ -n "$out" ]]; then echo "in-use"; else echo "free"; fi
}

PORT_STATE="$(port_state)"
case "$PORT_STATE" in
  free)
    ;;
  in-use)
    echo "[ERROR] Port $PORT is already in use by another process:"
    if command -v ss >/dev/null 2>&1; then
      ss -ltnpH "sport = :$PORT" 2>/dev/null \
        || ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2 \
        || true
    fi
    echo "[ERROR] Choose another port, e.g.: PORT=8887 $0"
    exit 1
    ;;
  *)
    echo "[ERROR] Cannot bind port $PORT ($PORT_STATE)."
    echo "[ERROR] Ports below 1024 need privileges (net.ipv4.ip_unprivileged_port_start=$(sysctl -n net.ipv4.ip_unprivileged_port_start 2>/dev/null || echo '?'))."
    echo "[ERROR] Pick an unprivileged port, e.g.: PORT=8887 $0"
    exit 1
    ;;
esac

# =========================
# RUN
# =========================
echo ""
echo "[+] Starting llama.cpp server (expert pool, EXPERT_CACHE=$EXPERT_CACHE)..."

# env: pass the cap only when set — an empty value is parsed as invalid and
# disables the pool. No HSA_OVERRIDE for gfx1201; ROCm libs via LD_LIBRARY_PATH.
ENV_ARGS=(env
  "LD_LIBRARY_PATH=$ROCM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "GGML_MOE_POOL_STATS=1"
  "LLAMA_MOE_POOL_RAIL_MIB=$POOL_RAIL_MIB"
)
if [[ -n "$POOL_CAP_MIB" ]]; then
  ENV_ARGS+=("LLAMA_MOE_POOL_CAP_MIB=$POOL_CAP_MIB")
fi
if [[ -n "$POOL_PROFILE" ]]; then
  ENV_ARGS+=("LLAMA_MOE_POOL_PROFILE=$POOL_PROFILE")
fi

"${ENV_ARGS[@]}" "$SERVER_BIN" "${SERVER_ARGS[@]}" \
  </dev/null >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "[+] Server pid $SERVER_PID -> $PID_FILE (log: $LOG_FILE)"

# =========================
# HEALTH WAIT
# =========================
echo "[+] Waiting for server to become ready (model load takes several minutes)..."
HEALTH_RETRIES=180
HEALTH_DELAY=5
for ((i = 1; i <= HEALTH_RETRIES; i++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ERROR] Server process stopped unexpectedly — last log lines:"
    tail -n 40 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[+] Server is healthy after ~$((i * HEALTH_DELAY))s"
    break
  fi
  if [[ $i -eq $HEALTH_RETRIES ]]; then
    echo "[WARN] Not healthy after $((HEALTH_RETRIES * HEALTH_DELAY))s — check logs"
  fi
  sleep "$HEALTH_DELAY"
done

# =========================
# EXPERT-POOL EVIDENCE
# =========================
LOG_TAIL="$(tail -n 4000 "$LOG_FILE" 2>/dev/null || true)"

STATUS_LINE="$(printf '%s\n' "$LOG_TAIL" | grep -E 'expert pool status=(enabled|disabled)' | tail -n 1 || true)"
if [[ -n "$STATUS_LINE" ]]; then
  echo "$STATUS_LINE"
  if [[ "$STATUS_LINE" == *"status=disabled"* ]]; then
    echo "[WARN] Pool disabled — this run is cache-off, not the pooled path."
  else
    echo "[i] Expect reason=admitted; limited_by=slots means EXPERT_CACHE was fully admitted, rail means the rail bound it."
  fi
else
  echo "[UNKNOWN] expert-pool startup status marker missing"
  echo "[UNKNOWN] Do not treat this run as a validated pooled run."
fi

FIRST_USE_LINE="$(printf '%s\n' "$LOG_TAIL" | grep -E 'expert pool first-use=(active|bypassed)' | tail -n 1 || true)"
if [[ -n "$FIRST_USE_LINE" ]]; then
  echo "$FIRST_USE_LINE"
else
  echo "[UNKNOWN] expert-pool first-use marker missing (no request served yet?)"
fi

# =========================
# SUMMARY
# =========================
echo ""
echo "============================================"
echo "  Ready on port $PORT"
echo "============================================"
echo "  OpenAI API:  http://localhost:$PORT/v1"
echo "  Web chat:    http://localhost:$PORT"
echo "  Server pid:  $SERVER_PID (file: $PID_FILE)"
echo ""
echo "Evidence:"
echo "  grep 'expert pool' $LOG_FILE"
echo "  grep 'expert pool runtime' $LOG_FILE   # after stop"
echo "  rocm-smi --showmemuse    # expect >1 GiB free with the pool enabled"
echo ""
echo "Stop (prints the hit/miss/eviction aggregate):"
echo "  kill \$(cat $PID_FILE)"
echo ""
