#!/usr/bin/env bash
#
# Update the gfx1201 fork and rebuild llama.cpp.
#
# Fetches from the configured remotes, fast-forwards the current branch when it
# tracks one, then rebuilds with the gfx1201 HIP flags and checks the code object.
#
# Usage:
#   ./update-and-build.sh                 # fetch + fast-forward + build
#   ./update-and-build.sh --fetch-only    # update the source only
#   ./update-and-build.sh --build-only    # rebuild the current tree
#   ./update-and-build.sh --with-upstream # also fetch ggml-org/llama.cpp (large)
#   ./update-and-build.sh --rebase-upstream  # replay local commits on upstream/master
#   ./update-and-build.sh --clean            # discard build/ and rebuild from scratch
#
# Environment: REPO_DIR, GPU_TARGET, JOBS, BUILD_DIR, ROCM_LIB_DIR
#
# For a periodic update+build, schedule this script (cron or a systemd timer)
# rather than looping it here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR/gfx1201-16gb-expert-pool}"
GPU_TARGET="${GPU_TARGET:-gfx1201}"
JOBS="${JOBS:-$(nproc)}"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build}"
ROCM_LIB_DIR="${ROCM_LIB_DIR:-/opt/rocm/lib}"
BUILD_LOG="${BUILD_LOG:-$BUILD_DIR/update-build.log}"

DO_PULL=1
DO_BUILD=1
DO_UPSTREAM=0
DO_REBASE=0
DO_CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --fetch-only)      DO_BUILD=0 ;;
    --build-only)      DO_PULL=0 ;;
    --with-upstream)   DO_UPSTREAM=1 ;;
    --rebase-upstream) DO_UPSTREAM=1; DO_REBASE=1 ;;
    --clean)           DO_CLEAN=1 ;;
    -h|--help)         sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "[ERROR] Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[ERROR] Not a git checkout: $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
BEFORE="$(git rev-parse HEAD)"

echo "============================================"
echo "  gfx1201 update + build"
echo "============================================"
echo "[+] Repo:   $REPO_DIR"
echo "[+] Branch: $BRANCH @ ${BEFORE:0:9}"
echo "[+] Target: $GPU_TARGET, jobs $JOBS"
echo "============================================"

# =========================
# UPDATE
# =========================
if [[ "$DO_PULL" == "1" ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[WARN] Working tree has local changes; skipping the update step."
    echo "[WARN] Commit or stash them, then re-run."
  else
    REMOTES=()
    for r in $(git remote); do
      if [[ "$r" == "upstream" && "$DO_UPSTREAM" != "1" ]]; then
        echo "[i] Skipping remote 'upstream' (use --with-upstream to fetch it)"
        continue
      fi
      REMOTES+=("$r")
    done

    for r in "${REMOTES[@]}"; do
      echo "[+] Fetching $r ..."
      git fetch --prune "$r"
    done

    if UP="$(git rev-parse --abbrev-ref '@{u}' 2>/dev/null)"; then
      if [[ "$DO_REBASE" == "1" ]]; then
        echo "[+] Rebasing $BRANCH onto upstream/master (conflicts possible) ..."
        if ! git rebase upstream/master; then
          echo "[ERROR] Rebase stopped on conflicts."
          echo "[ERROR] Resolve them and run 'git rebase --continue', or 'git rebase --abort'."
          exit 1
        fi
      else
        read -r AHEAD BEHIND < <(git rev-list --left-right --count "HEAD...$UP")
        if [[ "$BEHIND" == "0" ]]; then
          echo "[i] Up to date with $UP (ahead $AHEAD commit(s))."
        elif [[ "$AHEAD" == "0" ]]; then
          git merge --ff-only "$UP"
        else
          echo "[WARN] $BRANCH has diverged from $UP: $AHEAD local, $BEHIND remote."
          echo "[WARN] Skipping the update and building the current tree instead."
          echo "[i] Local history was rewritten (for example by a rebase on upstream)."
          echo "[i] To reconcile: git rebase $UP, then force-push if that is intended."
        fi
      fi
      AFTER="$(git rev-parse HEAD)"
      if [[ "$BEFORE" == "$AFTER" ]]; then
        echo "[i] HEAD unchanged at ${AFTER:0:9}."
      else
        echo "[+] Updated ${BEFORE:0:9} -> ${AFTER:0:9}"
        git log --oneline "$BEFORE..$AFTER" | sed 's/^/    /'
      fi
    else
      echo "[WARN] Branch '$BRANCH' has no upstream tracking ref; fetched only."
    fi
  fi
else
  echo "[i] Update step skipped (--build-only)."
fi

# =========================
# BUILD
# =========================
if [[ "$DO_BUILD" == "1" ]]; then
  if [[ "$DO_CLEAN" == "1" ]]; then
    case "$BUILD_DIR" in
      "$REPO_DIR"/*) echo "[+] Removing $BUILD_DIR for a clean build ..."; rm -rf "$BUILD_DIR" ;;
      *) echo "[ERROR] Refusing to remove $BUILD_DIR: not inside $REPO_DIR"; exit 1 ;;
    esac
  fi
  mkdir -p "$BUILD_DIR"

  if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "[+] Configuring (HIP, $GPU_TARGET) ..."
    cmake -B "$BUILD_DIR" -DGGML_HIP=ON -DGPU_TARGETS="$GPU_TARGET" 2>&1 | tee -a "$BUILD_LOG"
  else
    echo "[+] Reusing existing configuration in $BUILD_DIR"
  fi

  echo "[+] Building with $JOBS jobs (log: $BUILD_LOG) ..."
  cmake --build "$BUILD_DIR" -j"$JOBS" 2>&1 | tee -a "$BUILD_LOG" | tail -n 3

  SERVER_BIN="$BUILD_DIR/bin/llama-server"
  if [[ ! -x "$SERVER_BIN" ]]; then
    echo "[ERROR] Build finished but $SERVER_BIN is missing."
    exit 1
  fi

  # The fatbin must carry the requested target or the runtime finds no device.
  HIP_LIB="$(ls "$BUILD_DIR"/bin/libggml-hip.so.*.* 2>/dev/null | head -n 1 || true)"
  if [[ -n "$HIP_LIB" ]] && command -v strings >/dev/null 2>&1; then
    TARGETS="$(strings -n 8 "$HIP_LIB" | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ' || true)"
    echo "[+] Code objects: ${TARGETS:-none found}"
    if [[ "$TARGETS" != *"$GPU_TARGET"* ]]; then
      echo "[WARN] $GPU_TARGET not found in $HIP_LIB; the runtime may not see the GPU."
    fi
  fi

  echo "[+] Built: $SERVER_BIN ($(date -r "$SERVER_BIN" '+%Y-%m-%d %H:%M'))"
  echo "[+] Source: $(git rev-parse --short HEAD) on $BRANCH"
  echo "[i] Start the server with ./qwen-flash-131k-gfx1201-launch-script.sh"
  echo "[i] If it is already running, that launcher stops the old instance first."
  echo "[i] After a rebase, prefer --clean: stale objects can survive a base change."
else
  echo "[i] Build step skipped (--fetch-only)."
fi
