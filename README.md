# gfx906-16gb-expert-pool

An unofficial fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds a persistent
expert pool (expert cache) for MoE models whose experts are offloaded with `--n-cpu-moe`, and makes
that cache actually usable on a 16 GB gfx906 card (Radeon Pro VII / MI50 / MI60).

Experimental. Locally validated on one card and one model. Not affiliated with, endorsed by, or
submitted to the ggml-org project.

## Why this fork exists

- Upstream llama.cpp has **no expert cache**. `--n-cpu-moe N` moves routed experts to the CPU, but
  every token then re-copies the experts it needs over PCIe. At 128K context on a 16 GB card that
  costs roughly 40% of decode throughput.
- A persistent expert pool exists in a separate fork (see Credits), but its admission budget reserves
  the **full configured context KV** plus a hard 3 GiB cap. On a 16 GB device at 128K that admits
  **0 of 144** offloaded expert tensors, so the feature silently does nothing.
- This fork keeps the pool mechanism and replaces the budget with a rail-based ledger, adds a
  weighted per-layer planner, adds telemetry, and makes the pool fail closed when speculative
  draft/MTP decoding is selected.

The pool is not the win by itself. The admission fix is what turns "0 tensors fit" into 66 slots.

## Credits

- Expert pool implementation (`--moe-expert-cache`, pool allocator, LRU slots, graph and expert-ID
  remap): originally written by **zhanghewei** in
  [`memoriaru/llama.cpp`](https://github.com/memoriaru/llama.cpp) branch `moe-expert-pool` at
  `555d1ec9daa4564cb06adb39c469acfdcd1d1e93`. Carried here as the first commit, authorship
  preserved. See PROVENANCE.md.
- llama.cpp and ggml: the ggml authors, MIT. Includes the upstream `qwen4exp` architecture.
- Validated model: [AtomicChat/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF),
  `AD-3.84bpw-IQ4_XS-M64`.
- ROCm/gfx906 build base: [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906).

## Build (gfx906 / ROCm)

`docker/Dockerfile` builds against a pinned ROCm/gfx906 base image and compiles the fork with
`-DGGML_HIP=ON -DGPU_TARGETS=gfx906`:

```sh
docker build -f docker/Dockerfile -t local/llama.cpp-gfx906:expert-pool .
```

The image records the source commit and a source-tree digest in `/etc/llama.cpp-provenance`, so a
built image can be tied back to an exact tree.

## Build and run (gfx1201 / ROCm)

Build natively against host ROCm, no Docker image. `HSA_OVERRIDE_GFX_VERSION` is not needed, gfx1201
is native in ROCm 10:

```sh
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1201
cmake --build build -j"$(nproc)"
```

The built `build/bin/libggml-hip.so` carries a gfx1201 code object. If the ROCm libraries are not in
the loader cache, start the server with `LD_LIBRARY_PATH=/opt/rocm/lib`, otherwise it exits with
`libhipblas.so.3: cannot open shared object file`.

`scripts/00-update-and-build.sh` fetches the remotes, fast-forwards the branch and rebuilds with the
flags above. `scripts/01-qwen-flash-131k-gfx1201-launch-script.sh` is the native counterpart of
`scripts/run-example.sh`.

### Pool sizing on a 16 GB card with another model

The gfx906 numbers above do not transfer to a different model. Measured with
`Qwen3.8-Flash-Next-Uncensored-IQ4_XS` at 128K, Q8_0 K/V, `--n-cpu-moe 48`:

- One slot is 128 MiB (8,434,851,840 B for 66 slots) against 76.5 MiB on the validated model, so the
  gfx906 value of 66 slots needs about 2.9 GiB more VRAM here.
- 66 slots left about 3.08 GiB free and context creation failed with
  `allocating 3952.28 MiB on device 0: cudaMalloc failed: out of memory`. That request is the pp
  compute buffer, inflated by `-ub 2048`. The rail reserves that space, so ubatch 2048 needs a rail
  larger than 2048 MiB.
- Planner ledger from that run: `ceiling=14,948,499,456` (card total minus rail 2048 MiB) and
  `cap=9,596,764,160`, so 5.35 GB of weights were already on the device before the pool was sized.

Start at 40 slots (about 5.0 GiB) with `-ub 512`, and raise it only while
`rocm-smi --showmeminfo vram` shows headroom. Run once with `EXPERT_CACHE=0` to measure the non-pool
footprint before choosing a value.

This model uses hybrid attention (`full_attention_interval=4` plus per-layer compress ratios), so its
128K KV cache is much smaller than a full 48-layer cache. Measure it instead of estimating it.

Status: build and pool admission are verified on gfx1201; a full end-to-end 128K run is not yet
confirmed on that card.

## Usage

```sh
llama-server -m <model.gguf> --n-cpu-moe 48 -ngl 99 -c 131072 \
  -ctk q8_0 -ctv q8_0 -fa on --moe-expert-cache 66
```

- `--moe-expert-cache N` / `-mec N`: requested pool slots **per offloaded expert weight tensor**
  (not a global count, not tokens). Each tensor gets its own pool. `0` disables the pool.
- `-ngl`/`--n-cpu-moe` decide how many layers are offloaded. The pool only applies to offloaded MoE
  tensors.
- Decode-shaped operations (few tokens per step) use the pool. Large prefill batches bypass it and
  run the stock host-copy path, which is intentional.

Environment knobs:

| Variable | Meaning |
| --- | --- |
| `LLAMA_MOE_POOL_RAIL_MIB` | VRAM held back from the pool for compute buffers and later allocations. Default 2879 MiB, hard floor 1024 MiB. Lower it to admit more slots. |
| `LLAMA_MOE_POOL_CAP_MIB` | Optional absolute cap on total pool bytes. Unset means no cap. `0` is rejected. |
| `LLAMA_MOE_POOL_PROFILE` | Optional text file with one `blk.<layer_index> <weight>` line per layer for weighted slot allocation. Missing or malformed entries fall back to weight 1.0. |
| `GGML_MOE_POOL_STATS` | Set to any value for per-pool hit/miss detail on top of the aggregate reports. |

Telemetry, one line per event type, visible at the default log level:

```
expert pool status=enabled reason=admitted requested_slots=66 actual_slots=66 pool_count=144 bytes=5292195840 cap=7793213440 ceiling=15015608320 limited_by=slots alloc=uniform profile=none slots_min=66 slots_med=66 slots_max=66 total_slots=9504
expert pool first-use=active reason=distinct-experts-within-slots
expert pool runtime reason=shutdown hits=... misses=... evictions=... hit_rate=... copy_bytes=...
```

`limited_by` says what actually constrained admission (`slots`, `rail` or `cap`), and `copy_bytes` is
the number of bytes copied on misses. Watch `copy_bytes`, not the hit-rate percentage: a slot moved
to a cheaper tensor raises the hit count while increasing bytes moved.

## Measured results

MI50 16 GB, gfx906/ROCm, `Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64`, 128K context, Q8_0 K/V,
`--n-cpu-moe 48`, flash attention on, speculative decoding off, 18 threads, batch 2048 / ubatch 512.

| configuration | decode | VRAM (steady) | notes |
| --- | --- | --- | --- |
| `--moe-expert-cache 0` | 11.76 t/s | 10.17 GiB | stock CPU MoE path |
| `--moe-expert-cache 80` (actual 40 slots) | **16.39 t/s** | ~13.5 GiB | 144 pools, 61.8% hit rate |
| `--moe-expert-cache 66` | 16.90 / 17.60 t/s cold / warm | 15.24 GiB peak | shipping profile, ~69% hits |

Supporting measurements, with their limits:

- VRAM ceiling: 66 slots peaks at 14.93 GiB on short generations, 68 slots at 15.4 to 15.6 GiB, and
  72 slots was rejected (over 15.8 GiB, 147 MB free, visible microstutter). VRAM peak depends on
  generation length, so validate with your own workload.
- Cold cache is slower: the first requests run at roughly 6 t/s while the pool fills, reaching the
  warm figure after the working set is resident.
- The benefit is routing-locality dependent. A batch-1/ubatch-1 churn workload at 46.9% hits ran
  1.8x slower than cache-off, because every decode token drove 144 synchronous pool updates.
- A fixed-token perplexity comparison measured 1.0987 (cache off) versus 1.0480 (cache on). **This is
  not a quality improvement claim.** The two configurations run MoE arithmetic on different backends,
  and the sample was far too small to bound quality; treat the numbers as evidence of no measured
  degradation only.
- Enabling the pool is **not bit-identical** to cache-off: it moves MoE compute from the CPU backend
  to the accelerator, and those backends already differ in arithmetic for every quant type. Pool
  parity is bit-exact against the stock host-copy path when compared on the same backend.

Donor-reported numbers are not reproduced here and should not be read as MI50 results: the donor
measured +84% at 64 slots on an RTX 4090, a third party measured 2.8x slower at 16 slots and +54% at
160 slots on a PCIe 3.0 system, and an RX 9070 Vulkan report recorded regressions at 16/32/64 slots.

What this does not show: no contexts above 128K (admission can fall to zero), no vision workloads, no
MTP/speculative decoding, one model, one card.

## Updating from upstream

```sh
git fetch upstream
git rebase upstream/master
```

Conflicts are expected in `README.md` (this file replaces upstream's), `common/arg.cpp`,
`src/llama-context.cpp`, `src/llama-graph.cpp` and `ggml/src/ggml-backend.cpp`. After rebasing,
rebuild for gfx906 and re-run `test-expert-pool`.

## License

MIT, same as upstream llama.cpp.
