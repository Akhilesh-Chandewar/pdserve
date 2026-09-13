# pdserve — Benchmarks

Reproducible benchmark suite for the three serving strategies pdserve implements.
Everything here runs **without GPUs** — it's the roofline simulator shipped in this repo
(`cmd/pdserve`), driven by a fixed scenario matrix.

> **Honesty note:** these are *simulator* numbers from a first-principles roofline model
> (compute-bound prefill, bandwidth-bound decode). They reproduce the *qualitative*
> pathology of real systems (chunked-prefill decode stalls, ITL/TPOT flatness under
> disaggregation) — not exact hardware latencies. The real-GPU validation plan
> (vLLM/SGLang on rented A10s + free Colab T4) lives in [`GPU_PLAN.md`](GPU_PLAN.md).

## Headline result (rps=8, 6× burst at t=6s, 8 GPU instances, pre=2/dec=6)

| engine | done | ttft p99 (s) | tpot p99 (ms) | e2e p99 (s) | tok/s |
|---|---:|---:|---:|---:|---:|
| colocated | 64 | 4.096 | **66.5** | 21.9 | 269 |
| pd-disaggregated | 178 | 1.571 | **0.5** | 1.6 | 1125 |
| elastic (disagg + SLO flips) | 178 | 1.558 | **0.5** | 1.6 | 1125 |

- Colocated serving shows the classic pathology: every arrival burst re-prioritizes
  prefill chunks over in-flight decode steps → **TPOT p99 inflates 133×** (0.5ms → 66.5ms)
  and **~60% of requests don't complete** in the window (64 vs 178).
- PD disaggregation holds decode at **0.5ms TPOT p99 flat** across every burst level tested.
- The elastic policy matches static disaggregation here (flips=0 at this pool split) and
  **beats it when over-provisioned**: at burst=10×, elastic TTFT p99 = 3.35s vs 4.10s static —
  the monitor gave spare prefill capacity back to decode.

## Charts

### 1 — Decode stalls under burst (the money chart)
![TPOT p99 vs burst](charts/1_tpot_p99_vs_burst.png)

### 2 — TTFT p99 under burst
![TTFT p99 vs burst](charts/2_ttft_p99_vs_burst.png)

### 3 — Throughput collapse (colocated loses ~55%)
![Throughput vs burst](charts/3_throughput_vs_burst.png)

### 4 — Completed requests (disagg serves ~4× more under burst)
![Completed vs burst](charts/4_completed_vs_burst.png)

## Full results matrix

`results/sim_results.csv` — burst sensitivity at rps=8 (burst × {1,3,6,10}) and load sweep
at burst=6× (rps × {2,4,8}). All runs: 8 GPU instances, prefill=2/decode=6 pools,
KV connector = zero-copy memory, 12s arrival window, Poisson arrivals.

## Reproduce

```bash
bash benchmark/sweep.sh     # runs the matrix, writes results/sim_results.csv (~3 min)
python3 benchmark/plots.py  # regenerates charts/ from the CSV
```

Requirements: Go 1.21+ (any platform), Python 3 + matplotlib for charts.

## Scenario definitions

| Parameter | Value |
|---|---|
| GPU instances | 8 |
| Pool split (disagg) | prefill 2 / decode 6 |
| Arrivals | Poisson, rps ∈ {2,4,8} |
| Burst | factor ∈ {1,3,6,10} at t=6s for 3s |
| Prompt tokens | mean 512 |
| Output tokens | mean 128 |
| Model class | 7B @ 35% MFU, 312 TFLOPs peak, HBM 900GB/s (roofline constants) |
