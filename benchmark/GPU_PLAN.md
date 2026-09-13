# pdserve Real-GPU Benchmark Plan — A10 Edition

**Goal:** turn pdserve's simulation numbers (133× TPOT inflation, TTFT 4.1s→0.28s) into
published, reproducible real-GPU benchmarks — the artifact that makes the inference-claim
irrefutable. Budget: **~$10–15 of GPU time**. Effort: **1 focused day + 1 write-up evening**.

---

## 1. What to rent

| Option | Spec | Cost | Use |
|---|---|---|---|
| **Primary** | 2× NVIDIA A10 24GB (RunPod / Lambda / Vast) | ~$1.1–1.3/hr total | Colocated baseline + true PD disaggregation (1 prefill + 1 decode pool) |
| Fallback | 1× A10 | ~$0.55/hr | Single-GPU experiments only (skip the PD-split scenario) |

**Reserve 6 hours.** Runs are short; most time goes to setup and re-runs.

Model: **Qwen2.5-7B-Instruct** (or Llama-3.1-8B-Instruct) — small enough for A10, respected
enough that numbers get read. Two precisions: FP16 and AWQ-INT4.

## 2. Engines & versions (pin everything)

- **vLLM** (latest stable, record exact version) — `vllm serve`
- **SGLang** (latest stable, record exact version) — `sglang.launch_server`
- Load generator: **one harness for both engines** — `guidellm` (or vLLM's `bench serve`
  against both OpenAI-compatible endpoints). Same client, same seeds, same Poisson seed.
- Docker images pinned by digest; all configs exported into `configs/`.

## 3. Scenarios (the experiment matrix)

### A. Colocated baseline — reproduce the pathology
- Single A10, all traffic on one engine.
- **Chunked prefill ON vs OFF** (`--enable-chunked-prefill`).
- Load: Poisson, RPS ∈ {2, 4, 8} + **6× burst at t=20s** (mirror pdserve's LoadSpec).
- Input lengths: {256, 1024, 4096} tokens; output: 256 tokens.
- **Expected:** TPOT p99 spikes during bursts (the 133× story, now real).

### B. Prefix caching — quantify the win
- SGLang RadixAttention ON vs vLLM `--enable-prefix-caching` ON vs OFF.
- Workload with shared system prompt (~2K tokens) + varied user turns.
- Measure: cache hit rate, TTFT delta, tok/s delta.
- **Expected:** TTFT improvement grows with shared-prefix ratio; find the crossover where
  cache bookkeeping costs more than it saves (the "when prefix caching loses money" chart).

### C. PD disaggregation — the headline (needs 2× A10)
- vLLM PD-disagg (or SGLang PD disaggregation) with a dedicated prefill worker and decode
  worker, KV transferred between them.
- Same Poisson+burst load as Scenario A.
- **Expected:** TPOT stays flat during bursts; TTFT p99 drops vs colocated. This is the
  chart that mirrors pdserve's simulation output 1:1.

### D. Quantization sweep (filler while GPUs idle)
- FP16 vs AWQ-INT4 across scenarios A/B: throughput, TTFT/TPOT, and **cost per 1M tokens**.

## 4. Metrics to capture (per run, per engine)

- TTFT p50/p99 · TPOT p50/p99 · ITL p99 · E2E p99 (ms)
- Throughput: total tok/s, output tok/s
- KV cache hit rate (where exposed)
- GPU: utilization, memory headroom (`nvidia-smi dmon` logging)
- **Cost per 1M tokens** at the rented hourly rate

Raw output: CSV per run in `raw/` (never delete raw data; charts are derived).

## 5. Charts to publish (README, in this order)

1. **The money chart:** TPOT p99 **over time** during a burst window — colocated (chunked
   prefill) vs PD-disaggregated. Two lines, burst marked at t=20s. This is pdserve's sim
   chart made real.
2. TTFT p99 vs offered load (RPS) — colocated vs PD vs prefix-cached.
3. Throughput vs concurrency — FP16 vs AWQ, vLLM vs SGLang.
4. Prefix-cache hit rate vs shared-prefix ratio, with TTFT savings overlaid.
5. Cost per 1M tokens vs latency SLO (Pareto frontier, both engines, both precisions).

Style rules: matplotlib defaults, log-y where p99s diverge, every chart states engine
versions + GPU + precision in the caption. `charts/*.png` + the script that drew them.

## 6. Reproducibility (what makes it senior-signal)

- `runbook.md`: exact commands, order, and wall-clock timing for every run.
- `configs/`: one file per engine per scenario — a stranger must be able to re-run it.
- `results.md`: raw tables before any chart smoothing.
- Env pinning: CUDA version, torch version, engine versions, docker digests, seeds.

## 7. Publish sequence (the comp part)

1. Push `benchmark/` into the pdserve repo (or standalone `pdserve-bench` repo if cleaner).
2. README top: one-paragraph TL;DR with 3 headline numbers.
3. Blog post (Medium — you already have the account):
   **"I simulated LLM serving physics, then rented a GPU to check — here's what matched"**
   - pdserve's roofline model predictions vs measured reality
   - where the sim was right (TPOT flatness under PD, ITL flatness)
   - where it was wrong (say it honestly — that's the credibility)
4. Cross-link: profile README table gets a "Real-GPU benchmarks" column; CV pdserve bullet
   upgrades from simulation numbers to measured numbers; LinkedIn post with chart #1.

## 8. Fallback ladder

- No budget: 1× A10 for 3 hours (~$2) covers Scenarios A+B+D — still publishable.
- No budget at all: vLLM/SGLang on a free Colab T4 is worse but directionally valid —
  label it as such.
- GPU unavailable: publish the benchmark *harness* + runbook, marked "results pending" —
  still signals, just weaker.

## 9. Success criteria

- [ ] Scenario A reproduces TPOT p99 inflation ≥3× under burst
- [ ] Scenario C shows TPOT p99 within 2× of idle baseline during burst
- [ ] ≥4 charts + raw CSVs + pinned configs public
- [ ] Blog post live with honest sim-vs-real comparison
- [ ] CV + profile README updated with measured numbers
