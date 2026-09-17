# pdserve

**Prefill-Decode (PD) disaggregation for LLM serving, in Go** — with SLO-aware
elastic role flipping and pluggable KV-cache connectors, driven by a
deterministic discrete-event simulator.

In colocated LLM serving, prefill (compute-heavy) and decode
(memory-bandwidth-heavy) share GPU instances. New arrivals' prefill chunks
stall active token generation, inflating Time-Per-Output-Token (TPOT) and
forcing operators to over-provision GPUs to meet latency SLOs.

`pdserve` implements the three known mitigations as a runnable library:

| Concept | Where | What it does |
|---|---|---|
| **PD Disaggregation** (DistServe/SGLang/vLLM-style) | `pkg/engine` | Dedicated prefill and decode pools; KV cache streamed between them |
| **KV-Cache Connectors** (Mooncake/NIXL-style transport layer) | `pkg/kvcache` | Pluggable `Connector`: zero-copy in-process or bandwidth-modeled fabric transfer |
| **Dynamic & Elastic Disaggregation** (xLLM/LAPS-style) | `pkg/elastic` | SLO-aware monitor flips stateless GPU instances between prefill/decode at runtime — no weight reload |

## Demo results (simulated; 8 GPUs, 4 req/s, 6× burst @ t=20s)

All runs execute on a **discrete-event clock** (no wall-clock sleeps): a
600-second simulated benchmark finishes in milliseconds and is **bit-reproducible
for a given seed** — two runs with the same seed produce byte-identical results.

```
engine                               done  ttft_p99(s)  tpot_p99(ms)  e2e_p99(s)   tok/s
colocated                             214       0.135        <1.000       0.184    906.1
disagg(memory(zero-copy), pre=2/dec=6) 214      0.293        <1.000       0.294    905.9
elastic(pre=1/dec=7, flips=1)          214       0.293        <1.000       0.294    903.1
```

Under burst (`-rps 8 -burst-factor 6 -dur 12 -burst-at 6 -burst-len 3`):

```
engine              done  ttft_p99(s)  tpot_p99(ms)  e2e_p99(s)   tok/s
colocated            181       4.096        65.450       40.960    240.1
disagg(pre=2/dec=6)  181       1.715         0.248        1.716   1895.4
elastic(flips=1)     181       1.563         0.248        1.566   1867.1
```

**What this measures** (numerator and denominator, in one sentence): under
arrival bursts, a colocated GPU interleaves waiting prompts' prefill chunks
between decode steps, so the TPOT p99 tail inflates to the injected prefill-stall
magnitude (65ms), while a disaggregated decode pool never runs prefill and its
per-token latency stays at its steady-state value.

**What to be skeptical of:** these are *simulated* numbers, and the current
decode roofline underestimates absolute decode latency (it omits the
model-weight read term; see *Known limitations* below). The relative
colocated-vs-disaggregated stall behavior is the signal; absolute latencies are
not yet calibrated against hardware. Percentiles below the histogram's noise
floor (4× bucket width) print as `<floor` rather than fake precision.

## Quick start

```bash
go run ./cmd/pdserve                       # full comparison, defaults
go run ./cmd/pdserve -h                    # all flags

# Larger burst, TCP-bandwidth-modeled KV transport:
go run ./cmd/pdserve -rps 8 -burst-factor 10 -kv tcp

# Single engine, JSON lines output:
go run ./cmd/pdserve -mode elastic -dur 60 -json
```

Runs are deterministic and near-instant (discrete-event simulation).
Flags: `-mode {colocated,disagg,elastic,all}`, `-rps`, `-burst-factor`,
`-burst-at`, `-burst-len`, `-dur`, `-gpus`, `-kv {memory,tcp}`,
`-prefill-ratio`, `-seed`, `-json`.

## Library usage

```go
cfg := engine.DefaultConfig()
cfg.NumGPUs = 16
cfg.PrefillRatio = 0.25
cfg.KVConnector = "tcp"       // RDMA-class fabric modeled at cfg.KVTCPBandwidthGBps
cfg.Load.RPS = 12
cfg.Load.BurstAtSec = []float64{30}
cfg.SLO = scheduler.SLO{TTFTTargetUS: 2_000_000, TPOTTargetUS: 100_000}

// Deterministic simulation on the discrete-event clock:
cfg.Clk = clock.NewSim()
eng, _ := engine.NewDisaggregated(cfg)
eng.Start()
eng.Run() // drives the sim to quiescence; instant, bit-reproducible
fmt.Printf("%+v\n", eng.Registry().Snapshot())

// Wall-clock mode instead (real-time demo): leave cfg.Clk nil, then drive
// Start()/Stop() yourself on a timer.

// Elastic disaggregation (SLO-aware role flips):
eel, _ := elastic.NewElastic(cfg)
eel.Start()
eel.Run()
// ... monitor eel.Flips() to observe role transitions (drain+rebind)
eel.Stop()
```

## Benchmarks

Reproducible simulator benchmark suite with charts: **[benchmark/](benchmark/README.md)**
— colocated vs PD-disaggregated vs elastic across burst/load sweeps. Every
number there regenerates from one command (`make repro`) and is byte-identical
across machines thanks to the deterministic clock. Real-GPU validation plan
(vLLM/SGLang on A10 / free Colab T4): [`benchmark/GPU_PLAN.md`](benchmark/GPU_PLAN.md)
+ ready-to-run notebook [`benchmark/colab_t4_bench.ipynb`](benchmark/colab_t4_bench.ipynb).

## Architecture

```
pkg/
  clock/      Clock interface: Sim (deterministic discrete-event) | Real (wall-clock)
  rng/        seeded deterministic PRNG (all simulation randomness)
  types/      GPUInstance, Phase, Role, InstanceState
  metrics/    latency histograms with resolution floors, registry, windowed snapshots
  kvcache/    paged KV pool + Connector interface (memory | bandwidth-modeled TCP)
  scheduler/  Request model, FIFO admission queue, SLO spec
  engine/     event-chain engines (ColocatedEngine | DisaggregatedEngine,
              ResizePools = real drain+rebind), roofline SimRunner, LoadSpec
  elastic/    pure SLO Policy (hysteresis, cooldown, floors) + ElasticEngine monitor
cmd/pdserve/  benchmark CLI comparing all three strategies (JSON or table output)
```

Both engines are **event-chain driven**: every simulated delay is a timer
scheduled on a `clock.Clock`, never a blocking sleep. Under `clock.Sim` the run
is a deterministic discrete-event simulation; under `clock.Real` the same code
paths execute in real time — the seam where real ModelRunners will plug in.

### The elastic policy

Each `MonitorInterval` (250ms of clock time) the monitor samples a **windowed**
p99 of TTFT and TPOT plus prefill queue depth, and the policy reacts:

- TTFT p99 ≥ target **or** prefill backlog → move a decode GPU → prefill
- TPOT p99 ≥ target (and TTFT has margin, or is worse) → move a prefill GPU → decode
- Both healthy with >2.5x margin → return spare prefill capacity to decode

Flip decisions are gated by **hysteresis** (pressure must persist 2s) and a
**cooldown** (5s between flips), with hard floors on both pool sizes so neither
phase can starve. `ResizePools` performs a real **drain+rebind**: the flipped
worker's in-flight step completes before it rejoins the other pool, worker
counts are conserved, and a modeled drain latency is charged as simulated time.
The policy is pure and deterministic — unit-tested without a running engine.

## Known limitations

Stated plainly, in priority order (see `SPEC.md` for the full remediation plan):

1. **Decode roofline is incomplete (P0-2).** `SimRunner.DecodeUS` omits the
   model-weight read term and the layer count in KV bytes, so absolute decode
   latencies are far below real hardware (batch-1 TPOT for 7B fp16 on A100 is
   ~7ms, not ~5µs). Relative stall behavior between engines is unaffected.
   Fix lands with `pkg/model` (M3) and will re-baseline every published number.
2. **No continuous batching yet (P0-3).** Workers serve one request at a time;
   `MaxBatchDecode` is declared but not yet consumed. A shared `FormBatch`
   step-loop for both engines is the next milestone.
3. **Simulated, not measured.** Nothing here has been validated against a real
   GPU yet — the calibration plan is M5 in `SPEC.md`.

## Rules for claims

1. Every number is labeled **simulated** or **measured**. No exceptions.
2. No percentile is printed below the instrument's resolution (4× bucket width).
3. No single-run numbers presented as general results — the clock makes every
   run exactly reproducible, so a cited number is a *defined* number.
4. Every headline ratio names its numerator and denominator mechanism.
5. A "what this does not model" list is maintained (above) and prominent.

## Testing

```bash
go test ./...          # unit tests, all packages (determinism included)
go test -race ./...    # race-enabled (timing assertions relaxed for -race)
go vet ./... && gofmt -l .
make ci                # everything above
```

## License

MIT — see [LICENSE](LICENSE).
