# pdserve

**Prefill-Decode (PD) disaggregation for LLM serving, in Go** — with SLO-aware
elastic role flipping and pluggable KV-cache connectors.

In colocated LLM serving, prefill (compute-heavy) and decode
(memory-bandwidth-heavy) share GPU instances. New arrivals' prefill chunks
stall active token generation, inflating Time-Per-Output-Token (TPOT) by 2x–30x
and forcing operators to over-provision GPUs to meet latency SLOs.

`pdserve` implements the three known mitigations as a runnable library:

| Concept | Where | What it does |
|---|---|---|
| **PD Disaggregation** (DistServe/SGLang/vLLM-style) | `pkg/engine` | Dedicated prefill and decode pools; KV cache streamed between them |
| **KV-Cache Connectors** (Mooncake/NIXL-style transport layer) | `pkg/kvcache` | Pluggable `Connector`: zero-copy in-process or TCP (bandwidth-modeled) transfer |
| **Dynamic & Elastic Disaggregation** (xLLM/LAPS-style) | `pkg/elastic` | SLO-aware monitor flips stateless GPU instances between prefill/decode at runtime — no weight reload |

## Demo results (8 GPUs, 4 req/s, 6x burst @ t=20s)

```
engine                          done  ttft_p99(s) tpot_p99(ms)  e2e_p99(s)   tok/s
colocated                        106       4.096        66.5       23.10     300.7
disagg(memory, pre=2/dec=6)      214       0.282         0.5        0.32     718.9
elastic(pre=2/dec=6, flips=1)    214       0.281         0.5        0.32     718.8
```

Colocated serving shows the classic pathology: TPOT p99 inflated **133x** and
half the throughput, because every arrival burst re-prioritizes prefill chunks
over in-flight decode steps.

## Quick start

```bash
go run ./cmd/pdserve                       # full comparison, defaults
go run ./cmd/pdserve -h                    # all flags

# Larger burst, TCP KV transport:
go run ./cmd/pdserve -rps 8 -burst-factor 10 -kv tcp

# Single engine:
go run ./cmd/pdserve -mode elastic -dur 60
```

## Library usage

```go
cfg := engine.DefaultConfig()
cfg.NumGPUs = 16
cfg.PrefillRatio = 0.25
cfg.KVConnector = "tcp"       // RDMA-class fabric modeled at cfg.KVTCPBandwidthGBps
cfg.Load.RPS = 12
cfg.Load.BurstAtSec = []float64{30}
cfg.SLO = scheduler.SLO{TTFTTargetUS: 2_000_000, TPOTTargetUS: 100_000}

// Static PD disaggregation:
eng, _ := engine.NewDisaggregated(cfg)
eng.Start()
time.Sleep(60 * time.Second)
eng.Stop()
fmt.Printf("%+v\n", eng.Registry().Snapshot())

// Elastic disaggregation (SLO-aware role flips):
eel, _ := elastic.NewElastic(cfg)
eel.Start()
// ... monitor eel.Flips() to observe role transitions
eel.Stop()
```

## Architecture

```
pkg/
  types/      GPUInstance, Phase, Role, InstanceState
  metrics/    latency histograms, TPOT/TTFT/ITL/E2E registry, windowed snapshots
  kvcache/    paged KV pool + Connector interface (memory | TCP), transfer-cost model
  scheduler/  Request model, FIFO admission queue, SLO spec
  engine/     roofline ModelRunner, LoadSpec (Poisson+bursts),
              ColocatedEngine | DisaggregatedEngine (+ ResizePools primitive)
  elastic/    pure SLO Policy (hysteresis, cooldown, floors) + ElasticEngine monitor
cmd/pdserve/  benchmark CLI comparing all three strategies
```

### The roofline model

`SimRunner` models the two phases with first-principles physics:

- **Prefill** (compute-bound): `t = 2 * params * tokens / (MFU * peakFLOPs)`
  — a 512-token chunk on a 7B model at 35% MFU of 312 TFLOPs ≈ 68ms.
- **Decode** (bandwidth-bound): `t = ctx_tokens * kv_bytes / HBM_bandwidth`
  — the KV read dominates; longer contexts stretch every decode step.

These constants reproduce the qualitative behavior of real systems (chunked
prefill stalls, ITL flatness under disaggregation) without needing GPUs.

### The elastic policy

Each `MonitorInterval` (250ms) the monitor samples a **windowed** p99 of TTFT
and TPOT plus prefill queue depth, and the policy reacts:

- TTFT p99 ≥ target **or** prefill backlog → move a decode GPU → prefill
- TPOT p99 ≥ target (and TTFT has margin, or is worse) → move a prefill GPU → decode
- Both healthy with >2.5x margin → return spare prefill capacity to decode

Flip decisions are gated by **hysteresis** (pressure must persist 2s) and a
**cooldown** (5s between flips), with hard floors on both pool sizes so neither
phase can starve. The policy is pure and deterministic — unit-tested without a
running engine.

## Testing

```bash
go test ./...          # unit tests, all packages
go vet ./... && gofmt -l .
```

## Roadmap to production

The library is structured so simulated components can be swapped for real ones:

1. `ModelRunner` — implement over a real runtime (vLLM/SGLang RPC, TensorRT-LLM)
   instead of `SimRunner`.
2. `kvcache.Connector` — implement over NIXL/Mooncake for RDMA transfers.
3. `engine.Engine` — back `DisaggregatedEngine` with networked worker pools;
   `ResizePools` becomes a drain+rebind of stateless instances.
4. `elastic.Policy` — already pure; point `Signal` at live Prometheus windows.

## License

MIT — see [LICENSE](LICENSE).
