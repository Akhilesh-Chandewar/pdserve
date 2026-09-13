// Package engine implements the serving engines: a colocated baseline that
// interleaves prefill+decode on shared instances (causing cross-phase
// contention), and a PD-disaggregated engine with dedicated prefill/decode
// workers streaming KV cache between them.
package engine

import (
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/kvcache"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/scheduler"
)

// Config tunes the engines.
type Config struct {
	// NumGPUs is the total instance count.
	NumGPUs int
	// BandwidthGBps is HBM bandwidth per GPU (roofline model).
	BandwidthGBps float64
	// ComputeTFLOPs is peak compute per GPU.
	ComputeTFLOPs float64

	// KVCapacityBlocks per instance.
	KVCapacityBlocks int
	// KVBlockSize tokens per block.
	KVBlockSize int

	// MaxBatchDecode caps sequences per decode batch (1 = no batching).
	MaxBatchDecode int
	// ChunkedPrefill enables splitting long prompts into chunks (colocated).
	ChunkedPrefill bool
	// PrefillChunkTokens is the chunk size used when ChunkedPrefill is on.
	PrefillChunkTokens int

	// PrefillRatio is the static split for DisaggregatedEngine (0..1).
	PrefillRatio float64
	// KVConnector selects the KV transfer layer ("memory" or "tcp").
	KVConnector string
	// KVTCPBandwidthGBps is fabric bandwidth for the TCP connector.
	KVTCPBandwidthGBps float64

	// Load describes the arrival process.
	Load LoadSpec
	// SLO used by elastic policy.
	SLO scheduler.SLO
}

// DefaultConfig returns a small-cluster baseline (8 GPUs, A100-class).
func DefaultConfig() Config {
	return Config{
		NumGPUs:            8,
		BandwidthGBps:      2039, // A100 80GB HBM2e
		ComputeTFLOPs:      312,
		KVCapacityBlocks:   4096,
		KVBlockSize:        16,
		MaxBatchDecode:     1,
		ChunkedPrefill:     true,
		PrefillChunkTokens: 512,
		PrefillRatio:       0.25,
		KVConnector:        "memory",
		KVTCPBandwidthGBps: 50,
		Load:               DefaultLoad(),
		SLO:                scheduler.DefaultSLO(),
	}
}

// ModelRunner abstracts model execution. SimRunner implements it with a
// roofline model; production runners can plug in real GPU execution.
type ModelRunner interface {
	// PrefillUS returns microseconds to process n prompt tokens.
	PrefillUS(n int) float64
	// DecodeUS returns microseconds to emit one token for one sequence.
	DecodeUS(ctxTokens int) float64
}

// SimRunner is a roofline-model runner: prefill is compute-bound
// (time ~ 2*params*tokens / MFU*compute), decode is bandwidth-bound
// (time ~ ctx * kv_bytes_per_token / bandwidth).
type SimRunner struct {
	BandwidthGBps float64
	ComputeTFLOPs float64
	// ModelParams is the parameter count (default 7B).
	ModelParams float64
	// MFU is model flops utilization for prefill.
	MFU float64
}

// NewSimRunner builds a SimRunner.
func NewSimRunner(bandwidthGBps, computeTFLOPs float64) *SimRunner {
	return &SimRunner{BandwidthGBps: bandwidthGBps, ComputeTFLOPs: computeTFLOPs, ModelParams: 7e9, MFU: 0.35}
}

// PrefillUS implements ModelRunner (compute-bound roofline).
func (s *SimRunner) PrefillUS(n int) float64 {
	if n <= 0 {
		return 0
	}
	flops := 2 * s.ModelParams * float64(n)
	return flops / (s.ComputeTFLOPs * s.MFU * 1e12) * 1e6
}

// DecodeUS implements ModelRunner (bandwidth-bound roofline).
func (s *SimRunner) DecodeUS(ctxTokens int) float64 {
	if ctxTokens <= 0 {
		ctxTokens = 1
	}
	const kvBytesPerToken = 2 * 32 * 128 * 2 // K+V, 32 heads, head_dim 128, fp16
	bytes := float64(ctxTokens) * kvBytesPerToken
	return bytes / (s.BandwidthGBps * 1e9) * 1e6
}

// Engine is implemented by ColocatedEngine, DisaggregatedEngine and
// ElasticEngine.
type Engine interface {
	// Start boots workers and the arrival generator.
	Start()
	// Stop drains and shuts down.
	Stop()
	// Registry exposes metrics.
	Registry() *metrics.Registry
	// Name identifies the engine in benchmark output.
	Name() string
	// Queue returns the admission queue.
	Queue() *scheduler.Queue
	// PoolSizes reports (prefill, decode) instance counts.
	PoolSizes() (int, int)
}

func usDur(us float64) time.Duration {
	if us <= 0 {
		return 0
	}
	return time.Duration(us * float64(time.Microsecond))
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// finishRequest records metrics for a completed request.
func finishRequest(reg *metrics.Registry, req *scheduler.Request, pool *kvcache.Pool) {
	req.FinishAt = time.Now()
	tokens := maxInt(1, req.Generated)
	tpot := float64(req.FinishAt.Sub(req.FirstTokenAt)) / 1e3 / float64(tokens)
	reg.RecordRequest(req.TTFTUS(), tpot, req.E2EUS(), int64(req.Generated), false)
	pool.Release(req.ID)
}

func allocateKV(pool *kvcache.Pool, blockSize int, req *scheduler.Request) ([]*kvcache.Block, bool) {
	nblocks := (req.PromptTokens + blockSize - 1) / blockSize
	blocks, err := pool.Allocate(req.ID, nblocks)
	if err != nil {
		return nil, false
	}
	pool.Acquire(req.ID)
	return blocks, true
}
