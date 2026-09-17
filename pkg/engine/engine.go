// Package engine implements the serving engines: a colocated baseline that
// interleaves prefill+decode on shared instances (causing cross-phase
// contention), and a PD-disaggregated engine with dedicated prefill/decode
// workers streaming KV cache between them.
//
// Both engines are event-chain driven on a clock.Clock: every simulated
// delay is a scheduled timer, never a blocking sleep. Under clock.Sim the
// whole run executes as a deterministic discrete-event simulation (fast and
// bit-reproducible for a given seed, P1-1/P1-2); under clock.Real the same
// chains run on wall-clock timers.
package engine

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
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

	// Clk is the execution clock. Nil means clock.Real (wall-clock timers).
	// Pass clock.NewSim() for deterministic discrete-event runs.
	Clk clock.Clock
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
//
// Known limitation (spec P0-2, scheduled fix): KV bytes per token omits the
// layer count and the model-weight read term is not modeled, so decode times
// are ~32x low and batch-1 TPOT is unrealistically cheap. Acceptance tests
// for the corrected roofline land with pkg/model.
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
	// InFlight reports requests currently admitted and not finished
	// (queued or executing). Used by the CLI to drain before stopping.
	InFlight() int
	// Run drives a Sim-clock engine to completion (arrivals exhausted,
	// chains retired) and stops it. It is a no-op on Real-clock engines,
	// which the caller drives and stops itself.
	Run()
	// RunUntil drives a Sim-clock engine until the clock reaches the given
	// simulated horizon (no drain wait). No-op on Real-clock engines.
	RunUntil(d time.Duration)
}

// engineCore holds state shared by both engines: clock, registry, KV pool,
// arrival chain, stop plumbing and in-flight accounting.
type engineCore struct {
	cfg    Config
	clk    clock.Clock
	runner ModelRunner
	reg    *metrics.Registry
	pool   *kvcache.Pool
	queue  *scheduler.Queue
	arrive *ArrivalGen

	stopOnce sync.Once
	stop     chan struct{}
	t0       time.Time

	inflight     atomic.Int64
	arrivalsDone atomic.Bool
}

func (c *engineCore) init(cfg Config) {
	c.cfg = cfg
	c.clk = cfg.Clk
	if c.clk == nil {
		c.clk = clock.NewReal()
	}
	c.runner = NewSimRunner(cfg.BandwidthGBps, cfg.ComputeTFLOPs)
	c.reg = metrics.NewRegistryOn(c.clk)
	c.pool = kvcache.NewPool(cfg.KVCapacityBlocks, cfg.KVBlockSize)
	c.queue = scheduler.NewQueue(nil)
	c.stop = make(chan struct{})
	c.arrive = NewArrivalGen(c.clk, cfg.Load)
}

// Stopped reports whether Stop has been initiated.
func (c *engineCore) Stopped() bool {
	select {
	case <-c.stop:
		return true
	default:
		return false
	}
}

// requestStop closes the stop channel exactly once (P1-3: no double-close
// panic) and returns true on the first call.
func (c *engineCore) requestStop() bool {
	first := false
	c.stopOnce.Do(func() {
		first = true
		close(c.stop)
	})
	return first
}

// sleep schedules the continuation after d on the clock. Never blocks: the
// continuation runs as a clock event (synchronously under Sim).
func (c *engineCore) sleep(d time.Duration, cont func()) {
	c.clk.AfterFunc(d, cont)
}

// admission adds a request to the in-flight set; the returned release func
// must be called exactly once when the request's chain terminates.
func (c *engineCore) admit() (release func()) {
	c.inflight.Add(1)
	var once sync.Once
	return func() { once.Do(func() { c.inflight.Add(-1) }) }
}

// InFlight implements Engine.
func (c *engineCore) InFlight() int { return int(c.inflight.Load()) }

// ArrivalsDone reports whether the arrival process has ended (used by
// monitors and tools to detect quiescence).
func (c *engineCore) ArrivalsDone() bool { return c.arrivalsDone.Load() }

// Registry implements Engine.
func (c *engineCore) Registry() *metrics.Registry { return c.reg }

// runArrivalChain drives the arrival process as a self-perpetuating event
// chain: sample an inter-arrival, schedule the push, repeat until the load
// duration is reached or the engine stops. No polling goroutine, no wall
// clock.
func (c *engineCore) runArrivalChain(arrive *ArrivalGen, q *scheduler.Queue) {
	if c.Stopped() {
		c.arrivalsDone.Store(true)
		return
	}
	runSec := c.clk.Now().Sub(c.t0).Seconds()
	if runSec >= c.cfg.Load.DurationSec {
		c.arrivalsDone.Store(true)
		return
	}
	us := arrive.NextInterarrivalUS(runSec)
	c.clk.AfterFunc(usDur(us), func() {
		if c.Stopped() {
			c.arrivalsDone.Store(true)
			return
		}
		q.Push(arrive.Next())
		c.runArrivalChain(arrive, q)
	})
}

// workPending reports whether any queue holds work or any request is still
// in flight. Queues may be *scheduler.Queue or *handoffQueue (engine-
// internal) — both expose Len().
func (c *engineCore) workPending(qs ...interface{ Len() int }) bool {
	if c.InFlight() > 0 {
		return true
	}
	for _, q := range qs {
		if q.Len() > 0 {
			return true
		}
	}
	return false
}

// canRetire reports whether an idle worker chain may retire: the arrival
// process has ended and no work remains anywhere. Once retireable, no event
// can create new work, so retirement is final.
func (c *engineCore) canRetire(qs ...interface{ Len() int }) bool {
	return c.arrivalsDone.Load() && !c.workPending(qs...)
}

// Idle re-queue delays. An idle worker chain polls with exponential backoff
// so long simulated runs stay cheap; the cap bounds the phantom queue-wait
// a polling worker can add to a request (≤ maxIdlePoll of simulated time).
const (
	idlePollInterval = 200 * time.Microsecond
	maxIdlePoll      = 10 * time.Millisecond
)

// nextIdle doubles the idle delay up to maxIdlePoll.
func nextIdle(idle time.Duration) time.Duration {
	if idle <= 0 {
		return idlePollInterval
	}
	idle *= 2
	if idle > maxIdlePoll {
		return maxIdlePoll
	}
	return idle
}

// runSim drives a Sim-clock engine until the arrival duration has elapsed,
// all arrivals have drained through the system, and every worker chain has
// retired — then stops the engine. Deterministic and instant.
func (c *engineCore) runSim(qs ...interface{ Len() int }) {
	sim, ok := c.clk.(*clock.Sim)
	if !ok {
		return // Real engines are caller-driven
	}
	for sim.Len() > 0 {
		sim.Advance()
	}
	// Arrivals are generated for cfg.Load.DurationSec; run past that
	// horizon until queued/in-flight work drains.
	for c.workPending(qs...) && !c.Stopped() {
		if sim.Len() == 0 {
			break // chains will reschedule; nothing left to fire this instant
		}
		sim.Advance()
	}
	c.requestStop()
	for sim.Len() > 0 {
		sim.Advance()
	}
}

// runUntil drives the Sim clock until the horizon (or event exhaustion)
// without waiting for drain, then stops the engine.
func (c *engineCore) runUntil(horizon time.Duration, qs ...interface{ Len() int }) {
	sim, ok := c.clk.(*clock.Sim)
	if !ok {
		return
	}
	sim.Run(horizon)
	_ = qs
	c.requestStop()
	for sim.Len() > 0 {
		sim.Advance()
	}
}

// usDur converts microseconds to a Duration.
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

// finishRequest records metrics for a completed request and releases its KV
// blocks.
func finishRequest(reg *metrics.Registry, req *scheduler.Request, pool *kvcache.Pool, now time.Time) {
	req.FinishAt = now
	tokens := maxInt(1, req.Generated)
	tpot := float64(req.FinishAt.Sub(req.FirstTokenAt)) / 1e3 / float64(tokens)
	reg.RecordRequest(req.TTFTUS(), tpot, req.E2EUS(), int64(req.Generated))
	pool.Release(req.ID)
}

// allocateKV reserves prompt blocks for a request and pins them against
// eviction. The returned slice is consumed by KV-transfer paths; the
// allocation itself is what creates capacity pressure either way.
func allocateKV(pool *kvcache.Pool, blockSize int, req *scheduler.Request) ([]*kvcache.Block, bool) {
	nblocks := (req.PromptTokens + blockSize - 1) / blockSize
	blocks, err := pool.Allocate(req.ID, nblocks)
	if err != nil {
		return nil, false
	}
	pool.Acquire(req.ID)
	return blocks, true
}
