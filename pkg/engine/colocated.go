// colocated.go implements the baseline engine where prefill and decode share
// every GPU. New arrivals' prefill chunks stall active decoding, inflating
// TPOT (the 2x-30x tail the problem statement describes).
package engine

import (
	"fmt"
	"sync"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/kvcache"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/scheduler"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/types"
)

// ColocatedEngine runs prefill+decode on every GPU (vLLM/SGLang baseline).
type ColocatedEngine struct {
	cfg    Config
	runner ModelRunner
	reg    *metrics.Registry
	queue  *scheduler.Queue
	pool   *kvcache.Pool
	gpus   []*types.GPUInstance
	wg     sync.WaitGroup
	stop   chan struct{}
	arrive *ArrivalGen
	t0     time.Time
}

// NewColocated builds a colocated engine.
func NewColocated(cfg Config) (*ColocatedEngine, error) {
	if cfg.NumGPUs <= 0 {
		return nil, fmt.Errorf("numGPUs must be > 0")
	}
	gpus := make([]*types.GPUInstance, cfg.NumGPUs)
	for i := range gpus {
		gpus[i] = &types.GPUInstance{
			ID:            types.InstanceID(),
			Role:          types.RoleDecode, // colocated: every GPU does both phases
			State:         types.StateHealthy,
			WeightsLoaded: true,
			BandwidthGBps: cfg.BandwidthGBps,
			ComputeTFLOPs: cfg.ComputeTFLOPs,
		}
	}
	reg := metrics.NewRegistry()
	e := &ColocatedEngine{
		cfg:    cfg,
		runner: NewSimRunner(cfg.BandwidthGBps, cfg.ComputeTFLOPs),
		reg:    reg,
		queue:  scheduler.NewQueue(nil),
		pool:   kvcache.NewPool(cfg.KVCapacityBlocks, cfg.KVBlockSize),
		gpus:   gpus,
		stop:   make(chan struct{}),
	}
	e.arrive = NewArrivalGen(e.queue, cfg.Load)
	return e, nil
}

// Name implements Engine.
func (e *ColocatedEngine) Name() string { return "colocated" }

// Registry implements Engine.
func (e *ColocatedEngine) Registry() *metrics.Registry { return e.reg }

// Queue implements Engine.
func (e *ColocatedEngine) Queue() *scheduler.Queue { return e.queue }

// PoolSizes implements Engine: colocated GPUs serve both phases.
func (e *ColocatedEngine) PoolSizes() (int, int) { return e.cfg.NumGPUs, e.cfg.NumGPUs }

// Start implements Engine.
func (e *ColocatedEngine) Start() {
	e.t0 = time.Now()
	e.reg.Start()
	for _, g := range e.gpus {
		e.wg.Add(1)
		go e.workerLoop(g)
	}
	e.wg.Add(1)
	go e.arrivalLoop()
}

// Stop implements Engine.
func (e *ColocatedEngine) Stop() {
	close(e.stop)
	e.wg.Wait()
}

func (e *ColocatedEngine) arrivalLoop() {
	defer e.wg.Done()
	next := time.NewTimer(0)
	defer next.Stop()
	for {
		runSec := time.Since(e.t0).Seconds()
		if runSec >= e.cfg.Load.DurationSec {
			return
		}
		us := e.arrive.NextInterarrivalUS(runSec)
		next.Reset(usDur(us))
		select {
		case <-e.stop:
			return
		case <-next.C:
			e.queue.Push(e.arrive.Next(time.Now()))
		}
	}
}

func (e *ColocatedEngine) workerLoop(g *types.GPUInstance) {
	defer e.wg.Done()
	for {
		select {
		case <-e.stop:
			return
		default:
		}
		req, ok := e.queue.Pop()
		if !ok {
			time.Sleep(200 * time.Microsecond)
			continue
		}
		e.process(g, req)
	}
}

// process runs prefill then decode on one shared GPU. While this GPU is
// producing tokens, any queued request's prefill chunk gets scheduled first
// (vLLM-style chunked-prefill priority), stalling our decode step -- the
// source of TPOT spikes.
func (e *ColocatedEngine) process(g *types.GPUInstance, req *scheduler.Request) {
	blocks, okAlloc := allocateKV(e.pool, e.cfg.KVBlockSize, req)
	if !okAlloc {
		e.queue.Drop()
		e.reg.RecordRequest(0, 0, 0, 0, true)
		return
	}
	_ = blocks
	e.reg.RecordPrefillTokens(int64(req.PromptTokens))

	// Prefill phase (chunked).
	remaining := req.PromptTokens
	for remaining > 0 {
		chunk := e.cfg.PrefillChunkTokens
		if chunk <= 0 || remaining < chunk {
			chunk = remaining
		}
		time.Sleep(usDur(e.runner.PrefillUS(chunk)))
		remaining -= chunk
	}
	req.FirstTokenAt = time.Now()
	req.AdvanceDecode(1) // first token falls out of prefill

	// Decode phase, interrupted by new arrivals' prefill chunks.
	for !req.IsFinished() {
		if e.cfg.ChunkedPrefill && e.queue.Len() > 0 {
			// A waiting request's prefill chunk runs before our next decode
			// step: decode stalls (cross-phase contention).
			time.Sleep(usDur(e.runner.PrefillUS(e.cfg.PrefillChunkTokens)))
		}
		time.Sleep(usDur(e.runner.DecodeUS(req.PromptTokens + req.Generated)))
		req.AdvanceDecode(1)
	}
	finishRequest(e.reg, req, e.pool)
}
