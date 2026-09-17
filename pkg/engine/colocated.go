// colocated.go implements the baseline engine where prefill and decode share
// every GPU. New arrivals' prefill chunks stall active decoding, inflating
// TPOT (the 2x-30x tail the problem statement describes).
//
// Each GPU runs a self-perpetuating event chain: pop → prefill chunks →
// decode steps → next pop, with every delay scheduled on the clock. Chains
// retire when the arrival process has ended and no work remains, so Sim runs
// reach quiescence (P1-1).
package engine

import (
	"fmt"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/scheduler"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/types"
)

// ColocatedEngine runs prefill+decode on every GPU (vLLM/SGLang baseline).
type ColocatedEngine struct {
	core engineCore
	gpus []*types.GPUInstance
}

// NewColocated builds a colocated engine.
func NewColocated(cfg Config) (*ColocatedEngine, error) {
	if cfg.NumGPUs <= 0 {
		return nil, fmt.Errorf("numGPUs must be > 0, got %d", cfg.NumGPUs)
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
	e := &ColocatedEngine{gpus: gpus}
	e.core.init(cfg)
	return e, nil
}

// Name implements Engine.
func (e *ColocatedEngine) Name() string { return "colocated" }

// Registry implements Engine.
func (e *ColocatedEngine) Registry() *metrics.Registry { return e.core.Registry() }

// Queue implements Engine.
func (e *ColocatedEngine) Queue() *scheduler.Queue { return e.core.queue }

// PoolSizes implements Engine: colocated GPUs serve both phases.
func (e *ColocatedEngine) PoolSizes() (int, int) { return e.core.cfg.NumGPUs, e.core.cfg.NumGPUs }

// InFlight implements Engine.
func (e *ColocatedEngine) InFlight() int { return e.core.InFlight() }

// Clock exposes the execution clock (tests, tools).
func (e *ColocatedEngine) Clock() clock.Clock { return e.core.clk }

// Start implements Engine: boots one event chain per GPU plus the arrival
// chain.
func (e *ColocatedEngine) Start() {
	e.core.t0 = e.core.clk.Now()
	for _, g := range e.gpus {
		g.State = types.StateHealthy
		e.spawnWorker(g)
	}
	e.core.runArrivalChain(e.core.arrive, e.core.queue)
}

// Stop implements Engine: stops admission and drains in-flight chains; safe
// to call twice (P1-3).
func (e *ColocatedEngine) Stop() {
	e.core.requestStop()
}

// Run implements Engine: drives the Sim clock until arrivals are exhausted
// and the system drains, then stops. No-op on Real-clock engines.
func (e *ColocatedEngine) Run() { e.core.runSim(e.core.queue) }

// RunUntil implements Engine: drives the Sim clock to a simulated horizon
// without waiting for drain. No-op on Real-clock engines.
func (e *ColocatedEngine) RunUntil(d time.Duration) { e.core.runUntil(d, e.core.queue) }

// spawnWorker starts the GPU's event chain.
func (e *ColocatedEngine) spawnWorker(g *types.GPUInstance) {
	e.core.clk.AfterFunc(0, func() { e.workerStep(g, 0) })
}

// workerStep pops one request and processes it to completion as a chain of
// clock events; when idle it backs off and retires once no work can arrive.
func (e *ColocatedEngine) workerStep(g *types.GPUInstance, idle time.Duration) {
	if e.core.Stopped() {
		return
	}
	req, ok := e.core.queue.Pop()
	if !ok {
		if e.core.canRetire() {
			return // retire: no arrivals remain and no work exists anywhere
		}
		d := nextIdle(idle)
		e.core.sleep(d, func() { e.workerStep(g, d) })
		return
	}
	g.State = types.StateBusy
	e.process(g, req)
}

// process runs prefill then decode on one shared GPU as scheduled events.
// While this GPU is producing tokens, any queued request's prefill chunk gets
// scheduled first (vLLM-style chunked-prefill priority), stalling our decode
// step — the source of TPOT spikes.
func (e *ColocatedEngine) process(g *types.GPUInstance, req *scheduler.Request) {
	release := e.core.admit()
	blocks, okAlloc := allocateKV(e.core.pool, e.core.cfg.KVBlockSize, req)
	if !okAlloc {
		e.core.queue.Drop()
		e.core.reg.RecordRejection("kv_exhausted")
		g.State = types.StateHealthy
		e.core.sleep(idlePollInterval, func() { e.workerStep(g, 0) })
		release()
		return
	}
	_ = blocks
	e.core.reg.RecordPrefillTokens(int64(req.PromptTokens))

	// Prefill phase (chunked).
	remaining := req.PromptTokens
	var prefillStep func()
	prefillStep = func() {
		if e.core.Stopped() {
			release()
			return
		}
		chunk := e.core.cfg.PrefillChunkTokens
		if chunk <= 0 || remaining < chunk {
			chunk = remaining
		}
		e.core.sleep(usDur(e.core.runner.PrefillUS(chunk)), func() {
			remaining -= chunk
			if remaining > 0 {
				prefillStep()
				return
			}
			req.FirstTokenAt = e.core.clk.Now()
			req.AdvanceDecode(1) // first token falls out of prefill
			e.decodePhase(g, req, release)
		})
	}
	prefillStep()
}

// decodePhase emits the remaining output tokens, interrupted by new arrivals'
// prefill chunks (cross-phase contention).
func (e *ColocatedEngine) decodePhase(g *types.GPUInstance, req *scheduler.Request, release func()) {
	if e.core.Stopped() {
		release()
		return
	}
	if req.IsFinished() {
		g.State = types.StateHealthy
		finishRequest(e.core.reg, req, e.core.pool, e.core.clk.Now())
		release()
		e.core.sleep(idlePollInterval, func() { e.workerStep(g, 0) })
		return
	}
	dec := func() {
		e.core.sleep(usDur(e.core.runner.DecodeUS(req.PromptTokens+req.Generated)), func() {
			req.AdvanceDecode(1)
			e.decodePhase(g, req, release)
		})
	}
	if e.core.cfg.ChunkedPrefill && e.core.queue.Len() > 0 {
		// A waiting request's prefill chunk runs before our next decode
		// step: decode stalls (cross-phase contention).
		e.core.sleep(usDur(e.core.runner.PrefillUS(e.core.cfg.PrefillChunkTokens)), dec)
		return
	}
	dec()
}
