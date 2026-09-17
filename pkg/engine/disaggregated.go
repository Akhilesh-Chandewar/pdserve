// disaggregated.go implements PD disaggregation: dedicated prefill and decode
// pools with a KV-cache connector streaming cache state between them.
//
// Engine shape: each GPU owns a worker with a per-instance stop channel and
// one event chain. ResizePools stops the worker's chain (pending continuations
// see the stop flag and exit without rescheduling — the drain), rebinds the
// role, and respawns exactly one new chain. No second concurrent chain is
// ever created for the same GPU (P0-4 fix).
package engine

import (
	"fmt"
	"sync"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/kvcache"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/scheduler"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/types"
)

// DisaggregatedEngine runs dedicated prefill and decode pools.
type DisaggregatedEngine struct {
	core engineCore
	conn kvcache.Connector

	decodeQueue *handoffQueue

	mu          sync.Mutex
	prefillGPUs []*worker
	decodeGPUs  []*worker

	drainUS float64 // modeled per-flip drain latency
}

// worker owns one GPU instance's event chain. The stop channel is closed on
// drain/stop; every continuation checks it before doing work or rescheduling,
// so closing the channel drains the chain (in-flight step completes, queued
// continuations no-op).
type worker struct {
	inst *types.GPUInstance
	stop chan struct{}
}

func newWorker(inst *types.GPUInstance) *worker {
	return &worker{inst: inst, stop: make(chan struct{})}
}

func workerStopped(w *worker) bool {
	select {
	case <-w.stop:
		return true
	default:
		return false
	}
}

// handoff carries a prefilled request to the decode pool together with its
// in-flight admission, which transfers so the request stays accounted for
// exactly once across the prefill→decode seam.
type handoff struct {
	req     *scheduler.Request
	release func()
}

// handoffQueue holds requests that finished prefill and wait for a decode
// worker.
type handoffQueue struct {
	mu    sync.Mutex
	items []*handoff
}

func newHandoffQueue() *handoffQueue { return &handoffQueue{} }

// Push enqueues a handoff.
func (q *handoffQueue) Push(h *handoff) {
	q.mu.Lock()
	q.items = append(q.items, h)
	q.mu.Unlock()
}

// Pop dequeues the next handoff, ok=false when empty.
func (q *handoffQueue) Pop() (*handoff, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil, false
	}
	h := q.items[0]
	q.items = q.items[1:]
	return h, true
}

// Len returns current queue depth.
func (q *handoffQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

// NewDisaggregated builds a PD-disaggregated engine with a static split.
func NewDisaggregated(cfg Config) (*DisaggregatedEngine, error) {
	if cfg.NumGPUs < 2 {
		return nil, fmt.Errorf("disaggregation needs >= 2 GPUs, got %d", cfg.NumGPUs)
	}
	nPre := int(float64(cfg.NumGPUs) * cfg.PrefillRatio)
	if nPre < 1 {
		nPre = 1
	}
	if nPre >= cfg.NumGPUs {
		nPre = cfg.NumGPUs - 1
	}
	pre := make([]*worker, 0, nPre)
	dec := make([]*worker, 0, cfg.NumGPUs-nPre)
	for i := 0; i < cfg.NumGPUs; i++ {
		g := &types.GPUInstance{
			ID:            types.InstanceID(),
			Role:          types.RoleIdle,
			State:         types.StateHealthy,
			WeightsLoaded: true,
			BandwidthGBps: cfg.BandwidthGBps,
			ComputeTFLOPs: cfg.ComputeTFLOPs,
		}
		w := newWorker(g)
		if i < nPre {
			g.Role = types.RolePrefill
			pre = append(pre, w)
		} else {
			g.Role = types.RoleDecode
			dec = append(dec, w)
		}
	}

	e := &DisaggregatedEngine{
		conn:        kvcache.MemoryConnector{},
		decodeQueue: newHandoffQueue(),
		prefillGPUs: pre,
		decodeGPUs:  dec,
		drainUS:     500, // sub-ms modeled drain; configurable for cost studies
	}
	e.core.init(cfg)
	if cfg.KVConnector == "tcp" {
		tc, err := kvcache.NewTCPConnector(cfg.KVTCPBandwidthGBps)
		if err != nil {
			return nil, err
		}
		e.conn = tc
	}
	return e, nil
}

// Connector exposes the KV connector for tests.
func (e *DisaggregatedEngine) Connector() kvcache.Connector { return e.conn }

// SetDrainUS sets the modeled per-flip drain latency (drain cost is a
// first-class elastic metric per P0-4).
func (e *DisaggregatedEngine) SetDrainUS(us float64) { e.drainUS = us }

// DrainUS returns the modeled per-flip drain latency.
func (e *DisaggregatedEngine) DrainUS() float64 { return e.drainUS }

// Name implements Engine.
func (e *DisaggregatedEngine) Name() string {
	pre, dec := e.PoolSizes()
	return fmt.Sprintf("disagg(%s, pre=%d/dec=%d)", e.conn.Name(), pre, dec)
}

// Registry implements Engine.
func (e *DisaggregatedEngine) Registry() *metrics.Registry { return e.core.Registry() }

// Queue implements Engine.
func (e *DisaggregatedEngine) Queue() *scheduler.Queue { return e.core.queue }

// PoolSizes implements Engine.
func (e *DisaggregatedEngine) PoolSizes() (int, int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.prefillGPUs), len(e.decodeGPUs)
}

// InFlight implements Engine.
func (e *DisaggregatedEngine) InFlight() int { return e.core.InFlight() }

// ArrivalsDone reports whether the arrival process has ended.
func (e *DisaggregatedEngine) ArrivalsDone() bool { return e.core.ArrivalsDone() }

// Clock exposes the execution clock (tests, tools).
func (e *DisaggregatedEngine) Clock() clock.Clock { return e.core.clk }

// Start implements Engine: boots one chain per worker plus arrivals.
func (e *DisaggregatedEngine) Start() {
	e.core.t0 = e.core.clk.Now()
	e.mu.Lock()
	pre := append([]*worker(nil), e.prefillGPUs...)
	dec := append([]*worker(nil), e.decodeGPUs...)
	e.mu.Unlock()
	for _, w := range pre {
		e.spawnPrefill(w)
	}
	for _, w := range dec {
		e.spawnDecode(w)
	}
	e.core.runArrivalChain(e.core.arrive, e.core.queue)
}

// Stop implements Engine (double-close safe via core.requestStop, P1-3).
func (e *DisaggregatedEngine) Stop() {
	e.core.requestStop()
	e.mu.Lock()
	pre := append([]*worker(nil), e.prefillGPUs...)
	dec := append([]*worker(nil), e.decodeGPUs...)
	e.mu.Unlock()
	for _, w := range pre {
		stopWorker(w)
	}
	for _, w := range dec {
		stopWorker(w)
	}
}

// Run implements Engine: drives the Sim clock until arrivals are exhausted
// and the system drains, then stops. No-op on Real-clock engines.
func (e *DisaggregatedEngine) Run() { e.core.runSim(e.core.queue, e.decodeQueue) }

// RunUntil implements Engine: drives the Sim clock to a simulated horizon
// without waiting for drain. No-op on Real-clock engines.
func (e *DisaggregatedEngine) RunUntil(d time.Duration) {
	e.core.runUntil(d, e.core.queue, e.decodeQueue)
}

func stopWorker(w *worker) {
	if !workerStopped(w) {
		close(w.stop)
	}
}

// spawnPrefill starts (or restarts) a worker's prefill chain.
func (e *DisaggregatedEngine) spawnPrefill(w *worker) {
	w.inst.State = types.StateHealthy
	e.core.clk.AfterFunc(0, func() { e.prefillStep(w, 0) })
}

// spawnDecode starts (or restarts) a worker's decode chain.
func (e *DisaggregatedEngine) spawnDecode(w *worker) {
	w.inst.State = types.StateHealthy
	e.core.clk.AfterFunc(0, func() { e.decodeStep(w, 0) })
}

// prefillStep pops one request and runs prefill+KV-transfer as an event
// chain; when idle it backs off and retires once no work can arrive.
func (e *DisaggregatedEngine) prefillStep(w *worker, idle time.Duration) {
	if e.core.Stopped() || workerStopped(w) {
		return
	}
	req, ok := e.core.queue.Pop()
	if !ok {
		if e.core.canRetire(e.core.queue, e.decodeQueue) {
			return
		}
		d := nextIdle(idle)
		e.core.sleep(d, func() { e.prefillStep(w, d) })
		return
	}
	w.inst.State = types.StateBusy
	release := e.core.admit()
	blocks, okAlloc := allocateKV(e.core.pool, e.core.cfg.KVBlockSize, req)
	if !okAlloc {
		e.core.queue.Drop()
		e.core.reg.RecordRejection("kv_exhausted")
		w.inst.State = types.StateHealthy
		e.core.sleep(idlePollInterval, func() { e.prefillStep(w, 0) })
		release()
		return
	}
	e.core.reg.RecordPrefillTokens(int64(req.PromptTokens))
	e.core.sleep(usDur(e.core.runner.PrefillUS(req.PromptTokens)), func() {
		// Stream KV cache to the decode pool. The transfer cost is modeled
		// analytically and scheduled as simulated time (P1-8): no sockets.
		us := e.conn.TransferTimeUS(blocks)
		e.core.sleep(usDur(us), func() {
			req.FirstTokenAt = e.core.clk.Now()
			req.AdvanceDecode(1)
			e.decodeQueue.Push(&handoff{req: req, release: release})
			w.inst.State = types.StateHealthy
			e.core.sleep(idlePollInterval, func() { e.prefillStep(w, 0) })
		})
	})
}

// decodeStep pops one handed-off request and decodes it to completion as an
// event chain; when idle it backs off and retires once no work can arrive.
func (e *DisaggregatedEngine) decodeStep(w *worker, idle time.Duration) {
	if e.core.Stopped() || workerStopped(w) {
		return
	}
	h, ok := e.decodeQueue.Pop()
	if !ok {
		if e.core.canRetire(e.core.queue, e.decodeQueue) {
			return
		}
		d := nextIdle(backoffStep(idle))
		e.core.sleep(d, func() { e.decodeStep(w, backoffStep(idle)) })
		return
	}
	w.inst.State = types.StateBusy
	req, release := h.req, h.release
	var step func()
	step = func() {
		if e.core.Stopped() {
			release()
			return
		}
		if req.IsFinished() {
			w.inst.State = types.StateHealthy
			finishRequest(e.core.reg, req, e.core.pool, e.core.clk.Now())
			release()
			e.core.sleep(idlePollInterval, func() { e.decodeStep(w, 0) })
			return
		}
		// Pure decode: no prefill ever runs here, so ITL stays flat even
		// during arrival bursts.
		e.core.sleep(usDur(e.core.runner.DecodeUS(req.PromptTokens+req.Generated)), func() {
			req.AdvanceDecode(1)
			step()
		})
	}
	step()
}

// backoffStep returns the next backoff delay, capped at maxIdlePoll.
func backoffStep(idle time.Duration) time.Duration { return nextIdle(idle) }

// ResizePools drains n workers from decode→prefill and m from prefill→decode:
// close the worker's stop channel (its chain drains: the in-flight step
// completes, pending continuations no-op), mark the drain, rebind the role,
// respawn one new chain (P0-4 fix). Worker counts are conserved; flips move
// capacity instead of adding it.
func (e *DisaggregatedEngine) ResizePools(toPrefill, toDecode int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for ; toPrefill > 0 && len(e.decodeGPUs) > 0; toPrefill-- {
		w := e.decodeGPUs[len(e.decodeGPUs)-1]
		e.decodeGPUs = e.decodeGPUs[:len(e.decodeGPUs)-1]
		e.flip(w, types.RolePrefill)
		e.prefillGPUs = append(e.prefillGPUs, w)
	}
	for ; toDecode > 0 && len(e.prefillGPUs) > 0; toDecode-- {
		w := e.prefillGPUs[len(e.prefillGPUs)-1]
		e.prefillGPUs = e.prefillGPUs[:len(e.prefillGPUs)-1]
		e.flip(w, types.RoleDecode)
		e.decodeGPUs = append(e.decodeGPUs, w)
	}
}

// flip drains one worker and rebinds it to the new role. The modeled drain
// latency is charged as simulated time before the worker rejoins its new
// pool, so flip cost is visible in results. The worker's in-flight request
// completes on it (drain semantics); pending idle continuations no-op.
func (e *DisaggregatedEngine) flip(w *worker, to types.Role) {
	if workerStopped(w) {
		return // already drained (engine stopping)
	}
	w.inst.State = types.StateDraining
	close(w.stop) // drain: pending continuations no-op; in-flight step completes
	after := e.drainUS
	w.inst.Role = to
	w.inst.LastRoleFlip = e.core.clk.Now()
	e.core.clk.AfterFunc(usDur(after), func() {
		if e.core.Stopped() {
			w.inst.State = types.StateFailed
			return
		}
		w.stop = make(chan struct{}) // fresh lifecycle
		if to == types.RolePrefill {
			e.spawnPrefill(w)
		} else {
			e.spawnDecode(w)
		}
	})
}
