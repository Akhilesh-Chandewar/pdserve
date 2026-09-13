// disaggregated.go implements PD disaggregation: dedicated prefill and decode
// pools with a KV-cache connector streaming cache state between them.
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

// DisaggregatedEngine runs dedicated prefill and decode pools.
type DisaggregatedEngine struct {
	cfg    Config
	runner ModelRunner
	reg    *metrics.Registry

	prefillQ *scheduler.Queue
	decodeQ  *scheduler.Queue
	pool     *kvcache.Pool
	conn     kvcache.Connector

	prefillGPUs []*types.GPUInstance
	decodeGPUs  []*types.GPUInstance

	wg      sync.WaitGroup
	mu      sync.Mutex
	stopped bool
	stop    chan struct{}
	t0      time.Time
}

// NewDisaggregated builds a PD-disaggregated engine with a static split.
func NewDisaggregated(cfg Config) (*DisaggregatedEngine, error) {
	if cfg.NumGPUs < 2 {
		return nil, fmt.Errorf("disaggregation needs >= 2 GPUs")
	}
	nPre := int(float64(cfg.NumGPUs) * cfg.PrefillRatio)
	if nPre < 1 {
		nPre = 1
	}
	if nPre >= cfg.NumGPUs {
		nPre = cfg.NumGPUs - 1
	}
	pre := make([]*types.GPUInstance, 0, nPre)
	dec := make([]*types.GPUInstance, 0, cfg.NumGPUs-nPre)
	for i := 0; i < cfg.NumGPUs; i++ {
		g := &types.GPUInstance{
			ID:            types.InstanceID(),
			Role:          types.RoleIdle,
			State:         types.StateHealthy,
			WeightsLoaded: true,
			BandwidthGBps: cfg.BandwidthGBps,
			ComputeTFLOPs: cfg.ComputeTFLOPs,
			LastRoleFlip:  time.Now(),
		}
		if i < nPre {
			g.Role = types.RolePrefill
			pre = append(pre, g)
		} else {
			g.Role = types.RoleDecode
			dec = append(dec, g)
		}
	}

	reg := metrics.NewRegistry()
	var conn kvcache.Connector = kvcache.MemoryConnector{}
	if cfg.KVConnector == "tcp" {
		tc, err := kvcache.NewTCPConnector(cfg.KVTCPBandwidthGBps)
		if err != nil {
			return nil, err
		}
		conn = tc
	}

	e := &DisaggregatedEngine{
		cfg:         cfg,
		runner:      NewSimRunner(cfg.BandwidthGBps, cfg.ComputeTFLOPs),
		reg:         reg,
		prefillQ:    scheduler.NewQueue(nil),
		decodeQ:     scheduler.NewQueue(nil),
		pool:        kvcache.NewPool(cfg.KVCapacityBlocks, cfg.KVBlockSize),
		conn:        conn,
		prefillGPUs: pre,
		decodeGPUs:  dec,
		stop:        make(chan struct{}),
	}
	return e, nil
}

// Name implements Engine.
func (e *DisaggregatedEngine) Name() string {
	return fmt.Sprintf("disagg(%s, pre=%d/dec=%d)", e.conn.Name(), len(e.prefillGPUs), len(e.decodeGPUs))
}

// Registry implements Engine.
func (e *DisaggregatedEngine) Registry() *metrics.Registry { return e.reg }

// Queue implements Engine.
func (e *DisaggregatedEngine) Queue() *scheduler.Queue { return e.prefillQ }

// PoolSizes implements Engine.
func (e *DisaggregatedEngine) PoolSizes() (int, int) {
	return len(e.prefillGPUs), len(e.decodeGPUs)
}

// Connector exposes the KV connector for tests.
func (e *DisaggregatedEngine) Connector() kvcache.Connector { return e.conn }

// ResizePools moves n GPUs from decode to prefill and m from prefill to
// decode, spawning/retiring worker goroutines accordingly. This is the
// elastic role-flip primitive: instances are stateless and keep weights
// resident, so a flip costs no model reload.
func (e *DisaggregatedEngine) ResizePools(toPrefill, toDecode int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for ; toPrefill > 0 && len(e.decodeGPUs) > 0; toPrefill-- {
		g := e.decodeGPUs[len(e.decodeGPUs)-1]
		e.decodeGPUs = e.decodeGPUs[:len(e.decodeGPUs)-1]
		g.Role = types.RolePrefill
		g.LastRoleFlip = time.Now()
		e.prefillGPUs = append(e.prefillGPUs, g)
		e.wg.Add(1)
		go e.prefillLoop(g)
	}
	for ; toDecode > 0 && len(e.prefillGPUs) > 0; toDecode-- {
		g := e.prefillGPUs[len(e.prefillGPUs)-1]
		e.prefillGPUs = e.prefillGPUs[:len(e.prefillGPUs)-1]
		g.Role = types.RoleDecode
		g.LastRoleFlip = time.Now()
		e.decodeGPUs = append(e.decodeGPUs, g)
		e.wg.Add(1)
		go e.decodeLoop(g)
	}
}

// Start implements Engine.
func (e *DisaggregatedEngine) Start() {
	e.t0 = time.Now()
	e.reg.Start()
	for _, g := range e.prefillGPUs {
		e.wg.Add(1)
		go e.prefillLoop(g)
	}
	for _, g := range e.decodeGPUs {
		e.wg.Add(1)
		go e.decodeLoop(g)
	}
	e.wg.Add(1)
	go e.arrivalLoop()
}

// Stop implements Engine.
func (e *DisaggregatedEngine) Stop() {
	e.mu.Lock()
	if e.stopped {
		e.mu.Unlock()
		return
	}
	e.stopped = true
	e.mu.Unlock()
	close(e.stop)
	e.wg.Wait()
}

func (e *DisaggregatedEngine) arrivalLoop() {
	defer e.wg.Done()
	arrive := NewArrivalGen(e.prefillQ, e.cfg.Load)
	next := time.NewTimer(0)
	defer next.Stop()
	for {
		runSec := time.Since(e.t0).Seconds()
		if runSec >= e.cfg.Load.DurationSec {
			return
		}
		us := arrive.NextInterarrivalUS(runSec)
		next.Reset(usDur(us))
		select {
		case <-e.stop:
			return
		case <-next.C:
			e.prefillQ.Push(arrive.Next(time.Now()))
		}
	}
}

func (e *DisaggregatedEngine) prefillLoop(g *types.GPUInstance) {
	defer e.wg.Done()
	for {
		select {
		case <-e.stop:
			return
		default:
		}
		req, ok := e.prefillQ.Pop()
		if !ok {
			time.Sleep(200 * time.Microsecond)
			continue
		}
		blocks, okAlloc := allocateKV(e.pool, e.cfg.KVBlockSize, req)
		if !okAlloc {
			e.prefillQ.Drop()
			e.reg.RecordRequest(0, 0, 0, 0, true)
			continue
		}
		e.reg.RecordPrefillTokens(int64(req.PromptTokens))
		time.Sleep(usDur(e.runner.PrefillUS(req.PromptTokens)))

		// Stream KV cache to the decode pool. Transfer cost replaces the
		// compute stall a colocated decode would have suffered.
		_ = e.conn.Send(req.ID, blocks, "")
		req.FirstTokenAt = time.Now()
		req.AdvanceDecode(1)
		e.decodeQ.Push(req)
	}
}

func (e *DisaggregatedEngine) decodeLoop(g *types.GPUInstance) {
	defer e.wg.Done()
	for {
		select {
		case <-e.stop:
			return
		default:
		}
		req, ok := e.decodeQ.Pop()
		if !ok {
			time.Sleep(200 * time.Microsecond)
			continue
		}
		// Pure decode: no prefill ever runs here, so ITL stays flat even
		// during arrival bursts.
		for !req.IsFinished() {
			itl := e.runner.DecodeUS(req.PromptTokens + req.Generated)
			time.Sleep(usDur(itl))
			req.AdvanceDecode(1)
		}
		finishRequest(e.reg, req, e.pool)
	}
}
