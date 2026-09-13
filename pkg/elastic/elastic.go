// elastic.go implements ElasticEngine: PD disaggregation where a monitor
// flips stateless GPU instances between prefill and decode roles at runtime
// based on SLO pressure, without reloading weights.
package elastic

import (
	"fmt"
	"sync"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/engine"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/scheduler"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/types"
)

// MonitorInterval is how often the SLO monitor evaluates signals.
const MonitorInterval = 250 * time.Millisecond

// ElasticEngine is a DisaggregatedEngine whose pool split is dynamic.
type ElasticEngine struct {
	cfg engine.Config
	eng *engine.DisaggregatedEngine
	pol *Policy

	reg   *metrics.Registry
	mu    sync.Mutex
	flips []FlipEvent
	stop  chan struct{}
	wg    sync.WaitGroup

	closed bool
}

// FlipEvent records one role flip for observability.
type FlipEvent struct {
	At    time.Time
	From  types.Role
	To    types.Role
	Count int
}

// NewElastic wraps a DisaggregatedEngine with the SLO monitor.
func NewElastic(cfg engine.Config) (*ElasticEngine, error) {
	eng, err := engine.NewDisaggregated(cfg)
	if err != nil {
		return nil, err
	}
	return &ElasticEngine{
		cfg:  cfg,
		eng:  eng,
		pol:  NewPolicy(cfg.SLO),
		reg:  eng.Registry(),
		stop: make(chan struct{}),
	}, nil
}

// Name implements engine.Engine.
func (e *ElasticEngine) Name() string {
	return fmt.Sprintf("elastic(%s, flips=%d)", e.eng.Name(), len(e.Flips()))
}

// Registry implements engine.Engine.
func (e *ElasticEngine) Registry() *metrics.Registry { return e.eng.Registry() }

// Queue implements engine.Engine.
func (e *ElasticEngine) Queue() *scheduler.Queue { return e.eng.Queue() }

// PoolSizes implements engine.Engine.
func (e *ElasticEngine) PoolSizes() (int, int) { return e.eng.PoolSizes() }

// Flips returns recorded flip events.
func (e *ElasticEngine) Flips() []FlipEvent {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]FlipEvent(nil), e.flips...)
}

// Start implements engine.Engine; it also starts the monitor loop.
func (e *ElasticEngine) Start() {
	e.eng.Start()
	e.wg.Add(1)
	go e.monitorLoop()
}

// Stop drains and shuts down the engine and monitor.
func (e *ElasticEngine) Stop() {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return
	}
	e.closed = true
	close(e.stop)
	e.mu.Unlock()
	e.eng.Stop()
	e.wg.Wait()
}

func (e *ElasticEngine) monitorLoop() {
	defer e.wg.Done()
	tick := time.NewTicker(MonitorInterval)
	defer tick.Stop()
	for {
		select {
		case <-e.stop:
			return
		case <-tick.C:
			e.evaluate()
		}
	}
}

func (e *ElasticEngine) evaluate() {
	snap := e.eng.Registry().Snapshot()
	if snap.Completed < 8 {
		return
	}
	pre, dec := e.eng.PoolSizes()
	sig := Signal{
		Now:               time.Now(),
		QueueDepthPrefill: e.Queue().Len(),
		TTFTP99us:         snap.TTFTP99us,
		TPOTP99us:         snap.TPOTP99us,
		PrefillGPUs:       pre,
		DecodeGPUs:        dec,
		MinDecodeGPUs:     maxInt(1, e.cfg.NumGPUs/4),
		MinPrefillGPUs:    1,
	}
	d := e.pol.Evaluate(sig)
	if d.MovePrefillToDecode > 0 || d.MoveDecodeToPrefill > 0 {
		e.applyFlip(d)
	}
}

// applyFlip moves GPUs between pools and records the event. In production
// this is where a worker would drain, rebind its role, and rejoin the other
// pool with weights still resident. In the simulation we record the flip and
// delegate pool resizing to the disaggregated engine.
func (e *ElasticEngine) applyFlip(d Decision) {
	e.mu.Lock()
	flip := FlipEvent{At: time.Now(), Count: 1}
	if d.MovePrefillToDecode > 0 {
		flip.From = types.RolePrefill
		flip.To = types.RoleDecode
	} else {
		flip.From = types.RoleDecode
		flip.To = types.RolePrefill
	}
	e.flips = append(e.flips, flip)
	e.mu.Unlock()
	e.eng.ResizePools(d.MoveDecodeToPrefill, d.MovePrefillToDecode)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
