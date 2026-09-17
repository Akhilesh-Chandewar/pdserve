// elastic.go implements ElasticEngine: PD disaggregation where a monitor
// flips stateless GPU instances between prefill and decode roles at runtime
// based on SLO pressure, without reloading weights.
package elastic

import (
	"fmt"
	"sync"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
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

// Engine exposes the wrapped engine (flip primitive, drain model).
func (e *ElasticEngine) Engine() *engine.DisaggregatedEngine { return e.eng }

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

// InFlight implements engine.Engine.
func (e *ElasticEngine) InFlight() int { return e.eng.InFlight() }

// Flips returns recorded flip events.
func (e *ElasticEngine) Flips() []FlipEvent {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]FlipEvent(nil), e.flips...)
}

// Start implements engine.Engine; it also starts the clock-driven monitor.
// The monitor is a self-perpetuating timer chain on the engine's clock: it
// fires as part of the same event stream as arrivals and workers, so Sim runs
// stay deterministic and Real runs need no extra goroutine.
func (e *ElasticEngine) Start() {
	e.eng.Start()
	e.scheduleTick(e.eng.Clock())
}

// Run implements engine.Engine by delegating to the wrapped engine.
func (e *ElasticEngine) Run() { e.eng.Run() }

// RunUntil implements engine.Engine by delegating to the wrapped engine.
func (e *ElasticEngine) RunUntil(d time.Duration) { e.eng.RunUntil(d) }

// scheduleTick schedules the next monitor evaluation. Stop breaks the chain;
// the chain also retires once the engine is quiescent (arrivals done, nothing
// in flight or queued), so Sim runs reach the empty-queue state and drain.
func (e *ElasticEngine) scheduleTick(clk clock.Clock) {
	clk.AfterFunc(MonitorInterval, func() {
		if e.isStopped() {
			return
		}
		e.evaluate()
		if e.eng.ArrivalsDone() && e.eng.InFlight() == 0 && e.Queue().Len() == 0 {
			return // quiescent: retire the monitor chain
		}
		e.scheduleTick(clk)
	})
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
}

// isStopped reports whether Stop has been called.
func (e *ElasticEngine) isStopped() bool {
	select {
	case <-e.stop:
		return true
	default:
		return false
	}
}

// evaluate samples the current window and applies the policy decision.
func (e *ElasticEngine) evaluate() {
	snap := e.eng.Registry().Snapshot()
	if snap.Completed < 8 {
		return
	}
	pre, dec := e.eng.PoolSizes()
	sig := Signal{
		Now:               e.eng.Clock().Now(),
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

// applyFlip moves GPUs between pools and records the event. The underlying
// ResizePools drains each flipped worker's in-flight step before rebinding
// (P0-4 fix) and charges the modeled drain latency as simulated time.
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
