// Package scheduler provides the request model, admission queues and
// scheduling policies shared by colocated and disaggregated engines.
package scheduler

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
)

var reqCounter uint64

// NewRequestID returns a unique request identifier.
func NewRequestID() string {
	return fmt.Sprintf("req-%08d", atomic.AddUint64(&reqCounter, 1))
}

// Request is a single inference request flowing through the system.
type Request struct {
	ID           string
	PromptTokens int
	MaxOutput    int

	Arrival time.Time
	// FirstTokenAt is set when the first decode token is produced.
	FirstTokenAt time.Time
	// FinishAt is set when the request completes.
	FinishAt time.Time

	// Generated counts decode tokens produced so far.
	Generated int

	// KVHandle identifies allocated KV blocks.
	KVHandle string

	mu sync.Mutex
}

// IsFinished reports whether the request produced all its output tokens.
func (r *Request) IsFinished() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.Generated >= r.MaxOutput
}

// AdvanceDecode appends n output tokens.
func (r *Request) AdvanceDecode(n int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Generated += n
}

// E2EUS returns end-to-end latency in microseconds.
func (r *Request) E2EUS() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return float64(r.FinishAt.Sub(r.Arrival)) / 1e3
}

// TTFTUS returns time-to-first-token in microseconds.
func (r *Request) TTFTUS() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.FirstTokenAt.IsZero() {
		return 0
	}
	return float64(r.FirstTokenAt.Sub(r.Arrival)) / 1e3
}

// Queue is a thread-safe FIFO admission queue with a depth gauge.
type Queue struct {
	mu      sync.Mutex
	items   []*Request
	depthG  *metrics.Gauge
	dropped int64
}

// NewQueue builds a queue bound to a metrics gauge (may be nil).
func NewQueue(g *metrics.Gauge) *Queue {
	return &Queue{depthG: g}
}

// Push enqueues a request.
func (q *Queue) Push(r *Request) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.items = append(q.items, r)
	if q.depthG != nil {
		q.depthG.Add(1)
	}
}

// Pop dequeues the next request, ok=false when empty.
func (q *Queue) Pop() (*Request, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil, false
	}
	r := q.items[0]
	q.items = q.items[1:]
	if q.depthG != nil {
		q.depthG.Add(-1)
	}
	return r, true
}

// Len returns current queue depth.
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

// Dropped returns number of dropped (rejected) requests.
func (q *Queue) Dropped() int64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.dropped
}

// Drop marks one request as rejected.
func (q *Queue) Drop() {
	q.mu.Lock()
	q.dropped++
	q.mu.Unlock()
}

// SLO holds latency objectives used by the elastic monitor.
type SLO struct {
	// TTFTTargetUS is the p99 time-to-first-token budget in microseconds.
	TTFTTargetUS float64
	// TPOTTargetUS is the p99 time-per-output-token budget in microseconds.
	TPOTTargetUS float64
	// TargetAttainment is the desired fraction of requests meeting both budgets.
	TargetAttainment float64
}

// DefaultSLO returns a sane default: TTFT p99 <= 2s, TPOT p99 <= 100ms, 95% attainment.
func DefaultSLO() SLO {
	return SLO{TTFTTargetUS: 2_000_000, TPOTTargetUS: 100_000, TargetAttainment: 0.95}
}

// String renders the SLO.
func (s SLO) String() string {
	return fmt.Sprintf("ttft_p99<=%.0fms tpot_p99<=%.0fms attainment>=%.0f%%",
		s.TTFTTargetUS/1000, s.TPOTTargetUS/1000, s.TargetAttainment*100)
}
