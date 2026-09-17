// Package metrics provides a concurrency-safe metrics registry for serving
// benchmarks: latency histograms, TPOT/TTFT/ITL/E2E registries, windowed
// snapshots, and admission accounting with reason codes.
package metrics

import (
	"sort"
	"sync"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
)

// --- Histogram -------------------------------------------------------------

// Hist is a fixed-bucket latency histogram in microseconds. Callers must
// respect Resolution(): a percentile below 4× resolution is below the
// instrument's noise floor and must not be printed as a point estimate
// (spec rule 2, P0-1 fix 2–3).
type Hist struct {
	mu      sync.Mutex
	buckets []uint64 // bucket i covers [i*width, (i+1)*width) us
	width   float64
	count   uint64
	sum     float64 // us
	min     float64 // us
	max     float64 // us
}

// NewHist builds a histogram with buckets of the given microsecond width.
func NewHist(bucketWidthUS float64, numBuckets int) *Hist {
	if bucketWidthUS <= 0 {
		bucketWidthUS = 100
	}
	if numBuckets <= 0 {
		numBuckets = 1024
	}
	return &Hist{buckets: make([]uint64, numBuckets), width: bucketWidthUS, min: 1e18, max: -1}
}

// Observe records a latency sample.
func (h *Hist) Observe(us float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if us < 0 {
		us = 0
	}
	h.count++
	h.sum += us
	if us < h.min {
		h.min = us
	}
	if us > h.max {
		h.max = us
	}
	idx := int(us / h.width)
	if idx >= len(h.buckets) {
		idx = len(h.buckets) - 1
	}
	h.buckets[idx]++
}

// Count returns the number of observed samples.
func (h *Hist) Count() uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.count
}

// Percentile returns the p-th percentile (0..100) in microseconds,
// interpolated within the landing bucket (P0-1 fix 2).
func (h *Hist) Percentile(p float64) float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.percentileLocked(p)
}

func (h *Hist) percentileLocked(p float64) float64 {
	if h.count == 0 {
		return 0
	}
	target := (p / 100) * float64(h.count)
	if target < 1 {
		target = 1
	}
	var cum uint64
	for i, c := range h.buckets {
		prev := cum
		cum += c
		if float64(cum) >= target {
			// Fraction into bucket i needed to reach target.
			need := target - float64(prev)
			frac := 0.0
			if c > 0 {
				frac = need / float64(c)
				if frac > 1 {
					frac = 1
				}
			}
			return (float64(i) + frac) * h.width
		}
	}
	return h.max
}

// Mean returns the mean latency in microseconds.
func (h *Hist) Mean() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.count == 0 {
		return 0
	}
	return h.sum / float64(h.count)
}

// Min returns the minimum observed latency in microseconds, or +Inf when
// empty.
func (h *Hist) Min() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.min
}

// Max returns the maximum observed latency in microseconds, or -Inf when
// empty.
func (h *Hist) Max() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.max
}

// Resolution returns the bucket width in microseconds: the smallest
// difference this histogram can distinguish.
func (h *Hist) Resolution() float64 { return h.width }

// Resolved reports whether v is statistically above the noise floor
// (spec rule 2: never print a percentile below 4× resolution).
func (h *Hist) Resolved(v float64) bool { return v >= 4*h.width }

// --- Gauges ----------------------------------------------------------------

// Gauge is a simple concurrent counter/gauge (queue depths, busy workers).
type Gauge struct {
	mu sync.Mutex
	v  int64
}

// Add adds delta to the gauge.
func (g *Gauge) Add(delta int64) {
	g.mu.Lock()
	g.v += delta
	g.mu.Unlock()
}

// Value returns the current value.
func (g *Gauge) Value() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.v
}

// --- Snapshots and records ---------------------------------------------------

// Snapshot is an immutable view of aggregated run results.
type Snapshot struct {
	Completed     int64
	Failed        int64
	TTFTP50us     float64 // time to first token
	TTFTP99us     float64
	TPOTP50us     float64 // time per output token (decode itl)
	TPOTP99us     float64
	ITLP50us      float64 // inter-token latency
	ITLP99us      float64
	E2EP50us      float64 // end-to-end request latency
	E2EP99us      float64
	TotalTokens   int64
	DurationSec   float64
	TokensPerSec  float64
	PrefillTokens int64
	DecodeTokens  int64
}

// record is one raw per-request observation kept for windowed queries.
type record struct {
	at     time.Time
	ttft   float64
	tpot   float64
	tokens int64
}

// Registry aggregates per-request observations into a Snapshot. The registry
// is clock-aware: when built with NewRegistryOn it stamps observations with
// simulation time, so windowed SLO queries are meaningful in simulated runs
// and the wall-clock clock is never read (P1-1/P1-2).
type Registry struct {
	mu          sync.Mutex
	ttft        *Hist
	tpot        *Hist
	itl         *Hist
	e2e         *Hist
	completed   int64
	failed      int64
	totalTokens int64
	prefillTok  int64
	decodeTok   int64
	start       time.Time
	nowFn       func() time.Time
	recent      []record
	rejections  map[string]int64
	preemptions int64
}

// NewRegistry creates a wall-clock registry, starting its duration clock.
func NewRegistry() *Registry {
	r := &Registry{
		ttft:       NewHist(1000, 4096),
		tpot:       NewHist(250, 16384),
		itl:        NewHist(250, 16384),
		e2e:        NewHist(5000, 8192),
		rejections: map[string]int64{},
	}
	r.start = time.Now()
	r.nowFn = time.Now
	return r
}

// NewRegistryOn creates a registry bound to a clock (a Sim in simulated
// runs). The duration clock starts at the clock's current time.
func NewRegistryOn(clk clock.Clock) *Registry {
	r := &Registry{
		ttft:       NewHist(1000, 4096),
		tpot:       NewHist(250, 16384),
		itl:        NewHist(250, 16384),
		e2e:        NewHist(5000, 8192),
		rejections: map[string]int64{},
	}
	r.nowFn = clk.Now
	r.start = clk.Now()
	return r
}

// RecordRejection counts one admission rejection with a reason code
// ("kv_exhausted", "queue_full", ...), per P1-7.
func (r *Registry) RecordRejection(reason string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rejections[reason]++
}

// Rejections returns rejection counts by reason.
func (r *Registry) Rejections() map[string]int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]int64, len(r.rejections))
	for k, v := range r.rejections {
		out[k] = v
	}
	return out
}

// RecordPreemption counts one running request preempted (distinct from
// admission rejections, per P1-7).
func (r *Registry) RecordPreemption() {
	r.mu.Lock()
	r.preemptions++
	r.mu.Unlock()
}

// Preemptions returns the preemption count.
func (r *Registry) Preemptions() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.preemptions
}

// RecordRequest records a completed request with its per-phase timings.
// ttft, tpot and e2e are in microseconds; decodeTokens is the number of
// generated tokens (used for throughput accounting).
func (r *Registry) RecordRequest(ttft, tpot, e2e float64, decodeTokens int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.completed++
	r.totalTokens += decodeTokens
	r.decodeTok += decodeTokens
	r.ttft.Observe(ttft)
	if decodeTokens > 0 {
		r.tpot.Observe(tpot)
	}
	r.itl.Observe(tpot)
	r.e2e.Observe(e2e)
	r.recent = append(r.recent, record{at: r.nowFn(), ttft: ttft, tpot: tpot, tokens: decodeTokens})
	// Bound memory: keep the last 10k records for windowed queries.
	if len(r.recent) > 10_000 {
		r.recent = r.recent[len(r.recent)-10_000:]
	}
}

// RecordPrefillTokens accounts prompt tokens processed.
func (r *Registry) RecordPrefillTokens(n int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.prefillTok += n
}

// Window returns a snapshot of observations recorded in the last windowSec
// (in registry-time; simulation time under a Sim clock) — used by the SLO
// monitor for recent-pressure signals.
func (r *Registry) Window(windowSec float64) Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.nowFn()
	dur := windowSec
	elapsed := now.Sub(r.start).Seconds()
	if elapsed < dur {
		dur = elapsed
	}
	cutoff := now.Add(-time.Duration(dur * float64(time.Second)))
	s := Snapshot{DurationSec: dur}
	var ttft, tpot []float64
	for _, rec := range r.recent {
		if rec.at.Before(cutoff) {
			continue
		}
		s.Completed++
		s.TotalTokens += rec.tokens
		ttft = append(ttft, rec.ttft)
		tpot = append(tpot, rec.tpot)
	}
	s.TTFTP50us, s.TTFTP99us = percentiles(ttft)
	s.TPOTP50us, s.TPOTP99us = percentiles(tpot)
	if dur > 0 {
		s.TokensPerSec = float64(s.TotalTokens) / dur
	}
	return s
}

// percentiles returns p50/p99 of a sample slice.
func percentiles(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	sorted := SortUS(xs)
	idx := func(p float64) float64 {
		i := int(p / 100 * float64(len(sorted)-1))
		if i < 0 {
			i = 0
		}
		if i >= len(sorted) {
			i = len(sorted) - 1
		}
		return sorted[i]
	}
	return idx(50), idx(99)
}

// Snapshot returns the aggregated results.
func (r *Registry) Snapshot() Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	dur := r.nowFn().Sub(r.start).Seconds()
	s := Snapshot{
		Completed:     r.completed,
		Failed:        r.failed,
		TTFTP50us:     r.ttft.percentileLocked(50),
		TTFTP99us:     r.ttft.percentileLocked(99),
		TPOTP50us:     r.tpot.percentileLocked(50),
		TPOTP99us:     r.tpot.percentileLocked(99),
		ITLP50us:      r.itl.percentileLocked(50),
		ITLP99us:      r.itl.percentileLocked(99),
		E2EP50us:      r.e2e.percentileLocked(50),
		E2EP99us:      r.e2e.percentileLocked(99),
		TotalTokens:   r.totalTokens,
		PrefillTokens: r.prefillTok,
		DecodeTokens:  r.decodeTok,
		DurationSec:   dur,
	}
	if dur > 0 {
		s.TokensPerSec = float64(r.totalTokens) / dur
	}
	return s
}

// TTFT returns the underlying TTFT histogram (for resolution-aware
// reporting).
func (r *Registry) TTFT() *Hist { return r.ttft }

// TPOT returns the underlying TPOT histogram (for resolution-aware
// reporting).
func (r *Registry) TPOT() *Hist { return r.tpot }

// ITL returns the underlying ITL histogram (for resolution-aware reporting).
func (r *Registry) ITL() *Hist { return r.itl }

// E2E returns the underlying end-to-end histogram (for resolution-aware
// reporting).
func (r *Registry) E2E() *Hist { return r.e2e }

// SortUS is a helper for tests/tools that need sorted latency samples.
func SortUS(xs []float64) []float64 {
	out := append([]float64(nil), xs...)
	sort.Float64s(out)
	return out
}
