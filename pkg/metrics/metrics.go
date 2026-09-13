// Package metrics provides a concurrency-safe metrics registry for serving
// benchmarks: latency histograms, TPOT, TTFT, ITL, queue depths, and token
// throughput. It also exposes a simple deterministic PRNG for reproducible
// simulations.
package metrics

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// Hist is a fixed-bucket latency histogram in microseconds.
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
	return &Hist{buckets: make([]uint64, numBuckets), width: bucketWidthUS, min: math.Inf(1), max: math.Inf(-1)}
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

// Percentile returns the p-th percentile (0..100) in microseconds.
func (h *Hist) Percentile(p float64) float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.count == 0 {
		return 0
	}
	target := (p / 100) * float64(h.count)
	var cum uint64
	for i, c := range h.buckets {
		cum += c
		if float64(cum) >= target {
			return (float64(i) + 0.5) * h.width
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

// Min returns the minimum observed latency in microseconds.
func (h *Hist) Min() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.min
}

// Max returns the maximum observed latency in microseconds.
func (h *Hist) Max() float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.max
}

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
	failed bool
}

// Registry aggregates per-request observations into a Snapshot.
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
	recent      []record
}

// NewRegistry creates a registry, starting its duration clock.
func NewRegistry() *Registry {
	return &Registry{
		ttft: NewHist(1000, 4096),
		tpot: NewHist(1000, 8192),
		itl:  NewHist(1000, 8192),
		e2e:  NewHist(5000, 8192),
	}
}

// Start marks the beginning of the measured window.
func (r *Registry) Start() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.start = time.Now()
}

// RecordRequest records a completed request with its per-phase timings.
// ttft, tpot and e2e are in microseconds; decodeTokens is the number of
// generated tokens (used for throughput accounting).
func (r *Registry) RecordRequest(ttft, tpot, e2e float64, decodeTokens int64, failed bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if failed {
		r.failed++
		return
	}
	r.completed++
	r.totalTokens += decodeTokens
	r.decodeTok += decodeTokens
	r.ttft.Observe(ttft)
	if decodeTokens > 0 {
		r.tpot.Observe(tpot)
	}
	r.itl.Observe(tpot)
	r.e2e.Observe(e2e)
	r.recent = append(r.recent, record{at: time.Now(), ttft: ttft, tpot: tpot, tokens: decodeTokens, failed: false})
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
// seconds (used by the SLO monitor for recent-pressure signals).
func (r *Registry) Window(windowSec float64) Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	dur := minF(windowSec, time.Since(r.start).Seconds())
	cutoff := time.Now().Add(-time.Duration(dur * float64(time.Second)))
	s := Snapshot{DurationSec: dur}
	var ttft, tpot []float64
	for _, rec := range r.recent {
		if rec.at.Before(cutoff) {
			continue
		}
		if rec.failed {
			s.Failed++
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

func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// Snapshot returns the aggregated results.
func (r *Registry) Snapshot() Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	dur := time.Since(r.start).Seconds()
	s := Snapshot{
		Completed:     r.completed,
		Failed:        r.failed,
		TTFTP50us:     r.ttft.Percentile(50),
		TTFTP99us:     r.ttft.Percentile(99),
		TPOTP50us:     r.tpot.Percentile(50),
		TPOTP99us:     r.tpot.Percentile(99),
		ITLP50us:      r.itl.Percentile(50),
		ITLP99us:      r.itl.Percentile(99),
		E2EP50us:      r.e2e.Percentile(50),
		E2EP99us:      r.e2e.Percentile(99),
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

// RNG is a deterministic random source for reproducible runs.
type RNG struct {
	mu *sync.Mutex
	r  *rand.Rand
}

// NewRNG builds a seeded RNG.
func NewRNG(seed int64) *RNG {
	return &RNG{mu: &sync.Mutex{}, r: rand.New(rand.NewSource(seed))}
}

// Float64 returns a uniform sample in [0,1).
func (g *RNG) Float64() float64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.r.Float64()
}

// Exp returns an exponentially distributed sample with the given mean.
func (g *RNG) Exp(mean float64) float64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.r.ExpFloat64() * mean
}

// Intn returns a uniform int in [0,n).
func (g *RNG) Intn(n int) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.r.Intn(n)
}

// Norm returns a normal sample with mean/stddev.
func (g *RNG) Norm(mean, stddev float64) float64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.r.NormFloat64()*stddev + mean
}

// SortUS is a helper for tests/tools that need sorted latency samples.
func SortUS(xs []float64) []float64 {
	out := append([]float64(nil), xs...)
	sort.Float64s(out)
	return out
}
