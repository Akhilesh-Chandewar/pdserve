// load.go defines arrival processes: steady Poisson, bursts (traffic spikes
// that trigger elastic role flips), and ramps.
package engine

import (
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/scheduler"
)

// LoadSpec describes the arrival process used by ArrivalGen.
type LoadSpec struct {
	// Seed makes runs reproducible.
	Seed int64
	// RPS is the average arrival rate.
	RPS float64
	// DurationSec is how long arrivals are generated.
	DurationSec float64
	// BurstAtSec times at which load multiplies by BurstFactor.
	BurstAtSec []float64
	// BurstFactor is the multiplier applied during bursts.
	BurstFactor float64
	// BurstLenSec is how long each burst lasts.
	BurstLenSec float64
	// PromptTokensMean / Stddev: lognormal-ish prompt size distribution.
	PromptTokensMean int
	PromptTokensStd  int
	// OutputTokensMean / Std: uniform-ish output length distribution.
	OutputTokensMean int
	OutputTokensStd  int
}

// DefaultLoad returns a 60s run at 4 req/s with one 5s burst at t=20s.
func DefaultLoad() LoadSpec {
	return LoadSpec{
		Seed:             42,
		RPS:              4,
		DurationSec:      30,
		BurstAtSec:       []float64{20},
		BurstFactor:      6,
		BurstLenSec:      5,
		PromptTokensMean: 512,
		PromptTokensStd:  256,
		OutputTokensMean: 128,
		OutputTokensStd:  32,
	}
}

// ArrivalGen generates requests onto a queue following LoadSpec.
type ArrivalGen struct {
	q    *scheduler.Queue
	spec LoadSpec
	rng  *metrics.RNG
}

// NewArrivalGen binds a generator to an admission queue.
func NewArrivalGen(q *scheduler.Queue, spec LoadSpec) *ArrivalGen {
	return &ArrivalGen{q: q, spec: spec, rng: metrics.NewRNG(spec.Seed)}
}

// Pump emits all requests that should have arrived by time t (start-relative).
// Call it periodically from the engine's arrival loop.
func (g *ArrivalGen) Pump(now time.Time) {
	g.q.Push(g.Next(now))
}

// Next returns the batch of requests that arrived between the previous call
// and now. It is safe for concurrent use by one arrival loop.
func (g *ArrivalGen) Next(now time.Time) *scheduler.Request {
	return g.makeRequest(now)
}

// makeRequest samples prompt/output sizes and stamps arrival.
func (g *ArrivalGen) makeRequest(now time.Time) *scheduler.Request {
	prompt := g.rng.Norm(float64(g.spec.PromptTokensMean), float64(g.spec.PromptTokensStd))
	if prompt < 32 {
		prompt = 32
	}
	out := g.rng.Norm(float64(g.spec.OutputTokensMean), float64(g.spec.OutputTokensStd))
	if out < 1 {
		out = 1
	}
	return &scheduler.Request{
		ID:           scheduler.NewRequestID(),
		PromptTokens: int(prompt),
		MaxOutput:    int(out),
		Arrival:      now,
	}
}

// NextInterarrivalUS returns the next inter-arrival time in microseconds given
// the current second of the run (Poisson process modulated by bursts).
func (g *ArrivalGen) NextInterarrivalUS(runSec float64) float64 {
	rate := g.spec.RPS
	for _, b := range g.spec.BurstAtSec {
		if runSec >= b && runSec < b+g.spec.BurstLenSec {
			rate *= g.spec.BurstFactor
		}
	}
	if rate <= 0 {
		return 1e12 // effectively no arrivals
	}
	meanUS := 1e6 / rate
	return g.rng.Exp(meanUS)
}
