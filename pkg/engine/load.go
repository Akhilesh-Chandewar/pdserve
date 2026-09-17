// load.go defines arrival processes: steady Poisson, bursts (traffic spikes
// that trigger elastic role flips), and ramps.
package engine

import (
	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/rng"
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
	// PromptTokensMean / Std: prompt size distribution.
	PromptTokensMean int
	PromptTokensStd  int
	// OutputTokensMean / Std: output length distribution.
	OutputTokensMean int
	OutputTokensStd  int
}

// DefaultLoad returns a 30s run at 4 req/s with one 5s burst at t=20s.
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

// ArrivalGen generates requests following LoadSpec. All randomness flows
// through one serialized RNG and all time flows through a Clock, so a given
// seed + clock produces a fully determined arrival stream (P1-2).
type ArrivalGen struct {
	spec LoadSpec
	rng  *rng.RNG
	clk  clock.Clock
}

// NewArrivalGen binds a generator to a clock.
func NewArrivalGen(clk clock.Clock, spec LoadSpec) *ArrivalGen {
	return &ArrivalGen{spec: spec, rng: rng.NewRNG(spec.Seed), clk: clk}
}

// makeRequest samples prompt/output sizes and stamps arrival at current
// clock time.
func (g *ArrivalGen) makeRequest() *scheduler.Request {
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
		Arrival:      g.clk.Now(),
	}
}

// Next returns the request that arrived at the current clock instant. The
// engine drives arrivals by scheduling timers at each sampled inter-arrival
// time (not by polling), so Next is called exactly once per arrival event and
// the RNG stream is fully determined by the seed.
func (g *ArrivalGen) Next() *scheduler.Request {
	return g.makeRequest()
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
