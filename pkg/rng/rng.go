// Package rng provides the deterministic random source for simulations.
// Randomness does not belong in a metrics package (P1-5), so the PRNG moved
// here from pkg/metrics.
package rng

import (
	"math/rand"
	"sync"
)

// RNG is a mutex-serialized deterministic random source. Serialize every draw
// so call order fully determines the stream — a requirement for the
// byte-identical reproducibility test (spec M1 acceptance).
type RNG struct {
	mu sync.Mutex
	r  *rand.Rand
}

// NewRNG builds a seeded RNG.
func NewRNG(seed int64) *RNG {
	return &RNG{r: rand.New(rand.NewSource(seed))}
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

// Intn returns a uniform int in [0,n). It panics when n <= 0, matching
// math/rand semantics.
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
