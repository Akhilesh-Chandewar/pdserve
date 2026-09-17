package elastic

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/engine"
)

func simElastic(t *testing.T, mod func(*engine.Config)) (*ElasticEngine, engine.Config) {
	t.Helper()
	cfg := engine.DefaultConfig()
	cfg.Clk = clock.NewSim()
	cfg.Load = engine.DefaultLoad()
	cfg.Load.DurationSec = 60
	cfg.Load.BurstAtSec = []float64{20}
	cfg.Load.BurstFactor = 6
	cfg.Load.BurstLenSec = 5
	if mod != nil {
		mod(&cfg)
	}
	e, err := NewElastic(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return e, cfg
}

// TestElasticDeterminismWithFlips: same seed + same clock ⇒ byte-identical
// snapshots and identical flip counts, and the burst scenario actually
// triggers flips (drain+rebind is exercised, not dead code). Also checks
// pool-size conservation: flips move capacity, never add it (P0-4).
func TestElasticDeterminismWithFlips(t *testing.T) {
	el, cfg := simElastic(t, func(c *engine.Config) { c.PrefillRatio = 0.5 })
	el.Start()
	el.Run()

	out1, err := json.Marshal(el.Registry().Snapshot())
	if err != nil {
		t.Fatal(err)
	}
	flips1 := len(el.Flips())
	if flips1 == 0 {
		t.Fatal("expected flips under burst")
	}
	pre, dec := el.PoolSizes()
	if pre+dec != cfg.NumGPUs {
		t.Fatalf("pool sizes sum=%d, want %d", pre+dec, cfg.NumGPUs)
	}

	// Second identical run must match byte-for-byte.
	el2, _ := simElastic(t, func(c *engine.Config) { c.PrefillRatio = 0.5 })
	el2.Start()
	el2.Run()
	out2, err := json.Marshal(el2.Registry().Snapshot())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out1, out2) {
		t.Fatalf("elastic snapshots differ for same seed:\n%s\n%s", out1, out2)
	}
	if len(el2.Flips()) != flips1 {
		t.Fatalf("flip counts differ: %d vs %d", flips1, len(el2.Flips()))
	}
}
