package engine

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
)

func simConfig(mod func(*Config)) Config {
	cfg := DefaultConfig()
	cfg.Clk = clock.NewSim()
	cfg.Load = DefaultLoad()
	cfg.Load.DurationSec = 60
	mod(&cfg)
	return cfg
}

func snapshotJSON(t *testing.T, e Engine) []byte {
	t.Helper()
	s := e.Registry().Snapshot()
	out, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestDeterminism is the M1 acceptance gate: two runs with the same seed and
// clock must produce byte-identical metric snapshots (P1-2).
func TestDeterminism(t *testing.T) {
	for _, tc := range []struct {
		name string
		mk   func(Config) (Engine, error)
		mod  func(*Config)
	}{
		{"colocated", func(c Config) (Engine, error) { return NewColocated(c) }, nil},
		{"disagg", func(c Config) (Engine, error) { return NewDisaggregated(c) }, nil},
		{"disagg-tcp", func(c Config) (Engine, error) { return NewDisaggregated(c) }, func(c *Config) { c.KVConnector = "tcp" }},
		{"disagg-noburst", func(c Config) (Engine, error) { return NewDisaggregated(c) }, func(c *Config) { c.Load.BurstAtSec = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mkCfg := func() Config {
				var c Config
				if tc.mod != nil {
					c = simConfig(tc.mod)
				} else {
					c = simConfig(func(*Config) {})
				}
				return c
			}
			e1, err := tc.mk(mkCfg())
			if err != nil {
				t.Fatal(err)
			}
			e1.Start()
			e1.Run()

			e2, err := tc.mk(mkCfg())
			if err != nil {
				t.Fatal(err)
			}
			e2.Start()
			e2.Run()

			b1, b2 := snapshotJSON(t, e1), snapshotJSON(t, e2)
			if !bytes.Equal(b1, b2) {
				t.Fatalf("snapshots differ for same seed:\n%s\n%s", b1, b2)
			}
			s := e1.Registry().Snapshot()
			if s.Completed == 0 {
				t.Fatal("expected completions")
			}
		})
	}
}

// TestSimSpeed is the second M1 acceptance gate: a 600-second simulated run
// must complete in well under 1 second of wall time.
func TestSimSpeed(t *testing.T) {
	cfg := simConfig(func(c *Config) {
		c.Load.DurationSec = 600
		c.Load.RPS = 16
	})
	start := time.Now()
	e, err := NewDisaggregated(cfg)
	if err != nil {
		t.Fatal(err)
	}
	e.Start()
	e.Run()
	elapsed := time.Since(start)
	// The race detector instruments the hot event loop and slows it
	// several-fold; the <1s gate applies to normal builds.
	limit := time.Second
	if raceEnabled {
		limit = 30 * time.Second
	}
	if elapsed > limit {
		t.Fatalf("600s simulated run took %s, want <%s", elapsed, limit)
	}
	s := e.Registry().Snapshot()
	if s.Completed < 1000 {
		t.Fatalf("completed=%d, want a full 600s run's worth", s.Completed)
	}
	if s.DurationSec < 590 {
		t.Fatalf("simulated duration=%f s, want >=590", s.DurationSec)
	}
}

// TestChainsRetire asserts Run reaches quiescence: queues empty, nothing in
// flight, workers retired.
func TestChainsRetire(t *testing.T) {
	cfg := simConfig(func(*Config) {})
	e, err := NewColocated(cfg)
	if err != nil {
		t.Fatal(err)
	}
	e.Start()
	e.Run()
	if n := e.InFlight(); n != 0 {
		t.Fatalf("in-flight=%d after Run, want 0", n)
	}
	if n := e.Queue().Len(); n != 0 {
		t.Fatalf("queue depth=%d after Run, want 0", n)
	}
}
