package engine

import (
	"testing"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
)

func shortLoad() LoadSpec {
	l := DefaultLoad()
	l.DurationSec = 3
	l.RPS = 8
	l.BurstAtSec = []float64{1.5}
	l.BurstFactor = 3
	l.BurstLenSec = 1
	l.PromptTokensMean = 256
	l.OutputTokensMean = 32
	return l
}

func waitForCompletions(t *testing.T, reg *metrics.Registry, want int64, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if reg.Snapshot().Completed >= want {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("only %d completions after %s", reg.Snapshot().Completed, timeout)
}

func TestColocatedEndToEnd(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumGPUs = 4
	cfg.Load = shortLoad()
	e, err := NewColocated(cfg)
	if err != nil {
		t.Fatal(err)
	}
	e.Start()
	waitForCompletions(t, e.Registry(), 5, 30*time.Second)
	e.Stop()
	s := e.Registry().Snapshot()
	if s.Completed < 5 {
		t.Fatalf("completed=%d", s.Completed)
	}
	if s.TTFTP50us <= 0 {
		t.Fatal("ttft p50 not recorded")
	}
}

func TestDisaggregatedEndToEnd(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumGPUs = 4
	cfg.PrefillRatio = 0.25
	cfg.Load = shortLoad()
	e, err := NewDisaggregated(cfg)
	if err != nil {
		t.Fatal(err)
	}
	e.Start()
	waitForCompletions(t, e.Registry(), 5, 30*time.Second)
	e.Stop()
	s := e.Registry().Snapshot()
	if s.Completed < 5 {
		t.Fatalf("completed=%d", s.Completed)
	}
	pre, dec := e.PoolSizes()
	if pre != 1 || dec != 3 {
		t.Fatalf("pool sizes pre=%d dec=%d", pre, dec)
	}
}

func TestResizePools(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumGPUs = 4
	cfg.Load = shortLoad()
	e, err := NewDisaggregated(cfg)
	if err != nil {
		t.Fatal(err)
	}
	e.Start()
	time.Sleep(100 * time.Millisecond)
	e.ResizePools(1, 0)
	pre, dec := e.PoolSizes()
	if pre != 2 || dec != 2 {
		t.Fatalf("after resize pre=%d dec=%d, want 2/2", pre, dec)
	}
	e.ResizePools(0, 1)
	pre, dec = e.PoolSizes()
	if pre != 1 || dec != 3 {
		t.Fatalf("after resize pre=%d dec=%d, want 1/3", pre, dec)
	}
	e.Stop()
}

func TestSimRunnerRoofline(t *testing.T) {
	r := NewSimRunner(2000, 300)
	// Prefill 1024 tokens on 7B @ 35% MFU of 300 TFLOPs:
	// 2*7e9*1024 / (300*0.35*1e12) = ~136ms (realistic single-GPU prefill).
	p := r.PrefillUS(1024)
	if p < 130_000 || p > 145_000 {
		t.Fatalf("prefill=%f us, want ~136533", p)
	}
	// Decode at 4k ctx: 4096*16384 bytes / 2e12 B/s = ~33.5us
	d := r.DecodeUS(4096)
	if d < 25 || d > 45 {
		t.Fatalf("decode=%f us, want ~33", d)
	}
}
