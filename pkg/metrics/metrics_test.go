package metrics

import (
	"testing"
	"time"
)

func TestHistPercentiles(t *testing.T) {
	h := NewHist(1000, 1024)
	for i := 1; i <= 100; i++ {
		h.Observe(float64(i) * 1000) // 1ms..100ms
	}
	if got := h.Percentile(50); got < 49_000 || got > 52_000 {
		t.Fatalf("p50=%f, want ~50000us", got)
	}
	if got := h.Percentile(99); got < 98_000 || got > 101_000 {
		t.Fatalf("p99=%f, want ~100000us", got)
	}
}

func TestRegistrySnapshot(t *testing.T) {
	r := NewRegistry()
	r.Start()
	r.RecordPrefillTokens(100)
	r.RecordRequest(1_000_000, 50_000, 7_000_000, 128, false)
	r.RecordRequest(0, 0, 0, 0, true)
	s := r.Snapshot()
	if s.Completed != 1 || s.Failed != 1 {
		t.Fatalf("completed=%d failed=%d", s.Completed, s.Failed)
	}
	if s.TTFTP50us < 999_000 || s.TTFTP50us > 1_100_000 {
		t.Fatalf("ttft p50=%f", s.TTFTP50us)
	}
	if s.PrefillTokens != 100 {
		t.Fatalf("prefill tokens=%d", s.PrefillTokens)
	}
}

func TestRegistryWindow(t *testing.T) {
	r := NewRegistry()
	r.Start()
	r.RecordRequest(100_000, 10_000, 500_000, 10, false)
	w := r.Window(5)
	if w.Completed != 1 {
		t.Fatalf("window completed=%d, want 1", w.Completed)
	}
	// A window far in the past excludes everything.
	time.Sleep(10 * time.Millisecond)
	w2 := r.Window(0.005)
	if w2.Completed != 0 {
		t.Fatalf("stale window completed=%d, want 0", w2.Completed)
	}
}

func TestRNGDeterministic(t *testing.T) {
	a := NewRNG(7)
	b := NewRNG(7)
	for i := 0; i < 100; i++ {
		if a.Float64() != b.Float64() {
			t.Fatal("same seed must produce identical sequences")
		}
	}
}
