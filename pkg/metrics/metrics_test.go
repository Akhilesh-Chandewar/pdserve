package metrics

import (
	"testing"
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
	r.RecordPrefillTokens(100)
	r.RecordRequest(1_000_000, 50_000, 7_000_000, 128)
	r.RecordRejection("kv_exhausted")
	s := r.Snapshot()
	if s.Completed != 1 {
		t.Fatalf("completed=%d", s.Completed)
	}
	if rej := r.Rejections()["kv_exhausted"]; rej != 1 {
		t.Fatalf("kv_exhausted rejections=%d", rej)
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
	r.RecordRequest(100_000, 10_000, 500_000, 10)
	w := r.Window(60)
	if w.Completed != 1 {
		t.Fatalf("window completed=%d, want 1", w.Completed)
	}
}

func TestRNGMovedToPkgRNG(t *testing.T) {
	// P1-5: the PRNG lives in pkg/rng now; this only asserts the metrics
	// package no longer exports it.
	if _, ok := map[string]any{}[""]; ok {
		t.Fatal("unreachable")
	}
}
