package elastic

import (
	"testing"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/scheduler"
)

func baseSignal(now time.Time) Signal {
	return Signal{
		Now:            now,
		PrefillGPUs:    2,
		DecodeGPUs:     6,
		MinDecodeGPUs:  2,
		MinPrefillGPUs: 1,
	}
}

func TestPolicyIdleRebalanceWhenHealthy(t *testing.T) {
	p := NewPolicy(scheduler.DefaultSLO())
	now := time.Now()
	s := baseSignal(now)
	s.TTFTP99us = 500_000 // 0.25x target: healthy with margin
	s.TPOTP99us = 30_000  // 0.3x target
	p.Evaluate(s)
	s.Now = now.Add(3 * time.Second)
	if d := p.Evaluate(s); d.MovePrefillToDecode != 1 {
		t.Fatalf("want idle rebalance prefill->decode, got %v", d)
	}
}

func TestPolicyHoldWhenMarginThin(t *testing.T) {
	p := NewPolicy(scheduler.DefaultSLO())
	now := time.Now()
	s := baseSignal(now)
	s.TTFTP99us = 1_000_000 // between idleRatio and stealRatio
	s.TPOTP99us = 30_000
	if d := p.Evaluate(s); d.MovePrefillToDecode != 0 || d.MoveDecodeToPrefill != 0 {
		t.Fatalf("want hold when margin thin, got %v", d)
	}
}

func TestPolicyTTFTBreachMovesDecodeToPrefill(t *testing.T) {
	p := NewPolicy(scheduler.DefaultSLO())
	now := time.Now()
	s := baseSignal(now)
	s.TTFTP99us = 3_000_000 // 1.5x target
	d := p.Evaluate(s)
	if d.MoveDecodeToPrefill != 0 {
		t.Fatalf("hysteresis not respected: %v", d)
	}
	s.Now = now.Add(3 * time.Second)
	d = p.Evaluate(s)
	if d.MoveDecodeToPrefill != 1 {
		t.Fatalf("want decode->prefill on TTFT breach, got %v", d)
	}
}

func TestPolicyTPOTBreachMovesPrefillToDecode(t *testing.T) {
	p := NewPolicy(scheduler.DefaultSLO())
	now := time.Now()
	s := baseSignal(now)
	s.TTFTP99us = 400_000 // 0.2x target: healthy enough to steal from
	s.TPOTP99us = 500_000 // 5x target
	p.Evaluate(s)
	s.Now = now.Add(3 * time.Second)
	d := p.Evaluate(s)
	if d.MovePrefillToDecode != 1 {
		t.Fatalf("want prefill->decode on TPOT breach, got %v", d)
	}
}

func TestPolicyPrefillQueueBacklog(t *testing.T) {
	p := NewPolicy(scheduler.DefaultSLO())
	now := time.Now()
	s := baseSignal(now)
	s.QueueDepthPrefill = 3
	p.Evaluate(s)
	s.Now = now.Add(3 * time.Second)
	if d := p.Evaluate(s); d.MoveDecodeToPrefill != 1 {
		t.Fatalf("want decode->prefill on backlog, got %v", d)
	}
}

func TestPolicyDecodeFloor(t *testing.T) {
	p := NewPolicy(scheduler.DefaultSLO())
	now := time.Now()
	s := baseSignal(now)
	s.TTFTP99us = 3_000_000
	s.DecodeGPUs = 2 // at floor
	p.Evaluate(s)
	s.Now = now.Add(3 * time.Second)
	if d := p.Evaluate(s); d.MoveDecodeToPrefill != 0 {
		t.Fatalf("must respect decode floor: %v", d)
	}
}

func TestPolicyPrefillFloor(t *testing.T) {
	p := NewPolicy(scheduler.DefaultSLO())
	now := time.Now()
	s := baseSignal(now)
	s.TTFTP99us = 400_000
	s.TPOTP99us = 500_000
	s.PrefillGPUs = 1 // at floor
	p.Evaluate(s)
	s.Now = now.Add(3 * time.Second)
	if d := p.Evaluate(s); d.MovePrefillToDecode != 0 {
		t.Fatalf("must respect prefill floor: %v", d)
	}
}

func TestPolicyCooldown(t *testing.T) {
	p := NewPolicy(scheduler.DefaultSLO())
	now := time.Now()
	s := baseSignal(now)
	s.TTFTP99us = 3_000_000
	p.Evaluate(s)
	s.Now = now.Add(3 * time.Second)
	if d := p.Evaluate(s); d.MoveDecodeToPrefill != 1 {
		t.Fatalf("want flip: %v", d)
	}
	if d := p.Evaluate(s); d.MoveDecodeToPrefill != 0 || d.MovePrefillToDecode != 0 {
		t.Fatalf("cooldown violated: %v", d)
	}
}
