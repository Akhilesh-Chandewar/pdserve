package clock

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestSimAdvanceOrderAndNow(t *testing.T) {
	s := NewSim()
	var order []int
	s.AfterFunc(5*time.Microsecond, func() { order = append(order, 1) })
	s.AfterFunc(1*time.Microsecond, func() { order = append(order, 0) })
	s.AfterFunc(5*time.Microsecond, func() { order = append(order, 2) })
	for s.Advance() {
	}
	if len(order) != 3 || order[0] != 0 || order[1] != 1 || order[2] != 2 {
		t.Fatalf("order=%v, want [0 1 2]", order)
	}
	if !s.Now().Equal(time.Time{}.Add(5 * time.Microsecond)) {
		t.Fatalf("now=%v", s.Now())
	}
}

func TestSimTieBreakIsFIFO(t *testing.T) {
	s := NewSim()
	var seq []int
	for i := 0; i < 5; i++ {
		i := i
		s.AfterFunc(10*time.Microsecond, func() { seq = append(seq, i) })
	}
	for s.Advance() {
	}
	for i := 0; i < 5; i++ {
		if seq[i] != i {
			t.Fatalf("tie-break order broken: %v", seq)
		}
	}
}

func TestSimCancel(t *testing.T) {
	s := NewSim()
	var fired int32
	cancel := s.AfterFunc(time.Millisecond, func() { atomic.AddInt32(&fired, 1) })
	cancel()
	cancel() // double-cancel must be safe
	for s.Advance() {
	}
	if atomic.LoadInt32(&fired) != 0 {
		t.Fatal("cancelled event fired")
	}
}

func TestSimRunHorizon(t *testing.T) {
	s := NewSim()
	var count int32
	// Self-perpetuating 1ms chain.
	var chain func()
	chain = func() {
		atomic.AddInt32(&count, 1)
		s.AfterFunc(time.Millisecond, chain)
	}
	s.AfterFunc(0, chain)
	s.Run(10 * time.Millisecond)
	// Horizon: events at 0..10ms inclusive-ish; drain may complete one more.
	if n := atomic.LoadInt32(&count); n < 10 || n > 12 {
		t.Fatalf("chain fired %d times in 10ms, want ~11", n)
	}
}

func TestRealAfterFunc(t *testing.T) {
	r := NewReal()
	var fired int32
	done := make(chan struct{})
	r.AfterFunc(time.Millisecond, func() {
		atomic.AddInt32(&fired, 1)
		close(done)
	})
	<-done
	if atomic.LoadInt32(&fired) != 1 {
		t.Fatal("event did not fire exactly once")
	}
}
