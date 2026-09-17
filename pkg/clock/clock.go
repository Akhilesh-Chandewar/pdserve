// Package clock provides the two execution substrates pdserve runs on.
//
// Sim is a deterministic discrete-event scheduler: timers are ordered by fire
// time and callbacks run synchronously, so a simulated benchmark completes in
// microseconds of wall time and is bit-reproducible for a given seed. Real
// maps timers onto wall-clock time for the same code paths.
//
// Engines, arrival generators, KV connectors and monitors must all schedule
// work through a Clock — no direct time.Sleep / time.Now / time.NewTimer in
// simulation code (P1-1).
package clock

import (
	"container/heap"
	"sync"
	"time"
)

// Clock is the timer/time interface simulation code is written against.
type Clock interface {
	// Now returns the current clock time.
	Now() time.Time
	// AfterFunc schedules fn to run once after d. It returns a cancel
	// function; calling it prevents fn from running (already-running or
	// fired timers cannot be cancelled). Cancel is safe to call twice.
	AfterFunc(d time.Duration, fn func()) (cancel func())
}

// Sim is a deterministic discrete-event clock. Events are a binary heap
// ordered by (fireTime, seq); synchronous callback execution removes wall-
// clock scheduling jitter from results entirely.
type Sim struct {
	mu     sync.Mutex
	now    time.Time
	events eventHeap
	seq    uint64
}

// NewSim returns a Sim clock starting at time.Time{}. Pass Start to begin the
// run at a non-zero epoch.
func NewSim() *Sim {
	return &Sim{now: time.Time{}}
}

// Now implements Clock.
func (s *Sim) Now() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.now
}

// AfterFunc implements Clock. On a Sim the callback runs when Advance reaches
// its fire time — never before Advance is called.
func (s *Sim) AfterFunc(d time.Duration, fn func()) func() {
	if d < 0 {
		d = 0
	}
	s.mu.Lock()
	s.seq++
	ev := &event{at: s.now.Add(d), seq: s.seq, fn: fn}
	heap.Push(&s.events, ev)
	s.mu.Unlock()
	var once sync.Once
	return func() { once.Do(ev.cancel) }
}

// Advance pops and fires the next scheduled event, advancing Now to its fire
// time. It returns false when no events remain (the run is complete).
func (s *Sim) Advance() bool {
	s.mu.Lock()
	if len(s.events) == 0 {
		s.mu.Unlock()
		return false
	}
	ev := heap.Pop(&s.events).(*event)
	if ev.canceled {
		s.mu.Unlock()
		return s.Advance()
	}
	s.now = ev.at
	fn := ev.fn
	s.mu.Unlock()
	fn()
	return true
}

// Run fires events until the clock passes until or events are exhausted,
// returning the final clock time. Work already in flight when the horizon is
// crossed may schedule completion events past until; Run lets those fire so
// engines drain cleanly.
func (s *Sim) Run(until time.Duration) time.Time {
	deadline := time.Time{}.Add(until)
	for {
		s.mu.Lock()
		empty := len(s.events) == 0
		next := time.Time{}
		if !empty {
			next = s.events[0].at
		}
		s.mu.Unlock()
		if empty || next.After(deadline) {
			return s.Now()
		}
		if !s.Advance() {
			return s.Now()
		}
	}
}

// Steps fires up to maxEvents events regardless of their fire time, for
// tests and tools that need bounded execution.
func (s *Sim) Steps(maxEvents int) {
	for i := 0; i < maxEvents; i++ {
		if !s.Advance() {
			return
		}
	}
}

// Len returns the number of pending events.
func (s *Sim) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.events)
}

type event struct {
	at       time.Time
	seq      uint64
	fn       func()
	canceled bool
}

func (e *event) cancel() { e.canceled = true }

type eventHeap []*event

func (h eventHeap) Len() int { return len(h) }
func (h eventHeap) Less(i, j int) bool {
	return h[i].at.Before(h[j].at) || (h[i].at.Equal(h[j].at) && h[i].seq < h[j].seq)
}
func (h eventHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *eventHeap) Push(x interface{}) { *h = append(*h, x.(*event)) }
func (h *eventHeap) Pop() interface{} {
	old := *h
	n := len(old)
	ev := old[n-1]
	*h = old[:n-1]
	return ev
}

// Real is a wall-clock Clock built on time.AfterFunc. Engines can move from
// Sim to Real without code changes.
type Real struct{}

// NewReal returns the wall-clock clock.
func NewReal() *Real { return &Real{} }

// Now implements Clock.
func (Real) Now() time.Time { return time.Now() }

// AfterFunc implements Clock.
func (Real) AfterFunc(d time.Duration, fn func()) func() {
	t := time.AfterFunc(d, fn)
	return func() { t.Stop() }
}
