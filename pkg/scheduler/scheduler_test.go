package scheduler

import (
	"testing"
)

func TestQueueFIFO(t *testing.T) {
	q := NewQueue(nil)
	q.Push(&Request{ID: "a"})
	q.Push(&Request{ID: "b"})
	r, ok := q.Pop()
	if !ok || r.ID != "a" {
		t.Fatalf("first pop = %v, want a", r.ID)
	}
	r, ok = q.Pop()
	if !ok || r.ID != "b" {
		t.Fatalf("second pop = %v, want b", r.ID)
	}
	if _, ok := q.Pop(); ok {
		t.Fatal("empty queue should not pop")
	}
}

func TestRequestLifecycle(t *testing.T) {
	r := &Request{PromptTokens: 100, MaxOutput: 5}
	if r.IsFinished() {
		t.Fatal("fresh request not finished")
	}
	r.AdvanceDecode(5)
	if !r.IsFinished() {
		t.Fatal("request should be finished at MaxOutput")
	}
}

func TestQueueDropCount(t *testing.T) {
	q := NewQueue(nil)
	q.Drop()
	q.Drop()
	if q.Dropped() != 2 {
		t.Fatalf("dropped=%d, want 2", q.Dropped())
	}
}
