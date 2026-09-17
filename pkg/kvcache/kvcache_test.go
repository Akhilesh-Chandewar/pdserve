package kvcache

import (
	"testing"
)

func TestPoolAllocateAndCapacity(t *testing.T) {
	p := NewPool(4, 16)
	bs, err := p.Allocate("r1", 4)
	if err != nil {
		t.Fatalf("allocate: %v", err)
	}
	if len(bs) != 4 {
		t.Fatalf("got %d blocks, want 4", len(bs))
	}
	// r1 is unpinned, so r2's allocation evicts it (FIFO reclaim).
	bs2, err := p.Allocate("r2", 4)
	if err != nil {
		t.Fatalf("unpinned eviction should satisfy allocation: %v", err)
	}
	if len(bs2) != 4 {
		t.Fatalf("got %d blocks, want 4", len(bs2))
	}
	// A pool of pinned blocks is exhausted.
	p2 := NewPool(4, 16)
	p2.Allocate("a", 2)
	p2.Acquire("a")
	p2.Allocate("b", 2)
	p2.Acquire("b")
	if _, err := p2.Allocate("c", 1); err == nil {
		t.Fatal("expected exhaustion error for fully-pinned pool")
	}
}

func TestPoolEviction(t *testing.T) {
	p := NewPool(4, 16)
	if _, err := p.Allocate("r1", 4); err != nil {
		t.Fatal(err)
	}
	p.Release("r1") // unpinned + freed
	bs, err := p.Allocate("r2", 4)
	if err != nil {
		t.Fatalf("release should free blocks: %v", err)
	}
	if len(bs) != 4 {
		t.Fatalf("got %d blocks", len(bs))
	}
}

func TestPoolPinnedNotEvicted(t *testing.T) {
	p := NewPool(4, 16)
	if _, err := p.Allocate("r1", 4); err != nil {
		t.Fatal(err)
	}
	p.Acquire("r1")
	if _, err := p.Allocate("r2", 1); err == nil {
		t.Fatal("pinned blocks must not be evicted")
	}
}

func TestMemoryConnector(t *testing.T) {
	c := MemoryConnector{}
	if us := c.TransferTimeUS(nil); us != 0 {
		t.Fatalf("memory transfer cost = %f us, want 0", us)
	}
	if c.Name() == "" {
		t.Fatal("connector name empty")
	}
}

func TestTCPConnectorAnalytic(t *testing.T) {
	// P1-8: the connector is purely analytic now — no sockets.
	c, err := NewTCPConnector(50)
	if err != nil {
		t.Fatal(err)
	}
	blocks := make([]*Block, 4)
	for i := range blocks {
		blocks[i] = &Block{ID: "b", Tokens: 16, Bytes: 4096}
	}
	want := TransferTimeUS(blocks, 50)
	if got := c.TransferTimeUS(blocks); got != want {
		t.Fatalf("tcp transfer = %f us, want %f", got, want)
	}
	if _, err := NewTCPConnector(0); err == nil {
		t.Fatal("zero bandwidth must be rejected")
	}
}

func TestTransferTimeUS(t *testing.T) {
	blocks := []*Block{{Bytes: 1_000_000_000}} // 1GB
	us := TransferTimeUS(blocks, 50)           // 50 GB/s -> 20ms
	if us < 19_000 || us > 21_000 {
		t.Fatalf("transfer time %f us, want ~20000", us)
	}
	if TransferTimeUS(blocks, 0) != 0 {
		t.Fatal("zero bandwidth should give zero cost (guard)")
	}
}
