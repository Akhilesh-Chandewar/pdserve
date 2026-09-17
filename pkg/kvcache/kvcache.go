// Package kvcache models KV-cache blocks and the connectors that stream them
// between prefill and decode workers (the Mooncake/NIXL layer of pdserve).
package kvcache

import (
	"fmt"
	"sync"
)

// Block is one page of the paged KV cache.
type Block struct {
	ID        string
	RequestID string
	Seq       int
	Tokens    int
	Bytes     int
}

// Pool is a fixed-capacity paged KV-cache. Allocate evicts the oldest blocks
// of finished/evictable requests when full (ref-counted by Acquire/Release).
type Pool struct {
	mu        sync.Mutex
	capacity  int                 // total blocks
	blockSize int                 // tokens per block
	blocks    map[string]*Block   // blockID -> block
	byRequest map[string][]*Block // requestID -> its blocks
	order     []string            // requestIDs in first-allocation order (FIFO eviction)
	inUse     map[string]int
	nextID    int
}

// NewPool creates a pool with capacity blocks of blockSize tokens each.
func NewPool(capacity, blockSize int) *Pool {
	if capacity <= 0 {
		capacity = 1024
	}
	if blockSize <= 0 {
		blockSize = 16
	}
	return &Pool{
		capacity:  capacity,
		blockSize: blockSize,
		blocks:    map[string]*Block{},
		byRequest: map[string][]*Block{},
		inUse:     map[string]int{},
	}
}

// BlockSize returns tokens per block.
func (p *Pool) BlockSize() int { return p.blockSize }

// Capacity returns total block capacity.
func (p *Pool) Capacity() int { return p.capacity }

// Len returns currently allocated blocks.
func (p *Pool) Len() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.blocks)
}

// Acquire pins a request against eviction.
func (p *Pool) Acquire(requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.inUse[requestID]++
}

// Release unpins a request and frees all of its blocks.
func (p *Pool) Release(requestID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.inUse, requestID)
	p.evictLocked(requestID)
}

// evictLocked frees every block belonging to requestID and drops it from the
// eviction order. Safe to call for unknown requestIDs.
func (p *Pool) evictLocked(requestID string) {
	for _, b := range p.byRequest[requestID] {
		delete(p.blocks, b.ID)
	}
	delete(p.byRequest, requestID)
	for i, o := range p.order {
		if o == requestID {
			p.order = append(p.order[:i], p.order[i+1:]...)
			break
		}
	}
}

// Allocate creates nblocks blocks for a request, evicting unpinned requests'
// blocks (FIFO) if needed. Returns an error if the pool cannot satisfy it.
func (p *Pool) Allocate(requestID string, nblocks int) ([]*Block, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.blocks)+nblocks > p.capacity {
		for _, rid := range p.order {
			if len(p.blocks)+nblocks <= p.capacity {
				break
			}
			if _, pinned := p.inUse[rid]; pinned {
				continue
			}
			p.evictLocked(rid)
		}
		if len(p.blocks)+nblocks > p.capacity {
			return nil, fmt.Errorf("kv pool exhausted: %d/%d blocks in use", len(p.blocks), p.capacity)
		}
	}
	if len(p.byRequest[requestID]) == 0 {
		p.order = append(p.order, requestID)
	}
	out := make([]*Block, 0, nblocks)
	for i := 0; i < nblocks; i++ {
		p.nextID++
		b := &Block{
			ID:        fmt.Sprintf("blk-%08d", p.nextID),
			RequestID: requestID,
			Seq:       i,
			Tokens:    p.blockSize,
			Bytes:     p.blockSize * 2 * 128 * 2, // 2 (K+V) * heads*head_dim 128 * fp16
		}
		p.blocks[b.ID] = b
		p.byRequest[requestID] = append(p.byRequest[requestID], b)
		out = append(out, b)
	}
	return out, nil
}

// Connector transfers a KV-cache handle from a prefill worker to a decode
// worker. Implementations model the transfer cost analytically (P1-8):
// measuring loopback TCP says nothing about an RDMA fabric, so the sim path
// has no sockets at all.
type Connector interface {
	// TransferTimeUS returns modeled transfer latency in microseconds for
	// the given blocks.
	TransferTimeUS(blocks []*Block) float64
	// Name identifies the connector in benchmarks.
	Name() string
}

// MemoryConnector is a zero-copy same-process connector (models NVLink /
// shared pool). Transfer cost is ~0.
type MemoryConnector struct{}

// TransferTimeUS implements Connector.
func (MemoryConnector) TransferTimeUS([]*Block) float64 { return 0 }

// Name implements Connector.
func (MemoryConnector) Name() string { return "memory(zero-copy)" }

// TCPConnector models a network KV transfer (RDMA-class fabrics when
// bandwidth is high) as bytes / bandwidth. Purely analytic.
type TCPConnector struct {
	BandwidthGBps float64
}

// NewTCPConnector builds an analytic TCP-bandwidth connector.
func NewTCPConnector(bandwidthGBps float64) (*TCPConnector, error) {
	if bandwidthGBps <= 0 {
		return nil, fmt.Errorf("tcp connector bandwidth must be > 0, got %v", bandwidthGBps)
	}
	return &TCPConnector{BandwidthGBps: bandwidthGBps}, nil
}

// TransferTimeUS implements Connector.
func (c *TCPConnector) TransferTimeUS(blocks []*Block) float64 {
	return TransferTimeUS(blocks, c.BandwidthGBps)
}

// Name implements Connector.
func (c *TCPConnector) Name() string { return fmt.Sprintf("tcp(%.0f GB/s)", c.BandwidthGBps) }

// TransferTimeUS estimates transfer latency in microseconds for nblocks at
// the given fabric bandwidth.
func TransferTimeUS(blocks []*Block, bandwidthGBps float64) float64 {
	var bytes int64
	for _, b := range blocks {
		bytes += int64(b.Bytes)
	}
	if bandwidthGBps <= 0 {
		return 0
	}
	return float64(bytes) / (bandwidthGBps * 1e9) * 1e6 // seconds -> us
}
