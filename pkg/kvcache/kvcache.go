// Package kvcache models KV-cache blocks and the connectors that stream them
// between prefill and decode workers (the Mooncake/NIXL layer of pdserve).
package kvcache

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
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
// worker. Implementations: MemoryConnector (same process, zero-copy) and
// TCPConnector (network transfer, models RDMA cost via bandwidth).
type Connector interface {
	// Send blocks until the KV blocks have been "transmitted" to dst.
	Send(reqID string, blocks []*Block, dst string) error
	// Name identifies the connector in benchmarks.
	Name() string
}

// MemoryConnector is a zero-copy same-process connector (models NVLink /
// shared pool). Transfer cost is ~0.
type MemoryConnector struct{}

// Send implements Connector.
func (MemoryConnector) Send(string, []*Block, string) error { return nil }

// Name implements Connector.
func (MemoryConnector) Name() string { return "memory(zero-copy)" }

// TCPConnector streams blocks over TCP. Transfer time is
// bytes / bandwidthGBps, modeling RDMA-class fabrics when bandwidth is high.
type TCPConnector struct {
	BandwidthGBps float64
	listener      net.Listener
	addr          string
	wg            sync.WaitGroup
}

// NewTCPConnector starts a local listener for KV transfers.
func NewTCPConnector(bandwidthGBps float64) (*TCPConnector, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	c := &TCPConnector{BandwidthGBps: bandwidthGBps, listener: l, addr: l.Addr().String()}
	c.wg.Add(1)
	go c.serve()
	return c, nil
}

// Addr returns the listener address.
func (c *TCPConnector) Addr() string { return c.addr }

// Close shuts the listener down.
func (c *TCPConnector) Close() error {
	err := c.listener.Close()
	c.wg.Wait()
	return err
}

func (c *TCPConnector) serve() {
	defer c.wg.Done()
	for {
		conn, err := c.listener.Accept()
		if err != nil {
			return
		}
		go func(conn net.Conn) {
			defer conn.Close()
			br := bufio.NewReader(conn)
			var hdr [8]byte
			for {
				if _, err := io.ReadFull(br, hdr[:]); err != nil {
					return
				}
				n := binary.LittleEndian.Uint64(hdr[:])
				if _, err := io.CopyN(io.Discard, br, int64(n)); err != nil {
					return
				}
			}
		}(conn)
	}
}

// Send implements Connector by streaming serialized blocks over TCP.
func (c *TCPConnector) Send(reqID string, blocks []*Block, dst string) error {
	var conn net.Conn
	var err error
	if dst != "" {
		conn, err = net.DialTimeout("tcp", dst, 2*time.Second)
	} else {
		conn, err = net.DialTimeout("tcp", c.addr, 2*time.Second)
	}
	if err != nil {
		return fmt.Errorf("kv send dial: %w", err)
	}
	defer conn.Close()
	bw := bufio.NewWriter(conn)
	var buf [8]byte
	for _, b := range blocks {
		payload := make([]byte, b.Bytes)
		binary.LittleEndian.PutUint64(buf[:], uint64(len(payload)))
		if _, err := bw.Write(buf[:]); err != nil {
			return err
		}
		if _, err := bw.Write(payload); err != nil {
			return err
		}
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	return nil
}

// Name implements Connector.
func (c *TCPConnector) Name() string { return fmt.Sprintf("tcp(%.0f GB/s)", c.BandwidthGBps) }

// TransferTimeUS estimates transfer latency in microseconds for nblocks.
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
