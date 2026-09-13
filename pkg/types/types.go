// Package types defines the core domain types shared across pdserve.
package types

import (
	"fmt"
	"sync/atomic"
	"time"
)

// Phase represents the inference phase of a request.
type Phase int

const (
	// PhasePrefill processes the entire input prompt in parallel (compute-heavy).
	PhasePrefill Phase = iota
	// PhaseDecode generates output tokens autoregressively (memory-bandwidth-heavy).
	PhaseDecode
)

func (p Phase) String() string {
	switch p {
	case PhasePrefill:
		return "prefill"
	case PhaseDecode:
		return "decode"
	default:
		return "unknown"
	}
}

// Role is the role a GPU instance is currently serving.
type Role int

const (
	// RoleIdle means the instance holds no assigned role (elastic pools start here).
	RoleIdle Role = iota
	// RolePrefill means the instance is serving the prefill phase.
	RolePrefill
	// RoleDecode means the instance is serving the decode phase.
	RoleDecode
)

func (r Role) String() string {
	switch r {
	case RoleIdle:
		return "idle"
	case RolePrefill:
		return "prefill"
	case RoleDecode:
		return "decode"
	default:
		return "unknown"
	}
}

// InstanceState captures the health/busy state of a GPU instance.
type InstanceState int

const (
	// StateHealthy means the instance is alive and can accept work.
	StateHealthy InstanceState = iota
	// StateBusy means the instance is executing a batch.
	StateBusy
	// StateDraining means the instance is finishing in-flight work and will flip role.
	StateDraining
	// StateFailed means the instance is unreachable.
	StateFailed
)

func (s InstanceState) String() string {
	switch s {
	case StateHealthy:
		return "healthy"
	case StateBusy:
		return "busy"
	case StateDraining:
		return "draining"
	case StateFailed:
		return "failed"
	default:
		return "unknown"
	}
}

var instanceCounter uint64

// InstanceID returns a unique, monotonically increasing instance identifier.
func InstanceID() string {
	n := atomic.AddUint64(&instanceCounter, 1)
	return fmt.Sprintf("gpu-%04d", n)
}

// GPUInstance models one stateless GPU instance. In elastic disaggregation an
// instance may flip between RolePrefill and RoleDecode without reloading model
// weights (weights are assumed shared/paged from a common store).
type GPUInstance struct {
	ID    string
	Role  Role
	State InstanceState

	// WeightsLoaded reports whether model weights are resident. Elastic role
	// flips keep this true; a cold instance pays the load cost once.
	WeightsLoaded bool

	// BandwidthGBps is the HBM bandwidth used by the roofline model.
	BandwidthGBps float64
	// ComputeTFLOPs is the peak compute used by the roofline model.
	ComputeTFLOPs float64

	// LastRoleFlip records when the instance last changed role.
	LastRoleFlip time.Time
}
