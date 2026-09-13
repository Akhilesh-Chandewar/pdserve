// Package elastic implements SLO-aware dynamic role assignment: a runtime
// monitor watches TTFT/TPOT tails and queue depths and flips stateless GPU
// instances between prefill and decode without reloading weights (the
// xLLM/LAPS style of elastic disaggregation).
package elastic

import (
	"fmt"
	"time"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/scheduler"
)

// Signal is the observation fed to the Policy each interval.
type Signal struct {
	// Now is the observation time.
	Now time.Time
	// QueueDepthPrefill is the prefill admission depth.
	QueueDepthPrefill int
	// TTFTP99us / TPOTP99us are recent-window p99 observations.
	TTFTP99us float64
	TPOTP99us float64
	// PrefillGPUs / DecodeGPUs are current pool sizes.
	PrefillGPUs int
	DecodeGPUs  int
	// MinDecodeGPUs is a floor so decode never starves completely.
	MinDecodeGPUs int
	// MinPrefillGPUs is a floor so TTFT cannot diverge.
	MinPrefillGPUs int
}

// Decision is the policy output for one interval.
type Decision struct {
	// MovePrefillToDecode > 0 means flip N prefill GPUs to decode.
	MovePrefillToDecode int
	// MoveDecodeToPrefill > 0 means flip N decode GPUs to prefill.
	MoveDecodeToPrefill int
	// Reason explains the decision for logs/metrics.
	Reason string
}

// String renders the decision.
func (d Decision) String() string {
	if d.MovePrefillToDecode == 0 && d.MoveDecodeToPrefill == 0 {
		return fmt.Sprintf("hold (%s)", d.Reason)
	}
	if d.MovePrefillToDecode > 0 {
		return fmt.Sprintf("prefill->decode x%d (%s)", d.MovePrefillToDecode, d.Reason)
	}
	return fmt.Sprintf("decode->prefill x%d (%s)", d.MoveDecodeToPrefill, d.Reason)
}

// Pressure thresholds as fractions of the SLO targets:
//   - >= breachRatio: objective breached, react.
//   - < idleRatio with empty queue: healthy with margin, return spare
//     capacity to decode (where token throughput lives).
const (
	breachRatio = 1.0
	stealRatio  = 0.5 // TTFT must be healthier than this before fixing TPOT
	idleRatio   = 0.4
)

// Policy decides role flips from SLO pressure. It is pure and deterministic,
// so it is unit-testable without a running engine.
//
// Direction semantics:
//   - TTFT breach or prefill backlog  => move decode -> prefill.
//   - TPOT breach (TTFT healthy/worse) => move prefill -> decode.
//   - Both healthy with margin        => return spare prefill to decode.
type Policy struct {
	SLO scheduler.SLO
	// HysteresisSec requires the same pressure to persist this long before
	// flipping (prevents oscillation).
	HysteresisSec float64
	// CooldownSec is the minimum gap between flips.
	CooldownSec float64

	lastFlip  time.Time
	pending   string
	pendSince time.Time
}

// NewPolicy builds a policy with hysteresis 2s and cooldown 5s by default.
func NewPolicy(slo scheduler.SLO) *Policy {
	return &Policy{SLO: slo, HysteresisSec: 2.0, CooldownSec: 5.0}
}

// Evaluate returns the decision for this interval.
func (p *Policy) Evaluate(s Signal) Decision {
	// Respect cooldown between flips.
	if !p.lastFlip.IsZero() && s.Now.Sub(p.lastFlip).Seconds() < p.CooldownSec {
		return Decision{Reason: "cooldown"}
	}

	ttftRatio := s.TTFTP99us / maxF(1, p.SLO.TTFTTargetUS)
	tpotRatio := s.TPOTP99us / maxF(1, p.SLO.TPOTTargetUS)

	pressure := ""
	switch {
	case ttftRatio >= breachRatio || s.QueueDepthPrefill > 0:
		pressure = "dec2pre" // prefill starved: pull a GPU from decode
	case tpotRatio >= breachRatio:
		if ttftRatio < stealRatio || tpotRatio > ttftRatio {
			pressure = "pre2dec" // decode starved: push a GPU to decode
		}
	case ttftRatio < idleRatio && tpotRatio < idleRatio && s.QueueDepthPrefill == 0:
		pressure = "pre2dec-idle" // healthy with margin: bias toward decode
	}

	if pressure == "" {
		p.pending = ""
		return Decision{Reason: "mixed-pressure"}
	}

	// Hysteresis: require sustained pressure before flipping.
	if pressure != p.pending {
		p.pending = pressure
		p.pendSince = s.Now
		return Decision{Reason: "pressure-detected"}
	}
	if s.Now.Sub(p.pendSince).Seconds() < p.HysteresisSec {
		return Decision{Reason: "hysteresis"}
	}

	d := Decision{}
	switch pressure {
	case "dec2pre":
		if s.DecodeGPUs-1 >= s.MinDecodeGPUs {
			d.MoveDecodeToPrefill = 1
		} else {
			return Decision{Reason: "decode-floor"}
		}
	case "pre2dec", "pre2dec-idle":
		if s.PrefillGPUs-1 >= s.MinPrefillGPUs {
			d.MovePrefillToDecode = 1
		} else {
			return Decision{Reason: "prefill-floor"}
		}
	}
	p.lastFlip = s.Now
	p.pending = ""
	return d
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
