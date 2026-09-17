// Command pdserve runs the serving simulation benchmark across engine
// strategies (colocated, PD-disaggregated, elastic).
//
// All runs execute on a deterministic discrete-event clock (pkg/clock.Sim):
// a 600-second simulated benchmark completes in milliseconds of wall time
// and is bit-reproducible for a given seed.
//
// Output is NDJSON (one JSON object per engine, jq-friendly) with -json, or
// a human-readable table otherwise.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/Akhilesh-Chandewar/pdserve/pkg/clock"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/elastic"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/engine"
	"github.com/Akhilesh-Chandewar/pdserve/pkg/metrics"
)

func main() {
	var (
		mode         = flag.String("mode", "all", "engines to run: colocated|disagg|elastic|all (comma-separated)")
		rps          = flag.Float64("rps", 0, "arrival rate (default: LoadSpec default 4)")
		burstFactor  = flag.Float64("burst-factor", 0, "burst multiplier (default: LoadSpec default 6; 0 keeps default)")
		burstAt      = flag.String("burst-at", "", "comma-separated burst start times in s (default: LoadSpec default)")
		burstLen     = flag.Float64("burst-len", 0, "burst duration in s (default: LoadSpec default)")
		dur          = flag.Float64("dur", 0, "arrival window in s (default: LoadSpec default 30)")
		gpus         = flag.Int("gpus", 0, "total GPU instances (default: 8)")
		kv           = flag.String("kv", "", "KV connector: memory|tcp (default: memory)")
		prefillRatio = flag.Float64("prefill-ratio", 0, "prefill pool fraction for disagg/elastic (default: 0.25)")
		seed         = flag.Int64("seed", 0, "arrival RNG seed (default: LoadSpec default 42)")
		jsonOut      = flag.Bool("json", false, "emit NDJSON results (one object per engine)")
	)
	flag.Parse()

	cfg := engine.DefaultConfig()
	applyLoad(&cfg.Load, loadOverrides{
		rps: *rps, burstFactor: *burstFactor, burstAt: *burstAt, burstLen: *burstLen,
		dur: *dur, seed: *seed,
	})
	if *gpus > 0 {
		cfg.NumGPUs = *gpus
	}
	switch *kv {
	case "":
		// keep default
	case "memory", "tcp":
		cfg.KVConnector = *kv
	default:
		fatal("-kv must be memory or tcp")
	}
	if *prefillRatio > 0 {
		cfg.PrefillRatio = *prefillRatio
	}

	modes := parseModes(*mode)
	cfg.Clk = clock.NewSim()

	results := make([]*result, 0, len(modes))
	for _, m := range modes {
		res := runMode(m, cfg)
		results = append(results, res)
	}

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		for _, r := range results {
			if err := enc.Encode(r); err != nil {
				fatal(err.Error())
			}
		}
		return
	}
	printTable(results)
}

// loadOverrides carries non-zero flag overrides for the LoadSpec.
type loadOverrides struct {
	rps, burstFactor, burstLen, dur float64
	burstAt                         string
	seed                            int64
}

func applyLoad(l *engine.LoadSpec, o loadOverrides) {
	if o.rps > 0 {
		l.RPS = o.rps
	}
	if o.burstFactor > 0 {
		l.BurstFactor = o.burstFactor
	}
	if o.burstLen > 0 {
		l.BurstLenSec = o.burstLen
	}
	if o.dur > 0 {
		l.DurationSec = o.dur
	}
	if o.seed != 0 {
		l.Seed = o.seed
	}
	if o.burstAt != "" {
		var times []float64
		for _, part := range strings.Split(o.burstAt, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			v, err := strconv.ParseFloat(part, 64)
			if err != nil {
				fatal(fmt.Sprintf("bad -burst-at value %q: %v", part, err))
			}
			times = append(times, v)
		}
		l.BurstAtSec = times
	}
}

func parseModes(s string) []string {
	var out []string
	for _, part := range strings.Split(strings.ToLower(s), ",") {
		part = strings.TrimSpace(part)
		switch part {
		case "":
			continue
		case "all":
			out = append(out, "colocated", "disagg", "elastic")
		case "colocated", "disagg", "elastic":
			out = append(out, part)
		default:
			fatal(fmt.Sprintf("unknown mode %q (want colocated|disagg|elastic|all)", part))
		}
	}
	if len(out) == 0 {
		fatal("no modes selected")
	}
	return out
}

// result is one engine's NDJSON row. Percentile fields respect the
// instrument's resolution: when a p99 falls below 4x the histogram bucket
// width it is reported as resolved=false and the human table prints a
// "<floor" marker instead of a number (spec rule 2).
type result struct {
	Engine   string  `json:"engine"`
	Mode     string  `json:"mode"`
	Done     int64   `json:"done"`
	TTFTP99s float64 `json:"ttft_p99_s"`
	TPOTP99  float64 `json:"tpot_p99_ms"`
	// TPOTResolved reports whether tpot_p99_ms is above the histogram's
	// noise floor (4x bucket width).
	TPOTResolved bool             `json:"tpot_p99_resolved"`
	E2EP99s      float64          `json:"e2e_p99_s"`
	TokPerSec    float64          `json:"tok_per_sec"`
	Rejections   map[string]int64 `json:"rejections"`
	Flips        int              `json:"flips"`
	DurationSec  float64          `json:"duration_sec"`
	Seed         int64            `json:"seed"`
	RPS          float64          `json:"rps"`
	BurstFactor  float64          `json:"burst_factor"`
	Connector    string           `json:"connector"`
	GPUs         int              `json:"gpus"`
	PrefillRatio float64          `json:"prefill_ratio"`
}

func runMode(mode string, cfg engine.Config) *result {
	var (
		eng engine.Engine
		err error
	)
	switch mode {
	case "colocated":
		eng, err = engine.NewColocated(cfg)
	case "disagg":
		eng, err = engine.NewDisaggregated(cfg)
	case "elastic":
		var el *elastic.ElasticEngine
		el, err = elastic.NewElastic(cfg)
		if err == nil {
			eng = el
		}
	default:
		fatal("unreachable mode " + mode)
	}
	if err != nil {
		fatal(err.Error())
	}
	eng.Start()
	eng.Run() // Sim clock: deterministic, instant

	snap := eng.Registry().Snapshot()
	flips := 0
	if el, ok := eng.(interface{ Flips() []elastic.FlipEvent }); ok {
		flips = len(el.Flips())
	}

	tpot := snap.TPOTP99us / 1e3
	return &result{
		Engine:       eng.Name(),
		Mode:         mode,
		Done:         snap.Completed,
		TTFTP99s:     snap.TTFTP99us / 1e6,
		TPOTP99:      tpot,
		TPOTResolved: eng.Registry().TPOT().Resolved(snap.TPOTP99us),
		E2EP99s:      snap.E2EP99us / 1e6,
		TokPerSec:    snap.TokensPerSec,
		Rejections:   eng.Registry().Rejections(),
		Flips:        flips,
		DurationSec:  snap.DurationSec,
		Seed:         cfg.Load.Seed,
		RPS:          cfg.Load.RPS,
		BurstFactor:  cfg.Load.BurstFactor,
		Connector:    connectorName(cfg),
		GPUs:         cfg.NumGPUs,
		PrefillRatio: cfg.PrefillRatio,
	}
}

func connectorName(cfg engine.Config) string {
	if cfg.KVConnector == "tcp" {
		return fmt.Sprintf("tcp(%.0f GB/s)", cfg.KVTCPBandwidthGBps)
	}
	return "memory(zero-copy)"
}

func printTable(results []*result) {
	fmt.Printf("%-34s %7s %12s %14s %12s %10s\n",
		"engine", "done", "ttft_p99(s)", "tpot_p99(ms)", "e2e_p99(s)", "tok/s")
	for _, r := range results {
		tpot := fmt.Sprintf("%.3f", r.TPOTP99)
		if !r.TPOTResolved {
			tpot = fmt.Sprintf("<%.3f", metrics.NewHist(250, 1).Resolution()/1e3*4)
		}
		fmt.Printf("%-34s %7d %12.3f %14s %12.3f %10.1f\n",
			r.Engine, r.Done, r.TTFTP99s, tpot, r.E2EP99s, r.TokPerSec)
	}
	if rej := totalRejections(results); len(rej) > 0 {
		fmt.Printf("\nrejections by reason: %v\n", rej)
	}
	fmt.Printf("\n(simulated runs on a deterministic clock; percentiles below 4x histogram\n resolution print as <floor. Seed the physics caveats: see README.)\n")
}

func totalRejections(results []*result) map[string]int64 {
	out := map[string]int64{}
	for _, r := range results {
		for k, v := range r.Rejections {
			out[k] += v
		}
	}
	return out
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, "pdserve:", msg)
	os.Exit(1)
}
