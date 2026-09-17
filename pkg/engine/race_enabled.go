//go:build race

package engine

// raceEnabled is true when the test binary is built with -race. The race
// detector's instrumentation slows the discrete-event loop several-fold, so
// wall-time assertions about the simulator itself are relaxed in race mode.
const raceEnabled = true
