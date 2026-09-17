//go:build !race

package engine

// raceEnabled is false in normal builds; see race_enabled.go.
const raceEnabled = false
