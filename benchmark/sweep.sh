#!/usr/bin/env bash
# pdserve simulator sweep — produces benchmark/results/sim_results.csv
#
# M0 rewrite: the CLI emits NDJSON (one object per engine), parsed with jq.
# No fragile awk/sed scraping of human-formatted tables; engine names are
# carried in a dedicated `mode` field. Runs are deterministic (Sim clock),
# so the committed CSV is stable for a given seed set.
#
# Usage: bash benchmark/sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="benchmark/results/sim_results.csv"
mkdir -p benchmark/results
echo "mode,rps,burst_factor,connector,done,ttft_p99_s,tpot_p99_ms,e2e_p99_s,tok_per_sec" > "$OUT"

echo "### building binary once"
go build -o /tmp/pdserve-bench ./cmd/pdserve

# run_matrix emits one NDJSON line per engine into the CSV.
run_matrix () {
  local rps="$1" burst="$2"
  echo "### matrix: rps=$rps burst=${burst}x"
  /tmp/pdserve-bench -mode all -rps "$rps" -burst-factor "$burst" -dur 12 \
    -burst-at 6 -burst-len 3 -gpus 8 -kv memory -json \
  | jq -r '[.mode, (.rps|tostring), (.burst_factor|tostring), .connector, (.done|tostring), (.ttft_p99_s|tostring), (.tpot_p99_ms|tostring), (.e2e_p99_s|tostring), (.tok_per_sec|tostring)] | join(",")' \
  >> "$OUT"
}

# Burst sensitivity at fixed RPS (TPOT inflation vs burst pressure)
for b in 1 3 6 10; do
  run_matrix 8 "$b"
done

# Load sweep at fixed burst (TTFT p99 vs offered load)
for r in 2 4 8; do
  run_matrix "$r" 6
done

echo "### wrote $OUT"
cat "$OUT"
