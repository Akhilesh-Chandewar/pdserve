#!/usr/bin/env bash
# pdserve simulator sweep — produces benchmark/results/sim_results.csv
# Usage: bash benchmark/sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="benchmark/results/sim_results.csv"
mkdir -p benchmark/results
echo "mode,rps,burst_factor,connector,done,ttft_p99_s,tpot_p50_ms,tpot_p99_ms,e2e_p99_s,tok_per_sec" > "$OUT"

echo "### building binary once"
go build -o /tmp/pdserve-bench ./cmd/pdserve

run_matrix () {
  local rps="$1" burst="$2"
  echo "### matrix: rps=$rps burst=${burst}x"
  /tmp/pdserve-bench -mode all -rps "$rps" -burst-factor "$burst" -dur 12 \
    -burst-at 6 -burst-len 3 -gpus 8 -kv memory \
  | awk '/^=============== RESULTS/{f=1;next} f' \
  | grep -v '^engine ' \
  | sed -E 's/disagg\(memory\(zero-copy\), pre=2\/dec=6\)/disagg_pre2_dec6/g; s/elastic\(disagg\(memory\(zero-copy\), pre=2\/dec=6\), flips=[0-9]+\)/elastic/g; s/elastic\(disagg_pre2_dec6, flips=[0-9]+\)/elastic/g' \
  | while IFS= read -r line; do
      [ -z "$(echo "$line" | tr -d ' ')" ] && continue
      case "$line" in engine*) continue;; esac
      name=$(echo "$line" | sed -E 's/ +[0-9][0-9. ]*$//' | xargs)
      vals=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/) {printf "%s%s", $i, (i<NF?",":"")}}')
      echo "${name},${rps},${burst},memory,${vals}" >> "$OUT"
    done
}

# Burst sensitivity at fixed RPS (TPOT inflation vs burst pressure)
for b in 1 3 6 10; do
  run_matrix 8 "$b"
done

# Load sweep at fixed burst (TTFT p99 vs offered load)
for r in 2 4; do
  run_matrix "$r" 6
done

echo "### wrote $OUT"
cat "$OUT"
