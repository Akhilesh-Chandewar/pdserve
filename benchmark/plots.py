#!/usr/bin/env python3
"""Generate benchmark charts from benchmark/results/sim_results.csv."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results", "sim_results.csv")
OUT = os.path.join(HERE, "charts")
os.makedirs(OUT, exist_ok=True)

def family(name):
    if name.startswith("colocated"):
        return "colocated"
    if name.startswith("elastic"):
        return "elastic"
    return "disagg"

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        if not r.get("mode"):
            continue
        rows.append({
            "family": family(r["mode"]),
            "mode": r["mode"],
            "rps": float(r["rps"]),
            "burst": float(r["burst_factor"]),
            "done": int(r["done"]),
            "ttft99": float(r["ttft_p99_s"]),
            "tpot99": float(r["tpot_p99_ms"]),
            "tps": float(r["tok_per_sec"]),
        })

COLORS = {"colocated": "#d62728", "disagg": "#1f77b4", "elastic": "#2ca02c"}
LABELS = {"colocated": "Colocated", "disagg": "PD Disaggregated", "elastic": "Elastic"}

def rowsat(fam, rps):
    return sorted([r for r in rows if r["family"] == fam and r["rps"] == rps], key=lambda r: r["burst"])

# ---- Chart 1: TPOT p99 vs burst factor (rps=8) ----
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
for fam in ("colocated", "disagg", "elastic"):
    pts = rowsat(fam, 8)
    ax.plot([p["burst"] for p in pts], [p["tpot99"] for p in pts],
            marker="o", color=COLORS[fam], label=LABELS[fam], linewidth=2)
ax.set_xlabel("Burst factor (6x arrival spike at t=6s, rps=8, 8 GPUs)")
ax.set_ylabel("TPOT p99 (ms)")
ax.set_title("Decode stalls under burst: colocated vs disaggregated serving\npdserve roofline simulator, 7B-class model, 512-token prompts")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "1_tpot_p99_vs_burst.png"))
plt.close(fig)

# ---- Chart 2: TTFT p99 vs burst factor (rps=8) ----
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
for fam in ("colocated", "disagg", "elastic"):
    pts = rowsat(fam, 8)
    ax.plot([p["burst"] for p in pts], [p["ttft99"] for p in pts],
            marker="o", color=COLORS[fam], label=LABELS[fam], linewidth=2)
ax.set_xlabel("Burst factor (rps=8)")
ax.set_ylabel("TTFT p99 (s)")
ax.set_title("Time-to-first-token under burst pressure\npdserve roofline simulator")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "2_ttft_p99_vs_burst.png"))
plt.close(fig)

# ---- Chart 3: Throughput vs burst (rps=8) ----
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
for fam in ("colocated", "disagg", "elastic"):
    pts = rowsat(fam, 8)
    ax.plot([p["burst"] for p in pts], [p["tps"] for p in pts],
            marker="o", color=COLORS[fam], label=LABELS[fam], linewidth=2)
ax.set_xlabel("Burst factor (rps=8)")
ax.set_ylabel("Throughput (tokens/sec, fleet-wide)")
ax.set_title("Throughput collapse under burst: colocated loses ~55%\npdserve roofline simulator")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "3_throughput_vs_burst.png"))
plt.close(fig)

# ---- Chart 4: Completed requests vs burst (rps=8) ----
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
for fam in ("colocated", "disagg", "elastic"):
    pts = rowsat(fam, 8)
    ax.plot([p["burst"] for p in pts], [p["done"] for p in pts],
            marker="o", color=COLORS[fam], label=LABELS[fam], linewidth=2)
ax.set_xlabel("Burst factor (rps=8)")
ax.set_ylabel("Requests completed in window")
ax.set_title("Completed requests: disaggregation serves 4x more under burst\npdserve roofline simulator")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "4_completed_vs_burst.png"))
plt.close(fig)

print("charts written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" -", f)
