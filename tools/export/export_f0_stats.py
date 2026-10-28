"""Pitch statistics of a voice -> models/voices/<voice>/f0_stats.json; the engine centres the speaker's
pitch on its median_hz.

Usage: export_f0_stats.py <prepare_dir> <voice_id>

Same statistics and fields as tools/bench/f0_stats.py (RMVPE frame f0, threshold 0.03, voiced frames
only: median, mean and standard deviation of log2 f0, 10th and 90th percentiles), computed from the
f0 that `mimic prepare` extracted for the training slices. Where two pieces of one file overlap, the
frames count once.
"""
import json
import sys
from pathlib import Path

import numpy as np

from common import MODELS

PREP, VOICE = Path(sys.argv[1]), sys.argv[2]
OUT_DIR = MODELS / "voices" / VOICE
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = json.loads((PREP / "prepare.json").read_text(encoding="utf-8"))
hop = manifest["sample_rate"] // 100  # one f0 frame per 10 ms, in samples of the slice audio
values, covered = [], {}
for item in sorted(manifest["items"], key=lambda it: (it["source"], it["start"])):
    f0 = np.load(PREP / "f0" / f"{item['name']}.npy").astype(np.float64)
    at = item["start"] + np.arange(len(f0)) * hop
    values.append(f0[at >= covered.get(item["source"], 0)])
    covered[item["source"]] = max(covered.get(item["source"], 0), item["start"] + item["samples"])
v = np.concatenate(values)
v = v[np.isfinite(v) & (v > 0)]
if not len(v):
    sys.exit("no voiced frames; refusing to write unusable pitch statistics")
p10, p50, p90 = np.percentile(v, [10, 50, 90])
stats = {"median_hz": float(p50), "log2_mean": float(np.log2(v).mean()), "log2_std": float(np.log2(v).std(ddof=0)),
         "p10_hz": float(p10), "p90_hz": float(p90)}
out = OUT_DIR / "f0_stats.json"
out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
print(f"{len(v)} voiced frames from {len(manifest['items'])} slices -> {out}: " + ", ".join(f"{k} {x:.4f}" for k, x in stats.items()))
