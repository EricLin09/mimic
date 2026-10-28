"""Retrieval vectors -> models/voices/<voice>/retrieval.onnx (vectors baked in as an initializer).

Usage: export_retrieval.py <features.npy | file.index> <voice_id>

The vectors are the voice's training features: the [N, 768] matrix `mimic prepare` writes, or an
Applio / RVC FAISS index, which is read directly (no FAISS install needed).

Same search as Applio's realtime IndexWrapper: brute-force squared L2 over every stored
vector (clamped at 1e-8), top-8, weights (1/d)^2 normalised, then
out = retrieved * index_rate + feats * (1 - index_rate).
"""
import json
import sys

import numpy as np
import onnxruntime as ort
import torch

from common import MODELS, compare, read_index_vectors

SOURCE, VOICE = sys.argv[1], sys.argv[2]
OUT_DIR = MODELS / "voices" / VOICE
OUT_DIR.mkdir(parents=True, exist_ok=True)

vectors = np.load(SOURCE) if SOURCE.endswith(".npy") else read_index_vectors(SOURCE)
assert vectors.ndim == 2 and vectors.shape[1] == 768, f"expected [N, 768] vectors, got {vectors.shape}"
big = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32))
print(f"retrieval vectors: {tuple(big.shape)}")


class Retrieval(torch.nn.Module):
    def __init__(self, vectors):
        super().__init__()
        self.register_buffer("vectors", vectors)
        self.register_buffer("norms", (vectors ** 2).sum(dim=-1, keepdim=True).T.contiguous())

    def forward(self, feats, index_rate):
        q_norm = (feats ** 2).sum(dim=-1, keepdim=True)
        dist = (torch.addmm(self.norms, feats, self.vectors.T, alpha=-2.0, beta=1.0) + q_norm).clamp(min=1e-8)
        neg, ix = torch.topk(-dist, k=8, dim=-1)
        weight = (1.0 / -neg).square()
        weight = weight / weight.sum(dim=1, keepdim=True)
        retrieved = (self.vectors[ix] * weight.unsqueeze(2)).sum(dim=1)
        return retrieved * index_rate + feats * (1.0 - index_rate)


model = Retrieval(big).eval()
out = OUT_DIR / "retrieval.onnx"
with torch.inference_mode():
    torch.onnx.export(model, (torch.randn(50, 768), torch.tensor(0.75)), str(out), dynamo=False, opset_version=17,
                      input_names=["feats", "index_rate"], output_names=["blended"],
                      dynamic_axes={"feats": {0: "frames"}, "blended": {0: "frames"}})
(OUT_DIR / "retrieval.json").write_text(json.dumps({"vectors": big.shape[0], "k": 8, "source": SOURCE}, indent=2),
                                        encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")

# parity on vectors near the data (real queries look like stored ones)
rng = np.random.default_rng(0)
queries = (big[rng.integers(0, big.shape[0], 120)] + 0.05 * torch.randn(120, 768)).float()
rate = np.float32(0.75)
with torch.inference_mode():
    ref = model(queries, torch.tensor(rate))
got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
    None, {"feats": queries.numpy(), "index_rate": np.array(rate, dtype=np.float32)})[0]
compare("retrieval vs torch", ref, got)

# independent reference: exhaustive search in float64 (what FAISS does with nprobe = nlist)
db, q = big.numpy().astype(np.float64), queries.numpy().astype(np.float64)
dist = np.maximum((q * q).sum(1, keepdims=True) - 2 * q @ db.T + (db * db).sum(1), 1e-8)
ix = np.argsort(dist, axis=1)[:, :8]
w = np.square(1.0 / np.take_along_axis(dist, ix, axis=1))
w /= w.sum(axis=1, keepdims=True)
exact = (db[ix] * w[:, :, None]).sum(axis=1) * rate + q * (1 - rate)
compare("retrieval vs exhaustive float64", torch.from_numpy(exact), got)
