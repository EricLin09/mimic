"""Shared helpers for exporting RVC v2 voices to ONNX.

The networks come from the repository's own copy in train/rvc, so neither Applio nor FAISS is needed.
Run the scripts with the training environment (docs/training.md):
    train\\.venv\\Scripts\\python.exe tools\\export\\<script>.py ...
Output goes under models/ (set MIMIC_MODELS to write somewhere else).
"""
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.parametrize

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("MIMIC_MODELS", REPO / "models"))

sys.path.insert(0, str(REPO / "train"))


def resample(x: torch.Tensor, orig: int, new: int, width: int = 6) -> torch.Tensor:
    """1-D torchaudio.functional.resample (sinc_interp_hann, rolloff 0.99), without torchaudio."""
    g = math.gcd(orig, new)
    orig, new = orig // g, new // g
    if orig == new:
        return x
    base = min(orig, new) * 0.99
    w = math.ceil(width * orig / base)
    idx = torch.arange(-w, w + orig, dtype=torch.float64)[None, None] / orig
    t = ((torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + idx) * base).clamp(-width, width)
    window = torch.cos(t * math.pi / width / 2) ** 2
    t = t * math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), t.sin() / t) * window * (base / orig)
    y = torch.nn.functional.conv1d(torch.nn.functional.pad(x.float()[None, None], (w, w + orig)), kernel.float(), stride=orig)
    return y.transpose(1, 2).reshape(-1)[: math.ceil(new * len(x) / orig)]


def load_wav_16k(path) -> torch.Tensor:
    """A WAV as mono 16 kHz float32 (the realtime path's resampler)."""
    from wavio import read_wav

    audio, rate = read_wav(path)
    return resample(torch.from_numpy(audio), rate, 16000)


def strip_parametrizations(module: torch.nn.Module) -> torch.nn.Module:
    """Bake weight-norm style parametrizations into plain weights (same as Applio's realtime path)."""
    for _, sub in module.named_modules():
        if hasattr(sub, "weight_g") and hasattr(sub, "weight_v"):
            torch.nn.utils.remove_weight_norm(sub)
        if hasattr(sub, "parametrizations"):
            for name in list(sub.parametrizations.keys()):
                torch.nn.utils.parametrize.remove_parametrizations(sub, name, leave_parametrized=True)
    return module


def load_synthesizer(ckpt_path):
    """RVC v2 f0 synthesizer from an RVC / Applio inference checkpoint, weight norm baked in, in eval mode.

    Returns (net, config, checkpoint)."""
    from rvc.synthesizers import Synthesizer

    cpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    assert cpt.get("version") == "v2" and cpt.get("f0", 1) == 1, "only f0 RVC v2 models are supported"
    vocoder = cpt.get("vocoder", "HiFi-GAN")
    assert vocoder == "HiFi-GAN", f"only the NSF HiFi-GAN vocoder is supported, not {vocoder}"
    net = Synthesizer(*config)
    # inference checkpoints leave out the posterior encoder, which only training uses
    missing, unexpected = net.load_state_dict(cpt["weight"], strict=False)
    assert not unexpected and all(k.startswith("enc_q.") for k in missing), (unexpected, missing)
    return strip_parametrizations(net).eval(), config, cpt


def read_index_vectors(path) -> np.ndarray:
    """Every vector of an Applio / RVC FAISS index (IVF-Flat or Flat, L2), straight from the file.

    Same array as faiss' index.reconstruct_n(0, ntotal): row i is the i-th vector added. Follows
    faiss/impl/index_write.cpp for the two index types RVC writes; anything else is refused."""
    buf = Path(path).read_bytes()
    pos = 0

    def take(fmt):
        nonlocal pos
        vals = struct.unpack_from(fmt, buf, pos)
        pos += struct.calcsize(fmt)
        return vals

    def header():  # write_index_header: d, ntotal, two unused fields, is_trained, metric
        d, ntotal, _, _, _, metric = take("<iqqq?i")
        assert metric == 1, f"only L2 indexes are supported (metric {metric})"
        return d, ntotal

    def vector(itemsize):  # WRITEVECTOR: size_t count, then the items
        nonlocal pos
        (n,) = take("<Q")
        start, pos = pos, pos + n * itemsize
        return buf[start:pos]

    def flat():  # IndexFlatL2 body: the vectors in id order
        d, ntotal = header()
        return np.frombuffer(vector(4), "<f4").reshape(ntotal, d)

    kind = take("4s")[0]
    if kind == b"IxF2":
        return flat().astype(np.float32)
    assert kind == b"IwFl", f"unsupported FAISS index type {kind!r}"
    d, ntotal = header()
    nlist, _nprobe = take("<QQ")
    assert take("4s")[0] == b"IxF2", "IVF quantizer is not a flat L2 index"
    flat()  # coarse centroids, not needed
    (map_type,) = take("<b")
    vector(8)  # direct map array
    if map_type == 2:
        vector(16)  # hashtable entries
    assert take("4s")[0] == b"ilar", "inverted lists are not stored inline"
    n_lists, code_size = take("<QQ")
    assert n_lists == nlist and code_size == 4 * d, (n_lists, code_size)
    list_type = take("4s")[0]
    sizes = np.frombuffer(vector(8), "<u8")
    if list_type == b"full":
        lists = enumerate(sizes)
    else:
        assert list_type == b"sprs", f"unknown inverted list layout {list_type!r}"
        lists = zip(sizes[0::2], sizes[1::2])
    out = np.empty((ntotal, d), np.float32)
    seen = 0
    for _, n in lists:
        n = int(n)
        if n == 0:
            continue
        codes = np.frombuffer(buf, "<f4", n * d, pos).reshape(n, d)
        pos += n * code_size
        ids = np.frombuffer(buf, "<i8", n, pos)
        pos += 8 * n
        out[ids] = codes
        seen += n
    assert seen == ntotal and pos == len(buf), (seen, ntotal, pos, len(buf))
    return out


def synthetic_speech(seconds: float, rate: int = 16000) -> torch.Tensor:
    """Deterministic voiced test signal for parity checks, so no recording is needed: a gliding
    harmonic source with a formant-like bump, syllable-rate amplitude and a little noise."""
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 + 60 * np.sin(2 * np.pi * 0.7 * t) + 20 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    x = sum(np.sin(k * phase) / k * (1.0 + 0.8 * np.exp(-(((k * f0) - 700) / 300) ** 2)) for k in range(1, 40))
    envelope = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None)
    noise = np.random.default_rng(20261028).standard_normal(len(t))
    return torch.from_numpy((0.1 * x * envelope + 0.003 * noise).astype(np.float32))


def parity_audio(seconds: float, loader) -> torch.Tensor:
    """The recording MIMIC_PARITY_WAV names (read with `loader` at 16 kHz), else synthetic_speech."""
    path = os.environ.get("MIMIC_PARITY_WAV")
    audio = loader(path) if path else synthetic_speech(seconds)
    return audio[: int(16000 * seconds)]


def compare(name, reference: torch.Tensor, candidate) -> None:
    ref = reference.detach().cpu().numpy().astype(np.float64).ravel()
    got = np.asarray(candidate, dtype=np.float64).ravel()
    cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    print(f"{name}: max|diff| {np.abs(ref - got).max():.2e}, mean|diff| {np.abs(ref - got).mean():.2e}, cosine {cos:.6f}")
