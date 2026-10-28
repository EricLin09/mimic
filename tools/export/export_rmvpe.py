"""RMVPE pitch net -> models/rmvpe.onnx (+ mel filterbank for the Rust front end).

Usage: export_rmvpe.py [--weights rmvpe.pt]

The graph covers only the network: log-mel [1,128,F] -> salience [1,F,360], with F a
multiple of 32 (reflect-pad before calling). STFT/mel and cents decoding live in Rust;
models/rmvpe_mel.f32 is RMVPE's HTK-scale filterbank (128 x 513, little-endian f32, the same
numbers as librosa). Weights default to train/pretrained/rmvpe.pt (train/pretrained.py --shared).
The self-check runs on the recording MIMIC_PARITY_WAV names, else on synthetic speech.
"""
import argparse
import json

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from common import MODELS, compare, load_wav_16k, parity_audio
from rmvpe_net import E2E, N_CLASS

import pretrained  # noqa: E402  (train/, put on sys.path by common)
from rvc.mel_processing import mel_filterbank  # noqa: E402

CENTS_BASE, THRESHOLD = 1997.3794084376191, 0.03


def log_mel(audio: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """RMVPE's front end: centred STFT (periodic Hann 1024, hop 160), magnitude, mel, ln(max(x, 1e-5))."""
    spec = torch.stft(audio, n_fft=1024, hop_length=160, win_length=1024, window=torch.hann_window(1024),
                      center=True, return_complex=True)
    return torch.log(torch.clamp(basis @ torch.sqrt(spec.real.pow(2) + spec.imag.pow(2)), min=1e-5))


def decode(salience: np.ndarray) -> np.ndarray:
    """Weighted cents average over +-4 bins around the peak; unvoiced (0) where the peak <= THRESHOLD."""
    centers = salience.argmax(axis=1)
    padded = np.pad(salience, ((0, 0), (4, 4)))
    idx = centers[:, None] + np.arange(9)[None, :]
    weights = padded[np.arange(len(salience))[:, None], idx]
    cents = np.pad(20 * np.arange(N_CLASS) + CENTS_BASE, (4, 4))[idx]
    f0 = 10 * 2 ** ((weights * cents).sum(1) / weights.sum(1) / 1200)
    f0[(salience.max(axis=1) <= THRESHOLD) | (f0 == 10)] = 0
    return f0


p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--weights")
args = p.parse_args()
CKPT = args.weights or pretrained.ensure("rmvpe.pt")
OUT = MODELS / "rmvpe.onnx"
MODELS.mkdir(parents=True, exist_ok=True)

net = E2E(4, 1, (2, 2))
net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
net.eval()

with torch.inference_mode():
    torch.onnx.export(net, (torch.randn(1, 128, 96),), str(OUT), dynamo=False, opset_version=17,
                      input_names=["mel"], output_names=["salience"],
                      dynamic_axes={"mel": {2: "frames"}, "salience": {1: "frames"}})
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")

basis = mel_filterbank(16000, 1024, 128, 30, 8000, htk=True).astype("<f4")
(MODELS / "rmvpe_mel.f32").write_bytes(basis.tobytes())
(MODELS / "rmvpe.json").write_text(json.dumps({
    "sample_rate": 16000, "n_fft": 1024, "win_length": 1024, "hop": 160, "n_mels": 128,
    "fmin": 30, "fmax": 8000, "log_clamp": 1e-5, "mel_shape": list(basis.shape),
    "cents_base": CENTS_BASE, "cents_step": 20, "classes": N_CLASS, "threshold": THRESHOLD,
}, indent=2), encoding="utf-8")

audio = parity_audio(5, load_wav_16k)
with torch.inference_mode():
    m = log_mel(audio.unsqueeze(0), torch.from_numpy(basis))
    frames = m.shape[-1]
    m = F.pad(m, (0, 32 * ((frames - 1) // 32 + 1) - frames), mode="reflect")
    ref = net(m)
got = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"]).run(None, {"mel": m.numpy()})[0]
compare("rmvpe salience", ref, got)
f0_ref, f0_onnx = decode(ref[0, :frames].numpy()), decode(got[0, :frames])
voiced = (f0_ref > 0) & (f0_onnx > 0)
print(f"f0 frames {frames}, voiced {voiced.sum()}, voicing mismatches {((f0_ref > 0) != (f0_onnx > 0)).sum()}, "
      f"max |cents diff| {np.abs(1200 * np.log2(f0_onnx[voiced] / f0_ref[voiced])).max() if voiced.any() else 0:.3f}")
