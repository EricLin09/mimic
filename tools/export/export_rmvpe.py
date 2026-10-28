"""RMVPE pitch net -> models/rmvpe.onnx (+ mel filterbank for the Rust front end).

The graph covers only the network: log-mel [1,128,F] -> salience [1,F,360], with F a
multiple of 32 (reflect-pad before calling). STFT/mel and cents decoding live in Rust;
models/rmvpe_mel.f32 is librosa's exact filterbank (128 x 513, little-endian f32).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

# The shared nets are still exported from Applio's copies of the upstream code and weights (run with
# Applio's interpreter, -I -B; APPLIO_ROOT names the install). Only these two scripts need it; the
# voice exports do not.
APPLIO = Path(os.environ.get("APPLIO_ROOT", "C:/Applio"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I leaves the script's directory off
sys.path.insert(0, str(APPLIO))
os.chdir(APPLIO)  # Applio resolves several paths relative to the working directory

from rvc.lib.predictors.RMVPE import E2E, MelSpectrogram, RMVPE0Predictor  # noqa: E402
from common import MODELS, compare, parity_audio  # noqa: E402


def load_wav_16k(path) -> torch.Tensor:
    import librosa

    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    return torch.from_numpy(audio).float()


CKPT = APPLIO / "rvc/models/predictors/rmvpe.pt"
OUT = MODELS / "rmvpe.onnx"

net = E2E(4, 1, (2, 2))
net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
net.eval()

with torch.inference_mode():
    torch.onnx.export(net, (torch.randn(1, 128, 96),), str(OUT), dynamo=False, opset_version=17,
                      input_names=["mel"], output_names=["salience"],
                      dynamic_axes={"mel": {2: "frames"}, "salience": {1: "frames"}})
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")

mel = MelSpectrogram(128, 16000, 1024, 160, None, 30, 8000)
basis = mel.mel_basis.numpy().astype("<f4")
(MODELS / "rmvpe_mel.f32").write_bytes(basis.tobytes())
(MODELS / "rmvpe.json").write_text(json.dumps({
    "sample_rate": 16000, "n_fft": 1024, "win_length": 1024, "hop": 160, "n_mels": 128,
    "fmin": 30, "fmax": 8000, "log_clamp": 1e-5, "mel_shape": list(basis.shape),
    "cents_base": 1997.3794084376191, "cents_step": 20, "classes": 360, "threshold": 0.03,
}, indent=2), encoding="utf-8")

# parity on real speech: network output, then the full f0 curve vs Applio's predictor
audio = parity_audio(5, load_wav_16k)
with torch.inference_mode():
    m = mel(audio.unsqueeze(0), center=True)
    frames = m.shape[-1]
    m = F.pad(m, (0, 32 * ((frames - 1) // 32 + 1) - frames), mode="reflect")
    ref = net(m)
got = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"]).run(None, {"mel": m.numpy()})[0]
compare("rmvpe salience", ref, got)

predictor = RMVPE0Predictor(str(CKPT), device="cpu")
f0_ref = predictor.infer_from_audio(audio.numpy(), thred=0.03)
f0_onnx = predictor.decode(got[0, :frames], thred=0.03)
voiced = (f0_ref > 0) | (f0_onnx > 0)
print(f"f0 frames {len(f0_ref)}, voiced {voiced.sum()}, voicing mismatches {((f0_ref > 0) != (f0_onnx > 0)).sum()}, "
      f"max |cents diff| {np.abs(1200 * np.log2((f0_onnx[voiced] + 1e-9) / (f0_ref[voiced] + 1e-9))).max():.3f}")
