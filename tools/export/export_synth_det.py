"""Seam-free synthesizer export -> models/voices/<voice>/synth_det.onnx.

Usage: export_synth_det.py <checkpoint.pth> <voice_id>

Same RVC v2 network as export_synth.py, changed so consecutive realtime renders agree where
they overlap:
  * zp_noise [1,192,T] and src_noise [1,(T-skip)*hop,1] are inputs (the engine keys them to
    absolute time instead of drawing fresh noise per render);
  * phase0 (cycles) sets the NSF sine phase at the first output sample, so the excitation
    continues across renders;
  * the flow runs over the whole window and the cut to skip_head happens just before the
    decoder (Applio cuts before the flow, starving it of left context).
"""
import json
import math
import sys

import onnxruntime as ort
import torch
import torch.nn.functional as F

from common import MODELS, compare, load_synthesizer
from rvc.hifigan_nsf import SineGenerator


def _sine_forward(self, f0, upsampling_factor):
    # SineGenerator.forward with the start phase and the noise supplied from outside
    f0 = f0.unsqueeze(-1)
    batch, _, _ = f0.shape
    grid = torch.arange(1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device)
    inc = (f0 / self.sampling_rate) * grid
    rem = torch.fmod(inc[:, :-1, -1:] + 0.5, 1.0) - 0.5
    cum = rem.cumsum(dim=1).fmod(1.0).to(f0.dtype)
    inc = (inc + F.pad(cum, (0, 0, 1, 0), mode="constant")).reshape(batch, -1, 1)
    sine = torch.sin(2 * math.pi * (inc + self._phase0)) * self.sine_amplitude
    voiced = (f0 > self.voiced_threshold).float()
    voiced = F.interpolate(voiced.transpose(2, 1), scale_factor=float(upsampling_factor), mode="nearest").transpose(2, 1)
    noise = (voiced * self.noise_stddev + (1 - voiced) * (self.sine_amplitude / 3)) * self._src_noise
    return sine * voiced + noise, voiced, noise


SineGenerator.forward = _sine_forward  # in-process patch only

CKPT, VOICE = sys.argv[1], sys.argv[2]
OUT_DIR = MODELS / "voices" / VOICE
net, config, _ = load_synthesizer(CKPT)
upp = net.dec.upp
sine_gen = net.dec.m_source.l_sin_gen


class SynthGraph(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, phone, phone_lengths, pitch, pitchf, sid, skip_head, zp_noise, src_noise, phase0):
        net = self.net
        g = net.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = net.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * zp_noise * 0.66666) * x_mask
        z = net.flow(z_p, x_mask, g=g, reverse=True)
        keep = torch.arange(skip_head, z.shape[2], device=z.device)
        z, x_mask, f0 = z.index_select(2, keep), x_mask.index_select(2, keep), pitchf.index_select(1, keep)
        sine_gen._phase0, sine_gen._src_noise = phase0, src_noise
        return net.dec(z * x_mask, f0, g=g)


def example(frames, skip, seed):
    gen = torch.Generator().manual_seed(seed)
    f0 = 100 + 60 * torch.rand(1, frames, generator=gen)
    f0[:, 30:45] = 0
    mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
    coarse = torch.clip((mel - 50.0) * 254 / 1050.0 + 1, 1, 255).round().long()
    return (
        torch.randn(1, frames, 768, generator=gen),
        torch.tensor([frames], dtype=torch.int64),
        coarse,
        f0,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor(skip, dtype=torch.int64),
        torch.randn(1, 192, frames, generator=gen),
        torch.randn(1, (frames - skip) * upp, 1, generator=gen),
        torch.tensor(0.37),
    )


NAMES = ["phone", "phone_lengths", "pitch", "pitchf", "sid", "skip_head", "zp_noise", "src_noise", "phase0"]
AXES = {"phone": {1: "frames"}, "pitch": {1: "frames"}, "pitchf": {1: "frames"}, "zp_noise": {2: "frames"},
        "src_noise": {1: "samples"}, "audio": {2: "samples"}}
out = OUT_DIR / "synth_det.onnx"
graph = SynthGraph(net).eval()
with torch.inference_mode():
    torch.onnx.export(graph, example(200, 100, 0), str(out), dynamo=False, opset_version=17,
                      input_names=NAMES, output_names=["audio"], dynamic_axes=AXES)
print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")

# deterministic now, so the full graph can be compared directly, at a new length and skip
args = example(257, 131, 1)
with torch.inference_mode():
    ref = graph(*args)
got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {n: a.numpy() for n, a in zip(NAMES, args)})[0]
print("shapes", tuple(ref.shape), got.shape, "expected samples", (257 - 131) * upp)
compare("synth_det", ref, got)
(OUT_DIR / "synth_det.json").write_text(json.dumps({"sample_rate": config[-1], "hop": upp, "latent_channels": 192}, indent=2), encoding="utf-8")
