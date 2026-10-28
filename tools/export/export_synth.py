"""RVC v2 synthesizer (enc_p + flow + NSF HiFi-GAN) -> models/voices/<voice>/synth.onnx.

Usage: export_synth.py <checkpoint.pth> <voice_id>

The graph takes the whole conversion window and drops the first `skip_head` frames
before the flow, exactly like Applio's Synthesizer.infer_latent(skip_head=...).
Inputs need at least 12 frames (relative-attention padding is traced for long inputs).
"""
import json
import sys

import numpy as np
import onnxruntime as ort
import torch

from common import MODELS, compare, load_synthesizer

CKPT, VOICE = sys.argv[1], sys.argv[2]
OUT_DIR = MODELS / "voices" / VOICE
OUT_DIR.mkdir(parents=True, exist_ok=True)

net, config, _ = load_synthesizer(CKPT)
sample_rate, upp = config[-1], net.dec.upp


class SynthGraph(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, phone, phone_lengths, pitch, pitchf, sid, skip_head):
        net = self.net
        g = net.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = net.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        # tensor slicing would be frozen by the tracer; arange + index_select stays dynamic
        keep = torch.arange(skip_head, z_p.shape[2], device=z_p.device)
        z_p, x_mask, f0 = z_p.index_select(2, keep), x_mask.index_select(2, keep), pitchf.index_select(1, keep)
        z = net.flow(z_p, x_mask, g=g, reverse=True)
        return net.dec(z * x_mask, f0, g=g)


def example(frames=200, skip=100, seed=0):
    gen = torch.Generator().manual_seed(seed)
    f0 = 100 + 60 * torch.rand(1, frames, generator=gen)
    f0[:, 30:45] = 0  # an unvoiced stretch
    f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
    coarse = torch.clip((f0_mel - 50.0) * 254 / (1100.0 - 50.0) + 1, 1, 255).round().long()
    return (
        torch.randn(1, frames, 768, generator=gen),
        torch.tensor([frames], dtype=torch.int64),
        coarse,
        f0,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor(skip, dtype=torch.int64),
    )


NAMES = ["phone", "phone_lengths", "pitch", "pitchf", "sid", "skip_head"]
AXES = {"phone": {1: "frames"}, "pitch": {1: "frames"}, "pitchf": {1: "frames"}, "audio": {2: "samples"}}


def export(path, graph):
    with torch.inference_mode():
        torch.onnx.export(graph, example(), str(path), dynamo=False, opset_version=17,
                          input_names=NAMES, output_names=["audio"], dynamic_axes=AXES)


# 1) parity: same graph with both noise sources zeroed, against PyTorch, at a different length
real_randn_like = torch.randn_like
torch.randn_like = lambda t, *a, **k: torch.zeros_like(t)
try:
    parity_path = OUT_DIR / "_parity_synth.onnx"
    export(parity_path, SynthGraph(net))
    args = example(frames=257, skip=131, seed=1)
    with torch.inference_mode():
        ref = SynthGraph(net)(*args)
    sess = ort.InferenceSession(str(parity_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: a.numpy() for n, a in zip(NAMES, args)})[0]
    print("parity shapes", tuple(ref.shape), got.shape, "expected samples", (257 - 131) * upp)
    compare("synth (noise off)", ref, got)
finally:
    torch.randn_like = real_randn_like
    parity_path.unlink(missing_ok=True)

# 2) production graph with the model's own noise
out = OUT_DIR / "synth.onnx"
export(out, SynthGraph(net))
meta = {"sample_rate": sample_rate, "hop": upp, "speakers": config[-3], "checkpoint": str(CKPT)}
(OUT_DIR / "synth.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)", meta)
