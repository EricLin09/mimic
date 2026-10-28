"""contentvec (HuBERT base, RVC v2 features) -> models/contentvec.onnx, then check against PyTorch.

Usage: export_contentvec.py [--weights DIR]

DIR holds contentvec in the transformers layout (config.json + pytorch_model.bin):
train/pretrained/contentvec by default (train/pretrained.py --shared). Needs transformers
(pip install transformers==5.13.1, the version the current graph was exported with), which nothing
else in the repository uses. The self-check runs on the recording MIMIC_PARITY_WAV names, else on
synthetic speech.
"""
import argparse
from pathlib import Path

import onnxruntime as ort
import torch
from transformers import HubertModel

from common import MODELS, compare, load_wav_16k, parity_audio, strip_parametrizations

import pretrained  # noqa: E402  (train/, put on sys.path by common)

OUT = MODELS / "contentvec.onnx"


class HubertModelWithFinalProj(HubertModel):
    """contentvec's checkpoint layout (as in RVC): HuBERT plus a final_proj head. The head is not part of
    the graph, but with it the checkpoint loads without leftovers and the export matches the original
    file byte for byte (the tracer numbers values after the head's parameters)."""

    def __init__(self, config):
        super().__init__(config)
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)


class LastHidden(torch.nn.Module):
    def __init__(self, hubert):
        super().__init__()
        self.hubert = hubert

    def forward(self, audio):
        return self.hubert(audio)["last_hidden_state"]


p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--weights", type=Path)
args = p.parse_args()
if args.weights is None:
    pretrained.ensure("contentvec/config.json")
    weights = pretrained.ensure("contentvec/pytorch_model.bin").parent
else:
    weights = args.weights
MODELS.mkdir(parents=True, exist_ok=True)

hubert = HubertModelWithFinalProj.from_pretrained(str(weights), attn_implementation="eager")
model = LastHidden(strip_parametrizations(hubert)).eval()

with torch.inference_mode():
    torch.onnx.export(
        model,
        (torch.randn(1, 32000) * 0.1,),
        str(OUT),
        dynamo=False,
        opset_version=17,
        input_names=["audio"],
        output_names=["feats"],
        dynamic_axes={"audio": {1: "samples"}, "feats": {1: "frames"}},
    )
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")

audio = parity_audio(6, load_wav_16k).unsqueeze(0)
with torch.inference_mode():
    ref = model(audio)
got = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"]).run(None, {"audio": audio.numpy()})[0]
print("frames", tuple(ref.shape), tuple(got.shape))
compare("contentvec", ref, got)
