"""contentvec (HuBERT base, RVC v2 features) -> models/contentvec.onnx, then check against PyTorch."""
import os
import sys
from pathlib import Path

import onnxruntime as ort
import torch

# The shared nets are still exported from Applio's copies of the upstream code and weights (run with
# Applio's interpreter, -I -B; APPLIO_ROOT names the install). Only these two scripts need it; the
# voice exports do not.
APPLIO = Path(os.environ.get("APPLIO_ROOT", "C:/Applio"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I leaves the script's directory off
sys.path.insert(0, str(APPLIO))
os.chdir(APPLIO)  # Applio resolves several paths relative to the working directory

from rvc.lib.utils import HubertModelWithFinalProj  # noqa: E402
from common import MODELS, compare, parity_audio, strip_parametrizations  # noqa: E402


def load_wav_16k(path) -> torch.Tensor:
    import librosa

    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    return torch.from_numpy(audio).float()


OUT = MODELS / "contentvec.onnx"


class LastHidden(torch.nn.Module):
    def __init__(self, hubert):
        super().__init__()
        self.hubert = hubert

    def forward(self, audio):
        return self.hubert(audio)["last_hidden_state"]


hubert = HubertModelWithFinalProj.from_pretrained(
    str(APPLIO / "rvc/models/embedders/contentvec"), attn_implementation="eager"
)
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

# parity on real speech, CPU only
audio = parity_audio(6, load_wav_16k).unsqueeze(0)
with torch.inference_mode():
    ref = model(audio)
got = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"]).run(None, {"audio": audio.numpy()})[0]
print("frames", tuple(ref.shape), tuple(got.shape))
compare("contentvec", ref, got)
