"""Generate CPU-only Applio/torchaudio DSP oracles without importing app entrypoints.

Run with the reference Python: python -I -B tools/bench/gen_fixtures.py.
Only engine/tests/fixtures/*.bin are written. No personal recording is read.
The default fixture is a deterministic sweep + harmonics + seeded noise.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# Disable optional disk-backed JIT caches before importing librosa/numba.
os.environ["NUMBA_DISABLE_JIT"] = "1"
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def synthetic_audio(seed: int = 20261027) -> np.ndarray:
    """Return 0.5 seconds of generated 48 kHz audio, with no speech recording."""
    rate, samples = 48000, 24000
    time = np.arange(samples, dtype=np.float64) / rate
    sweep_phase = 2 * np.pi * (80 * time + 0.5 * (7800 - 80) / 0.5 * time ** 2)
    fundamental = 145 + 25 * np.sin(2 * np.pi * 3 * time)
    phase = 2 * np.pi * np.cumsum(fundamental) / rate
    harmonics = np.sin(phase) + 0.3 * np.sin(2 * phase + 0.2) + 0.15 * np.sin(4 * phase)
    envelope = np.sin(np.pi * (np.arange(samples) + 0.5) / samples) ** 2
    noise = np.random.default_rng(seed).standard_normal(samples)
    return (envelope * (0.055 * np.sin(sweep_phase) + 0.065 * harmonics + 0.012 * noise)).astype(np.float32)


def source_definitions(path: Path, names: set[str], namespace: dict) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError(f"Reference definitions missing in {path}: {names}")
    # Execute exact original definitions, without app import side effects/log setup.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def coarse_from_training(path: Path, f0: np.ndarray) -> np.ndarray:
    """The coarse pitch the voices were trained on: train/dataset.py's coarse_f0 and its constants,
    executed from the source (the module itself imports the training stack). Until 2026-10-28 the
    oracle was Applio's realtime pipeline, whose mel bounds are the Hz bounds."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    wanted = {"F0_BIN", "F0_MEL_MIN", "F0_MEL_MAX", "coarse_f0"}
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in wanted) or (
        isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in wanted)]
    found = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in nodes}
    if found != wanted:
        raise RuntimeError(f"Training quantizer changed; inspect {path} rather than silently changing the oracle")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["coarse_f0"](f0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--applio", type=Path, default=Path(os.environ.get("APPLIO_ROOT", r"C:\Applio")))
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = repo / "engine" / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    rmvpe_path = args.applio / "rvc/lib/predictors/RMVPE.py"
    if not rmvpe_path.exists():
        alternate = args.applio / "rvc/lib/predictors/f0.py"
        if alternate.exists():
            rmvpe_path = alternate
    core_path = args.applio / "rvc/realtime/core.py"
    dataset_path = repo / "train/dataset.py"
    model_path = repo / "models/rmvpe.onnx"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    namespace = {"torch": torch, "nn": torch.nn, "F": F, "np": np, "mel": mel, "List": list, "N_CLASS": 360, "N_MELS": 128}
    source_definitions(rmvpe_path, {"MelSpectrogram", "RMVPE0Predictor"}, namespace)
    source_definitions(core_path, {"phase_vocoder"}, namespace)
    files = {}

    def save(name: str, values) -> np.ndarray:
        array = np.ascontiguousarray(values, dtype="<f4")
        data = array.tobytes()
        (out / f"{name}.bin").write_bytes(data)
        files[name] = {"shape": list(array.shape), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return array

    audio48 = synthetic_audio(args.seed)

    def resample(audio, orig, new):
        return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(audio)), orig, new,
            lowpass_filter_width=6, rolloff=0.99, resampling_method="sinc_interp_hann").numpy()

    save("audio_48k", audio48)
    audio40 = save("audio_40k", resample(audio48, 48000, 40000))
    audio16 = save("resample_48k_16k", resample(audio48, 48000, 16000))
    save("resample_40k_48k", resample(audio40, 40000, 48000))
    stream_meta = {}
    for label, audio, orig, new in [("48k_16k", audio48, 48000, 16000), ("40k_48k", audio40, 40000, 48000)]:
        divisor = math.gcd(orig, new)
        reduced_orig, reduced_new = orig // divisor, new // divisor
        width = math.ceil(6 * reduced_orig / (min(reduced_orig, reduced_new) * 0.99))
        # Prepending width input zeros is exact fractional-delay compensation.
        # An integer output shift would be wrong: 19/3 and 42/5 are nonintegers.
        delayed = resample(np.pad(audio, (width, 0)), orig, new)
        save(f"stream_{label}", delayed)
        stream_meta[label] = {"width_input": width, "delay_output": width * reduced_new / reduced_orig, "flush_input": math.ceil(width / reduced_orig) * reduced_orig}

    extractor = namespace["MelSpectrogram"](128, 16000, 1024, 160, None, 30, 8000).cpu()
    with torch.inference_mode():
        log_mel = extractor(torch.from_numpy(audio16).unsqueeze(0), center=True)
    save("mel_basis", extractor.mel_basis.numpy())
    save("log_mel", log_mel.numpy()[0])
    frames = log_mel.shape[-1]
    padded = F.pad(log_mel, (0, math.ceil(frames / 32) * 32 - frames), mode="reflect")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    hidden = session.run(None, {session.get_inputs()[0].name: padded.numpy()})[0][0, :frames].astype(np.float32)
    # Edge/threshold/tie fixtures supplement real CPU model output.
    edges = np.zeros((8, 360), dtype=np.float32)
    edges[1, 0] = 0.8
    edges[2, 359] = 0.8
    edges[3, 140] = np.float32(0.03)
    edges[4, 140] = np.float32(0.030001)
    edges[5, 200:202] = [0.7, 0.7]
    edges[6, :5] = [0.4, 0.2, 0.1, 0.05, 0.025]
    edges[7, -5:] = [0.025, 0.05, 0.1, 0.2, 0.4]
    hidden = save("salience", np.concatenate([hidden, edges]))
    predictor = namespace["RMVPE0Predictor"].__new__(namespace["RMVPE0Predictor"])
    predictor.cents_mapping = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        f0 = predictor.decode(hidden.copy(), thred=0.03)
    save("f0", f0)
    coarse_input = save("coarse_input", np.concatenate([f0.astype(np.float32), np.array([0, 1, 49, 50, 100, 220, 440, 880, 1100, 2000], dtype=np.float32)]))
    save("coarse", coarse_from_training(dataset_path, coarse_input))

    for n in (2400, 511):
        t = torch.arange(n, dtype=torch.float32) / 48000
        a = (0.17 * torch.sin(2 * torch.pi * 183 * t) + 0.07 * torch.sin(2 * torch.pi * 549 * t + 0.2) + 0.011).float()
        b = (0.15 * torch.sin(2 * torch.pi * 187 * t + 0.47) + 0.065 * torch.sin(2 * torch.pi * 557 * t - 0.1) - 0.008).float()
        if n % 2 == 0:
            a += 0.005 * torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)
            b -= 0.004 * torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)
        fade_in = torch.sin(0.5 * torch.pi * torch.linspace(0, 1, n)) ** 2
        reference = namespace["phase_vocoder"](a, b, 1 - fade_in, fade_in)
        save(f"pv_{n}_a", a.numpy())
        save(f"pv_{n}_b", b.numpy())
        save(f"pv_{n}_out", reference.numpy())

    manifest = {
        "format": "raw little-endian f32, row-major; manifest is UTF-8 JSON",
        "source_signal": {"kind": "deterministic_synthetic_sweep_harmonics_noise", "seed": args.seed,
                          "frames": 24000, "sample_rate": 48000, "contains_personal_recording": False,
                          "generator_sha256": sha256(Path(__file__))},
        "source_hashes": {"applio/rmvpe_reference.py": sha256(rmvpe_path),
                          "applio/realtime/core.py": sha256(core_path),
                          "train/dataset.py": sha256(dataset_path),
                          "models/rmvpe.onnx": sha256(model_path)},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "torchaudio": torchaudio.__version__, "onnxruntime": ort.__version__},
        "providers": session.get_providers(), "mel_frames": frames,
        "stream_alignment": stream_meta, "fixtures": files,
    }
    (out / "manifest.bin").write_bytes((json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    total = sum(path.stat().st_size for path in out.glob("*.bin"))
    if total >= 2_000_000:
        raise RuntimeError(f"Fixture size budget exceeded: {total}")
    print(f"CPU fixtures written: {len(files)} arrays + manifest; {total} bytes; frames={frames}; providers={session.get_providers()}")
    for label, metadata in stream_meta.items():
        print(f"stream {label}: width={metadata['width_input']} input samples; delay={metadata['delay_output']:.9f} output samples")


if __name__ == "__main__":
    main()
