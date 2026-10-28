"""Third-party weights the trainer and the shared-net exports start from: download and verify.

  train\\.venv\\Scripts\\python.exe train\\pretrained.py [--shared] [--dir DIR] [--from LOCAL_DIR]

Without --shared: the RVC v2 40 kHz base models fine-tuning starts from (f0G40k / f0D40k). --shared adds
the weights tools/export needs for the two shared graphs: rmvpe.pt and contentvec (config.json +
pytorch_model.bin).

Sources, each pinned to one revision: the RVC project's own Hugging Face repository
lj1995/VoiceConversionWebUI (base models and RMVPE; MIT per its model card) and lengyue233/content-vec-best
(contentvec converted to the transformers layout; MIT per its model card). Every file is checked against
its SHA-256 before it is used, and only then moved into place. --from copies from an existing local copy
instead (for example an Applio install's rvc/models), picking by name and SHA-256. The RVC README says
the base models were trained on about 50 hours of VCTK. Weights never go into the repository.
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

RVC = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/"
CONTENTVEC = "https://huggingface.co/lengyue233/content-vec-best/resolve/ab04aa7067b99ee05cc82499bc64916b980a1967/"
DEFAULT_DIR = Path(__file__).resolve().parent / "pretrained"
# path under DEFAULT_DIR -> (url, sha256, bytes), from the repositories' metadata at those revisions
FILES = {
    "f0G40k.pth": (RVC + "pretrained_v2/f0G40k.pth",
                   "3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb", 73_106_273),
    "f0D40k.pth": (RVC + "pretrained_v2/f0D40k.pth",
                   "6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09", 142_875_703),
    "rmvpe.pt": (RVC + "rmvpe.pt", "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193", 181_184_272),
    "contentvec/config.json": (CONTENTVEC + "config.json",
                               "2ddde063b795d38d9051a7215a092fecf4cfe148b54251e38de51d88d356898b", 1_388),
    "contentvec/pytorch_model.bin": (CONTENTVEC + "pytorch_model.bin",
                                     "d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e", 378_342_945),
}
BASE = ["f0G40k.pth", "f0D40k.pth"]
SHARED = ["rmvpe.pt", "contentvec/config.json", "contentvec/pytorch_model.bin"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _check(path: Path, name: str) -> None:
    _, digest, size = FILES[name]
    if path.stat().st_size != size:
        raise RuntimeError(f"{path}: {path.stat().st_size} bytes, expected {size}")
    got = sha256(path)
    if got != digest:
        raise RuntimeError(f"{path}: sha256 {got}, expected {digest}")


def _download(url: str, part: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def _local_copy(source: Path, name: str) -> Path:
    """The file under `source` with this file name, size and SHA-256 (searched recursively)."""
    _, digest, size = FILES[name]
    for p in sorted(source.rglob(Path(name).name)):
        if p.is_file() and p.stat().st_size == size and sha256(p) == digest:
            return p
    raise FileNotFoundError(f"no copy of {name} with sha256 {digest} under {source}")


def ensure(name: str, folder: Path = DEFAULT_DIR, source: Path = None) -> Path:
    """Path to a verified copy of `name` in `folder`, fetching it (or copying it from `source`) if missing."""
    if name not in FILES:
        raise KeyError(f"unknown weights {name}; known: {', '.join(FILES)}")
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _check(path, name)
        return path
    part = path.with_suffix(path.suffix + ".part")
    if source is not None:
        found = _local_copy(source, name)
        print(f"copying {name} from {found}")
        shutil.copyfile(found, part)
    else:
        print(f"downloading {name} from {FILES[name][0]}")
        _download(FILES[name][0], part)
    try:
        _check(part, name)
    except RuntimeError:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shared", action="store_true", help="also the RMVPE and contentvec weights for tools/export")
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="where the weights live (default: train/pretrained)")
    p.add_argument("--from", dest="source", type=Path, help="copy from this folder (searched by name) instead of downloading")
    args = p.parse_args()
    for name in BASE + (SHARED if args.shared else []):
        path = ensure(name, args.dir, args.source)
        print(f"{path}: sha256 ok")


if __name__ == "__main__":
    sys.exit(main())
