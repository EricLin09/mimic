"""The RVC v2 40 kHz base models fine-tuning starts from (f0G40k / f0D40k): download and verify.

  train\\.venv\\Scripts\\python.exe train\\pretrained.py [--dir DIR] [--from LOCAL_DIR]

Source: the RVC project's own Hugging Face repository, lj1995/VoiceConversionWebUI (MIT per its model
card), pinned to one revision; every file is checked against its SHA-256 before it is used, and only
then moved into place. --from copies from an existing local copy instead (for example an Applio
install's rvc/models/pretraineds/hifi-gan), with the same check. The RVC README says the base models
were trained on about 50 hours of VCTK. Weights never go into the repository (.gitignore: *.pth).
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REVISION = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
BASE_URL = f"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/{REVISION}/pretrained_v2/"
DEFAULT_DIR = Path(__file__).resolve().parent / "pretrained"
# name -> (sha256, bytes), from the repository's LFS metadata at REVISION
FILES = {
    "f0G40k.pth": ("3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb", 73_106_273),
    "f0D40k.pth": ("6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09", 142_875_703),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _check(path: Path, name: str) -> None:
    digest, size = FILES[name]
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


def ensure(name: str, folder: Path = DEFAULT_DIR, source: Path = None) -> Path:
    """Path to a verified copy of `name` in `folder`, fetching it (or copying it from `source`) if missing."""
    if name not in FILES:
        raise KeyError(f"unknown base model {name}; known: {', '.join(FILES)}")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if path.exists():
        _check(path, name)
        return path
    part = path.with_suffix(path.suffix + ".part")
    if source is not None:
        print(f"copying {name} from {source}")
        shutil.copyfile(source / name, part)
    else:
        print(f"downloading {name} from {BASE_URL}")
        _download(BASE_URL + name, part)
    try:
        _check(part, name)
    except RuntimeError:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="where the weights live (default: train/pretrained)")
    p.add_argument("--from", dest="source", type=Path, help="copy from this folder instead of downloading")
    args = p.parse_args()
    for name in FILES:
        path = ensure(name, args.dir, args.source)
        print(f"{path}: sha256 ok")


if __name__ == "__main__":
    sys.exit(main())
