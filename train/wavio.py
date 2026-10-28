"""Minimal WAV reader/writer (mono or multichannel, PCM 16/24/32 or float32), so neither the trainer
nor the bench scripts need soundfile."""
import struct
from pathlib import Path

import numpy as np


def read_wav(path) -> tuple[np.ndarray, int]:
    """(samples as float32 in -1..1, averaged to mono, sample rate)."""
    buf = Path(path).read_bytes()
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(buf):
        cid, size = buf[pos:pos + 4], struct.unpack_from("<I", buf, pos + 4)[0]
        body = buf[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", body)
            if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: the real format is the subformat GUID's first two bytes
                tag = struct.unpack_from("<H", body, 24)[0]
            fmt = (tag, channels, rate, bits)
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, channels, rate, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(data, "<f4")
    elif tag == 1 and bits == 16:
        x = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 8388608.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(data, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported WAV format tag {tag}, {bits} bits")
    x = x[: len(x) - len(x) % channels].reshape(-1, channels)
    return (x.mean(axis=1) if channels > 1 else x[:, 0]).astype(np.float32), rate


def write_wav(path, audio: np.ndarray, rate: int) -> None:
    """Mono float32 WAV."""
    data = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32)
    Path(path).write_bytes(header + b"data" + struct.pack("<I", len(data)) + data)
