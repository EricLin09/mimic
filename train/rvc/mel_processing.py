"""Linear and mel spectrograms for training (from Applio rvc/train/mel_processing.py). The mel
filterbank is computed here in numpy, the same numbers as librosa.filters.mel (Slaney scale and
normalisation), so librosa is not needed."""
import numpy as np
import torch

_mel_basis = {}
_hann_window = {}


def _hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    mels = f / f_sp
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-300) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    freqs = f_sp * m
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """[n_mels, n_fft // 2 + 1] float32, equal to librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
    fmin=fmin, fmax=fmax) with its defaults (htk=False, norm="slaney")."""
    if fmax is None:
        fmax = float(sr) / 2
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """Magnitude STFT of [batch, samples], reflect-padded by (n_fft - hop) / 2 each side, so the frame
    count is samples // hop. sqrt(re^2 + im^2 + 1e-6), [batch, n_fft // 2 + 1, frames]."""
    key = f"{win_size}_{y.dtype}_{y.device}"
    if key not in _hann_window:
        _hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window[key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """Natural-log mel spectrogram (clamped at 1e-5) of a magnitude spectrogram."""
    key = f"{n_fft}_{num_mels}_{sample_rate}_{fmin}_{fmax}_{spec.dtype}_{spec.device}"
    if key not in _mel_basis:
        mel = mel_filterbank(sample_rate, n_fft, num_mels, fmin, fmax)
        _mel_basis[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)

    melspec = torch.matmul(_mel_basis[key], spec)
    return torch.log(melspec.clamp(min=1e-5) * 1)


def mel_spectrogram_torch(y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):
    """spectrogram_torch followed by spec_to_mel_torch."""
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)
    return spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax)
