"""Correlation-adaptive crossfades; run with --test or --benchmark.

Only NumPy and the Python standard library are required. STFT code is an
offline reference: a live implementation must keep the stated overlap state.
"""

from __future__ import annotations

import argparse
import time
import numpy as np


def correlation(x: np.ndarray, y: np.ndarray, eps: float = 1e-20) -> float:
    """Return zero-mean cosine correlation, with silence mapped to zero."""
    xc = np.asarray(x, dtype=np.float64) - np.mean(x)
    yc = np.asarray(y, dtype=np.float64) - np.mean(y)
    power = float(np.dot(xc, xc) * np.dot(yc, yc))
    if power <= eps:
        return 0.0
    return float(np.clip(np.dot(xc, yc) / np.sqrt(power), -1.0, 1.0))


def adaptive_gains(
    size: int, rho: float, *, gain_limit: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize complementary sine-squared amplitude gains.

    The ellipse identity is exact when the safety gain limit is inactive.
    Negative correlation can cause gain overshoot and nonmonotonic b.
    """
    if size < 2 or not -1.0 <= rho <= 1.0 or gain_limit < 1.0:
        raise ValueError("Require size >= 2, rho in [-1, 1], gain_limit >= 1")
    theta = np.linspace(0.0, np.pi / 2.0, size)
    v = np.sin(theta) ** 2
    u = 1.0 - v
    denominator = u * u + v * v + 2.0 * rho * u * v
    normalizer = np.minimum(
        1.0 / np.sqrt(np.maximum(denominator, 1e-20)), gain_limit
    )
    a, b = u * normalizer, v * normalizer
    a[0], b[0], a[-1], b[-1] = 1.0, 0.0, 0.0, 1.0
    return a, b


def adaptive_crossfade(
    x: np.ndarray, y: np.ndarray, rho: float | None = None
) -> np.ndarray:
    """Use equal-power gains; equal input powers are a precondition."""
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Require two mono arrays with the same shape")
    a, b = adaptive_gains(len(x), correlation(x, y) if rho is None else rho)
    return a * x + b * y


def covariance_crossfade(
    x: np.ndarray, y: np.ndarray, *, gain_limit: float = 2.0
) -> np.ndarray:
    """Allow unequal endpoint powers with a smooth interpolated power target.

    Moments include DC because actual output mean-square includes DC. Remove
    DC upstream if a Pearson-style correlation is desired.
    """
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("Require equal mono arrays of length >= 2")
    v = np.sin(np.linspace(0.0, np.pi / 2.0, len(x))) ** 2
    u = 1.0 - v
    px, py, cross = np.mean(x * x), np.mean(y * y), np.mean(x * y)
    expected = u * u * px + v * v * py + 2.0 * u * v * cross
    target = u * px + v * py
    gain = np.minimum(
        np.sqrt(np.maximum(target, 0.0) / np.maximum(expected, 1e-20)),
        gain_limit,
    )
    gain[0] = gain[-1] = 1.0
    return gain * (u * x + v * y)


def stft_band_crossfade(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sample_rate: int = 48000,
    fft_size: int = 512,
    hop: int = 128,
    gain_limit: float = 2.0,
) -> np.ndarray:
    """Reference six-band STFT mixer with real gains and normalized OLA.

    Statistics are exponentially smoothed in time and gains are interpolated
    in frequency. Input phases are retained; no independent phase reset occurs.
    Centered windows consume fft_size/2 samples of future context. Live use
    must analyze the surrounding stream, including frames outside the fade.
    """
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("Require equal mono arrays of length >= 2")
    if fft_size % 2 or hop <= 0 or hop > fft_size // 2:
        raise ValueError("Require an even FFT and 0 < hop <= fft_size/2")
    pad = fft_size // 2
    extra = (-len(x)) % hop
    xp = np.pad(x, (pad, pad + extra), mode="reflect")
    yp = np.pad(y, (pad, pad + extra), mode="reflect")
    window = np.sqrt(np.hanning(fft_size + 1)[:-1])
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    edges = np.array([0, 250, 700, 1600, 3500, 7500, sample_rate / 2 + 1])
    if np.any(np.diff(edges) <= 0):
        raise ValueError("The six-band layout requires sample_rate >= 16000")
    band_ids = np.clip(np.searchsorted(edges, frequencies, side="right") - 1, 0, 5)
    if any(not np.any(band_ids == j) for j in range(6)):
        raise ValueError("FFT resolution is too low for the six-band layout")
    centers = np.array([np.mean(frequencies[band_ids == j]) for j in range(6)])
    result, weight = np.zeros_like(xp), np.zeros_like(xp)
    frames_x = np.lib.stride_tricks.sliding_window_view(xp, fft_size)[::hop]
    frames_y = np.lib.stride_tricks.sliding_window_view(yp, fft_size)[::hop]
    sx = np.fft.rfft(frames_x * window, axis=1)
    sy = np.fft.rfft(frames_y * window, axis=1)
    selectors = np.array([band_ids == j for j in range(6)], dtype=np.float64).T
    selectors /= np.sum(selectors, axis=0)
    current = np.stack((
        (np.abs(sx) ** 2) @ selectors,
        (np.abs(sy) ** 2) @ selectors,
        np.real(sx * np.conj(sy)) @ selectors,
    ), axis=2)
    moments = current.copy()
    for frame in range(1, len(moments)):
        moments[frame] = 0.65 * moments[frame - 1] + 0.35 * current[frame]
    progress = np.clip(np.arange(len(moments)) * hop / (len(x) - 1), 0.0, 1.0)
    v = (np.sin(progress * np.pi / 2.0) ** 2)[:, None]
    u = 1.0 - v
    px, py, cross = moments[:, :, 0], moments[:, :, 1], moments[:, :, 2]
    expected = u * u * px + v * v * py + 2.0 * u * v * cross
    target = u * px + v * py
    band_gain = np.minimum(
        np.sqrt(np.maximum(target, 0.0) / np.maximum(expected, 1e-20)), gain_limit
    )
    gains = np.array([np.interp(frequencies, centers, row) for row in band_gain])
    mixed_frames = np.fft.irfft(gains * (u * sx + v * sy), fft_size, axis=1) * window
    for frame, start in enumerate(range(0, len(xp) - fft_size + 1, hop)):
        result[start : start + fft_size] += mixed_frames[frame]
        weight[start : start + fft_size] += window * window
    return (result / np.maximum(weight, 1e-20))[pad : pad + len(x)]


def rms_pp_db(x: np.ndarray, window: int = 480, hop: int = 48) -> float:
    """Measure unweighted rectangular-window RMS over complete windows."""
    power = np.convolve(x * x, np.ones(window) / window, mode="valid")[::hop]
    levels = 10.0 * np.log10(np.maximum(power, 1e-20))
    return float(np.ptp(levels))


def synthetic_pair(rho: float, *, periodic_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Create a common harmonic source plus independent noise components.

    Periodic noise repeats a seeded 5 ms band-limited noise cycle. This gives
    a stationary, exactly measurable fixture, not a guarantee for real speech.
    """
    rng = np.random.default_rng(20261027)
    size = 240 if periodic_noise else 2400
    phase = 2 * np.pi * np.arange(size) / 240
    source = np.sin(phase) + 0.4 * np.sin(3 * phase) + 0.2 * np.sin(5 * phase)
    source -= np.mean(source)
    source /= np.sqrt(np.mean(source * source))
    noises = []
    for _ in range(2):
        noise = rng.normal(size=size)
        if periodic_noise:
            spectrum = np.fft.rfft(noise)
            spectrum[:30] = 0.0
            noise = np.fft.irfft(spectrum, size)
        noise -= np.mean(noise)
        noise -= np.mean(noise * source) * source
        for old in noises:
            noise -= np.mean(noise * old) * old
        noise /= np.sqrt(np.mean(noise * noise))
        noises.append(noise)
    x = np.sqrt(rho) * source + np.sqrt(1.0 - rho) * noises[0]
    y = np.sqrt(rho) * source + np.sqrt(1.0 - rho) * noises[1]
    if periodic_noise:
        x, y = np.tile(x, 10), np.tile(y, 10)
    return x * 0.15, y * 0.15


def run_tests() -> None:
    for rho in (0.0, 0.3, 0.5, 0.8, 0.95, 1.0):
        a, b = adaptive_gains(2400, rho)
        assert np.max(np.abs(a * a + b * b + 2 * rho * a * b - 1)) < 1e-12
        assert np.min(np.diff(b)) >= -1e-14
        assert (a[0], b[0], a[-1], b[-1]) == (1, 0, 0, 1)
    a, b = adaptive_gains(2400, -0.5)
    assert np.max(b) > 1.0
    assert np.max(np.abs(a * a + b * b - a * b - 1)) < 1e-12
    a, b = adaptive_gains(2400, -1.0)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    print("periodic band-limited noise; 10 ms RMS windows / 1 ms hop")
    print("rho   measured_rho  old_pp_db  adaptive_pp_db")
    for rho in (0.3, 0.5, 0.8, 0.95):
        x, y = synthetic_pair(rho)
        u = np.cos(np.linspace(0.0, np.pi / 2.0, len(x))) ** 2
        old = u * x + (1.0 - u) * y
        mixed = adaptive_crossfade(x, y)
        ripple = rms_pp_db(mixed)
        print(f"{rho:.2f}  {correlation(x, y):.6f}      {rms_pp_db(old):.4f}     {ripple:.4f}")
        assert ripple < 0.3
    x, y = synthetic_pair(0.5)
    reconstructed = stft_band_crossfade(x, x)
    assert np.max(np.abs(reconstructed - x)) < 1e-12
    assert np.all(np.isfinite(stft_band_crossfade(x, y)))
    assert np.max(np.abs(covariance_crossfade(x, x) - x)) < 1e-12
    print("nonperiodic Gaussian-noise stress test (diagnostic, no universal bound)")
    for rho in (0.3, 0.5, 0.8, 0.95):
        x, y = synthetic_pair(rho, periodic_noise=False)
        print(f"rho={rho:.2f}, adaptive_pp_db={rms_pp_db(adaptive_crossfade(x, y)):.4f}")
    print("PASS: gain identity, endpoints, monotonicity, negative-correlation safety, OLA identity")


def run_benchmarks() -> None:
    x, y = synthetic_pair(0.5)
    for name, operation, repeats in (
        ("global + correlation", lambda: adaptive_crossfade(x, y), 2000),
        ("unequal-power moments", lambda: covariance_crossfade(x, y), 2000),
        ("six-band STFT", lambda: stft_band_crossfade(x, y), 200),
    ):
        for _ in range(10):
            operation()
        samples = []
        for _ in range(repeats):
            before = time.perf_counter_ns()
            operation()
            samples.append((time.perf_counter_ns() - before) / 1e6)
        print(f"{name}: median={np.median(samples):.4f} ms, p95={np.percentile(samples, 95):.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.test or not args.benchmark:
        run_tests()
    if args.benchmark:
        run_benchmarks()
