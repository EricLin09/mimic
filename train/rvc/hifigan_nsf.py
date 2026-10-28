"""NSF HiFi-GAN vocoder (from Applio rvc/lib/algorithm/generators/hifigan_nsf.py; SineGenerator moved
in from generators/hifigan.py, the plain HiFi-GAN generator it shared that file with is left out)."""
import math
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.parametrizations import weight_norm

from rvc.commons import init_weights
from rvc.residuals import LRELU_SLOPE, ResBlock


class SineGenerator(torch.nn.Module):
    """Sine excitation at f0 (plus harmonics), Gaussian noise added; unvoiced frames get noise only.

    Args:
        sampling_rate: output sample rate in Hz.
        num_harmonics: overtones above f0.
        sine_amplitude: sine amplitude.
        noise_stddev: noise level on voiced samples (unvoiced: sine_amplitude / 3).
        voiced_threshold: f0 above this is voiced.
    """

    def __init__(self, sampling_rate: int, num_harmonics: int = 0, sine_amplitude: float = 0.1,
                 noise_stddev: float = 0.003, voiced_threshold: float = 0.0):
        super(SineGenerator, self).__init__()
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.sine_amplitude = sine_amplitude
        self.noise_stddev = noise_stddev
        self.voiced_threshold = voiced_threshold
        self.waveform_dim = self.num_harmonics + 1  # fundamental + harmonics

    def _compute_voiced_unvoiced(self, f0: torch.Tensor):
        uv_mask = (f0 > self.voiced_threshold).float()
        return uv_mask

    def _generate_sine_wave(self, f0: torch.Tensor, upsampling_factor: int):
        batch_size, length, _ = f0.shape

        upsampling_grid = torch.arange(1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device)

        # per-sample phase increments, with the phase carried across frames
        phase_increments = (f0 / self.sampling_rate) * upsampling_grid
        phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        phase_increments += torch.nn.functional.pad(cumulative_phase, (0, 0, 1, 0), mode="constant")

        phase_increments = phase_increments.reshape(batch_size, -1, 1)

        harmonic_scale = torch.arange(1, self.waveform_dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        phase_increments *= harmonic_scale

        # random start phase for the harmonics; the fundamental starts at 0
        random_phase = torch.rand(1, 1, self.waveform_dim, device=f0.device)
        random_phase[..., 0] = 0
        phase_increments += random_phase

        sine_waves = torch.sin(2 * np.pi * phase_increments)
        return sine_waves

    def forward(self, f0: torch.Tensor, upsampling_factor: int):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)

            sine_waves = self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude

            voiced_mask = self._compute_voiced_unvoiced(f0)

            voiced_mask = torch.nn.functional.interpolate(
                voiced_mask.transpose(2, 1),
                scale_factor=float(upsampling_factor),
                mode="nearest",
            ).transpose(2, 1)

            noise_amplitude = voiced_mask * self.noise_stddev + (1 - voiced_mask) * (self.sine_amplitude / 3)

            noise = noise_amplitude * torch.randn_like(sine_waves)

            sine_waveforms = sine_waves * voiced_mask + noise

        return sine_waveforms, voiced_mask, noise


class SourceModuleHnNSF(torch.nn.Module):
    """Harmonic-plus-noise source: SineGenerator merged to one channel by a linear layer and tanh.

    Args:
        sample_rate: output sample rate in Hz.
        harmonic_num: overtones above f0.
        sine_amp: sine amplitude.
        add_noise_std: voiced noise level.
        voiced_threshod: f0 above this is voiced.
    """

    def __init__(self, sample_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, add_noise_std: float = 0.003,
                 voiced_threshod: float = 0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGenerator(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upsample_factor: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class HiFiGANNSFGenerator(torch.nn.Module):
    """HiFi-GAN generator driven by an NSF excitation: the source is injected, downsampled, after each
    upsampling stage.

    Args:
        initial_channel: latent channels in.
        resblock_kernel_sizes: kernel size of each ResBlock per stage.
        resblock_dilation_sizes: dilations of each ResBlock per stage.
        upsample_rates: upsampling factor per stage (their product is the hop).
        upsample_initial_channel: channels after the input convolution.
        upsample_kernel_sizes: transposed-convolution kernel per stage.
        gin_channels: speaker-conditioning channels.
        sr: output sample rate in Hz.
    """

    def __init__(self, initial_channel: int, resblock_kernel_sizes: list, resblock_dilation_sizes: list,
                 upsample_rates: list, upsample_initial_channel: int, upsample_kernel_sizes: list, gin_channels: int,
                 sr: int):
        super(HiFiGANNSFGenerator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)

        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        channels = [upsample_initial_channel // (2 ** (i + 1)) for i in range(len(upsample_rates))]
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1 for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            if u % 2 == 0:
                padding = (k - u) // 2
            else:
                padding = u // 2 + u % 2

            self.ups.append(
                weight_norm(
                    torch.nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        channels[i],
                        k,
                        u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )

            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2

            self.noise_convs.append(
                torch.nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        self.resblocks = torch.nn.ModuleList(
            [
                ResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.conv_post = torch.nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None):
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            x = ups(x)
            x = x + noise_convs(har_source)
            xs = sum(
                [
                    resblock(x)
                    for j, resblock in enumerate(self.resblocks)
                    if j in range(i * self.num_kernels, (i + 1) * self.num_kernels)
                ]
            )
            x = xs / self.num_kernels

        x = torch.nn.functional.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))

        return x
