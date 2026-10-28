"""RVC v2 multi-period discriminator (from Applio rvc/lib/algorithm/discriminators.py; only the v2
period set, without the multi-resolution discriminator RefineGAN uses or gradient checkpointing)."""
import torch
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.commons import get_padding
from rvc.residuals import LRELU_SLOPE

PERIODS_V2 = [2, 3, 5, 7, 11, 17, 23, 37]


class MultiPeriodDiscriminator(torch.nn.Module):
    """One scale discriminator on the raw waveform plus one period discriminator per period.

    forward(y, y_hat) returns (real scores, generated scores, real feature maps, generated feature maps),
    one entry per sub-discriminator."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in PERIODS_V2]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """Scale discriminator: grouped strided 1-D convolutions over the waveform."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(torch.nn.Module):
    """Period discriminator: the waveform folded into [time / period, period], 2-D convolutions along time.

    Args:
        period: folding period in samples.
        kernel_size: convolution kernel along time.
        stride: convolution stride along time.
        use_spectral_norm: spectral instead of weight normalisation.
    """

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.conv_post = norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap
