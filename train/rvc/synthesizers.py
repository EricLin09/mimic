"""RVC v2 synthesizer with pitch guidance and the NSF HiFi-GAN vocoder
(from Applio rvc/lib/algorithm/synthesizers.py, that configuration only)."""
import torch

from rvc.commons import rand_slice_segments, slice_segments
from rvc.encoders import PosteriorEncoder, TextEncoder
from rvc.hifigan_nsf import HiFiGANNSFGenerator
from rvc.residuals import ResidualCouplingBlock


class Synthesizer(torch.nn.Module):
    """VITS-style conversion model: content encoder (enc_p) -> flow -> NSF HiFi-GAN decoder (dec), with
    the posterior encoder (enc_q) used only in training.

    The positional arguments are the `config` list stored in RVC / Applio checkpoints, so
    `Synthesizer(*cpt["config"])` rebuilds a saved model.

    Args:
        spec_channels: linear spectrogram bins seen by enc_q (n_fft // 2 + 1).
        segment_size: training slice length in frames.
        inter_channels: latent channels.
        hidden_channels: encoder width.
        filter_channels: encoder FFN inner width.
        n_heads: attention heads.
        n_layers: encoder layers.
        kernel_size: encoder FFN kernel size.
        p_dropout: dropout.
        resblock: kept for checkpoint compatibility; the vocoder always uses ResBlock1.
        resblock_kernel_sizes: vocoder ResBlock kernels.
        resblock_dilation_sizes: vocoder ResBlock dilations.
        upsample_rates: vocoder upsampling factors (their product is the hop).
        upsample_initial_channel: vocoder width.
        upsample_kernel_sizes: vocoder transposed-convolution kernels.
        spk_embed_dim: speaker embedding rows.
        gin_channels: speaker embedding width.
        sr: output sample rate in Hz.
        text_enc_hidden_dim: content feature width (768 for v2).
    """

    def __init__(self, spec_channels: int, segment_size: int, inter_channels: int, hidden_channels: int,
                 filter_channels: int, n_heads: int, n_layers: int, kernel_size: int, p_dropout: float, resblock: str,
                 resblock_kernel_sizes: list, resblock_dilation_sizes: list, upsample_rates: list,
                 upsample_initial_channel: int, upsample_kernel_sizes: list, spk_embed_dim: int, gin_channels: int,
                 sr: int, text_enc_hidden_dim: int = 768):
        super().__init__()
        self.segment_size = segment_size

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
        )
        self.dec = HiFiGANNSFGenerator(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(spk_embed_dim, gin_channels)

    def forward(self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds):
        """Training pass: decodes a random `segment_size` slice of the posterior latent.

        Returns (audio slice, slice start frames, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q))."""
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.no_grad()
    def infer(self, phone, phone_lengths, pitch, nsff0, sid):
        """Whole-utterance conversion with the prior's noise scale 0.66666, as RVC renders."""
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        return self.dec(z * x_mask, nsff0, g=g)
