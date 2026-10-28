"""Content (prior) encoder and posterior encoder (from Applio rvc/lib/algorithm/encoders.py)."""
import math
from typing import Optional

import torch

from rvc.attentions import FFN, MultiHeadAttention
from rvc.commons import sequence_mask
from rvc.modules import WaveNet
from rvc.normalization import LayerNorm


class Encoder(torch.nn.Module):
    """Transformer encoder: relative-position self-attention and conv FFN, post-norm.

    Args:
        hidden_channels: model width.
        filter_channels: FFN inner width.
        n_heads: attention heads.
        n_layers: encoder layers.
        kernel_size: FFN convolution kernel size.
        p_dropout: dropout.
        window_size: relative-position window of the attention.
    """

    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int, n_layers: int, kernel_size: int = 1,
                 p_dropout: float = 0.0, window_size: int = 10):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = torch.nn.Dropout(p_dropout)

        self.attn_layers = torch.nn.ModuleList(
            [
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size)
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_1 = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.ffn_layers = torch.nn.ModuleList(
            [FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout) for _ in range(n_layers)]
        )
        self.norm_layers_2 = torch.nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        return x * x_mask


class TextEncoder(torch.nn.Module):
    """Prior encoder: content features [B, T, embedding_dim] plus the coarse pitch embedding ->
    mean and log-std of the latent prior, [B, out_channels, T] each.

    Args:
        out_channels: latent channels.
        hidden_channels: encoder width.
        filter_channels: FFN inner width.
        n_heads: attention heads.
        n_layers: encoder layers.
        kernel_size: FFN kernel size.
        p_dropout: dropout.
        embedding_dim: content feature width (768 for RVC v2).
    """

    def __init__(self, out_channels: int, hidden_channels: int, filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int, p_dropout: float, embedding_dim: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.emb_phone = torch.nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = torch.nn.Embedding(256, hidden_channels)

        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor, lengths: torch.Tensor):
        x = self.emb_phone(phone)
        x += self.emb_pitch(pitch)

        x *= math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = x.transpose(1, -1)  # [B, H, T]

        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class PosteriorEncoder(torch.nn.Module):
    """Training-only encoder: linear spectrogram -> sampled latent z, its mean and log-std.

    Args:
        in_channels: spectrogram bins.
        out_channels: latent channels.
        hidden_channels: WaveNet channels.
        kernel_size: WaveNet kernel size.
        dilation_rate: WaveNet dilation base.
        n_layers: WaveNet layers.
        gin_channels: speaker-conditioning channels.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int,
                 n_layers: int, gin_channels: int = 0):
        super().__init__()
        self.out_channels = out_channels
        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, g: Optional[torch.Tensor] = None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        z = m + torch.randn_like(m) * torch.exp(logs)
        z *= x_mask

        return z, m, logs, x_mask
