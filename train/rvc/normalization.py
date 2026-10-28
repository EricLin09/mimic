"""Channel LayerNorm (from Applio rvc/lib/algorithm/normalization.py)."""
import torch


class LayerNorm(torch.nn.Module):
    """LayerNorm over the channel axis of [batch, channels, time].

    The normalized shape is the stored channel count rather than x.size(-1): the same numbers, but the
    ONNX tracer cannot export layer_norm whose shape comes from x.size(-1) (the export scripts used to
    patch this in)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
