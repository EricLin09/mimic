"""Small tensor helpers shared by the networks and the trainer (from Applio rvc/lib/algorithm/commons.py)."""
from typing import Optional

import torch


def init_weights(m, mean=0.0, std=0.01):
    """Normal init for every convolution in `m` (applied with Module.apply)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """'Same' padding of a dilated convolution."""
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    """[[before, after], ...] per dimension, first dimension first -> F.pad's flat, last-first list."""
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4, dim: int = 2):
    """Per batch item, `segment_size` steps from ids_str[i] along the last axis of a 2-D (dim=2) or 3-D
    (dim=3) tensor."""
    if dim == 2:
        ret = torch.zeros_like(x[:, :segment_size])
    elif dim == 3:
        ret = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size
        if dim == 2:
            ret[i] = x[i, idx_str:idx_end]
        else:
            ret[i] = x[i, :, idx_str:idx_end]

    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """A random `segment_size` window inside each item's valid length; returns (slices, start indices)."""
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size, dim=3)
    return ret, ids_str


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    """[batch, max_length] boolean mask, True inside each sequence's length."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def grad_norm(parameters, norm_type: float = 2.0):
    """Total gradient norm over the parameters that have a gradient (logging only, nothing is clipped)."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if not parameters:
        return 0.0

    return torch.linalg.vector_norm(
        torch.stack([p.grad.norm(norm_type) for p in parameters]), ord=norm_type
    ).item()
