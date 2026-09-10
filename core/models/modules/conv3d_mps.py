"""Evaluate a depth-preserving Conv3d as a batched Conv2d with the same weights.

Metal's large 3-D mask convolution is much slower than its 2-D implementation.
Unfolding the depth axis changes execution, not the learned operation. Floating
point accumulation can differ; pixel/quality controls must accompany adoption.
"""
import torch.nn.functional as F


def conv3d_as_conv2d(value, conv):
    if conv.stride[0] != 1 or conv.dilation != (1, 1, 1) or conv.groups != 1:
        raise ValueError('Expected ungrouped, undilated convolution with unit depth stride')
    batch, channels, depth, height, width = value.shape
    kd, kh, kw = conv.kernel_size
    pd, ph, pw = conv.padding
    padded = F.pad(value, (0, 0, 0, 0, pd, pd))
    windows = padded.unfold(2, kd, 1)
    output_depth = windows.shape[2]
    windows = windows.permute(0, 2, 1, 5, 3, 4).reshape(
        batch * output_depth, channels * kd, height, width)
    weight = conv.weight.reshape(conv.out_channels, channels * kd, kh, kw)
    output = F.conv2d(windows, weight, conv.bias, stride=conv.stride[1:], padding=(ph, pw))
    return output.reshape(batch, output_depth, conv.out_channels,
                          output.shape[-2], output.shape[-1]).permute(0, 2, 1, 3, 4)
