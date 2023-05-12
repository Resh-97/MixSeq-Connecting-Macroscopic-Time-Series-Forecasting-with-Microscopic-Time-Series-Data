import torch
import torch.nn.functional as F

"""
casual convolution to create query and keys matrices
"""


class CasualConvolution1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CasualConvolution1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CasualConvolution1d, self).forward(F.pad(input, (self.__padding, 0)))
