import torch.nn as nn
from modules.convolutional_layer import ConvolutionalLayer

class ResidualBlock(nn.Module):
    def __init__(self, filters_in):
        super().__init__()
        self.conv_reduce = ConvolutionalLayer(filters_in, filters_in//2, 1)
        self.conv_expand = ConvolutionalLayer(filters_in//2, filters_in, 3, padding=3//2)

    def forward(self, x):
        output = self.conv_reduce(x)
        output = self.conv_expand(output)
        return x + output
