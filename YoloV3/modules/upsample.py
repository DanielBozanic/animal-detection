import torch.nn as nn
import torch.nn.functional as F
from modules.convolutional_layer import ConvolutionalLayer

class Upsample(nn.Module):
    def __init__(self, filters_in, filters_out=0):
        super().__init__()
        self.conv = ConvolutionalLayer(filters_in, filters_out, kernel_size=1)

    def forward(self, route_head):
        return F.interpolate(self.conv(route_head), scale_factor=2)