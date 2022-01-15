import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalLayer(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(filters_out, momentum=0.01)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)
