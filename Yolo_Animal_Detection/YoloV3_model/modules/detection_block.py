import torch.nn as nn
from YoloV3_model.modules.convolutional_layer import ConvolutionalLayer

class DetectionBlock(nn.Module):
    def __init__(self, filters_in, filters_out, route=True):
        super().__init__()
        self.route = route
        self.conv1 = ConvolutionalLayer(filters_in, filters_out, kernel_size=1)
        self.conv2 = ConvolutionalLayer(filters_out, filters_out * 2, kernel_size=3, padding=3 // 2)
        self.conv3 = ConvolutionalLayer(filters_out * 2, filters_out, kernel_size=1)
        self.conv4 = ConvolutionalLayer(filters_out, filters_out * 2, kernel_size=3, padding=3 // 2)
        self.conv5 = ConvolutionalLayer(filters_out * 2, filters_out, kernel_size=1)
        self.conv6 = ConvolutionalLayer(filters_out, filters_out * 2, kernel_size=3, padding=3 // 2)
        self.conv7 = nn.Conv2d(filters_out * 2, 255, 1, 1)

    def forward(self, x):
        route = self.conv1(x)
        route = self.conv2(route)
        route = self.conv3(route)
        route = self.conv4(route)
        route = self.conv5(route)
        output = self.conv6(route)
        output = self.conv7(output)
        if self.route:
            return route, output
        else:
            return output