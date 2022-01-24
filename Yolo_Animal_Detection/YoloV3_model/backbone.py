import torch.nn as nn
from YoloV3_model.modules.convolutional_layer import ConvolutionalLayer
from YoloV3_model.modules.residual_block import ResidualBlock

class DarknetBlock(nn.Module):
    def __init__(self, filters_in, length):
        super().__init__()
        self.__filters_in = filters_in
        self.__length = length
        self.conv_down = ConvolutionalLayer(self.__filters_in, self.__filters_in*2, 3, 2, 3//2)
        self.res_blocks = nn.Sequential(*self.__form_residual_blocks())

    def forward(self, x):
        output = self.conv_down(x)
        output = self.res_blocks(output)
        return output

    def __form_residual_blocks(self):
        res_blocks = []
        for i in range(self.__length):
            res_blocks.append(ResidualBlock(self.__filters_in*2))
        return res_blocks


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.__filter_in_sizes = [32, 64, 128, 256, 512]
        self.__lengths = [1, 2, 8, 8, 4]
        conv1 = ConvolutionalLayer(3, 32, kernel_size=3, padding=3//2)
        darknet_blocks = self.__form_darknet_blocks()
        self.extractor = nn.Sequential(conv1, *darknet_blocks)

    def forward(self, x):
        return self.extractor(x)

    def __form_darknet_blocks(self):
        darknet_blocks = []
        for filters_in, length in zip(self.__filter_in_sizes, self.__lengths):
            darknet_blocks.append(DarknetBlock(filters_in, length))
        return darknet_blocks
