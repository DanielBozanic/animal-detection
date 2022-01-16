import torch
import torch.nn as nn
from modules.detection_block import DetectionBlock
from modules.upsample import Upsample
from backbone_route_extractor import BackboneRouteExtractor

class YoloV3(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone_routes = BackboneRouteExtractor(self.backbone)
        self.detector1 = DetectionBlock(1024, 512)
        self.upmerger1 = Upsample(512, 256)
        self.detector2 = DetectionBlock(768, 256)
        self.upmerger2 = Upsample(256, 128)
        self.detector3 = DetectionBlock(384, 128, route=False)

    def forward(self, x):
        output = self.backbone(x)

        route1, output1 = self.detector1(output)
        output = self.upmerger1(route1)
        output = torch.cat([output, self.backbone_routes.get_routes[1]], 1)

        route2, output2 = self.detector2(output)
        output = self.upmerger2(route2)
        output = torch.cat([output, self.backbone_routes.get_routes[0]], 1)

        output3 = self.detector3(output)
        return [output1, output2, output3]
