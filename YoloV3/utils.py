import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2


def read_classes(file_path):
    with open(file_path, 'r') as f:
        classes = f.read().split('\n')[:-1]
    return classes


def intersection_over_union(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if (x1 >= x2) or (y1 >= y2):
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    box_area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box_area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = box_area1 + box_area2 - intersection
    return intersection/union


def non_maximum_suppression(boxes, thresh=0.5):
    if boxes.size(0) == 0:
        return boxes

    confidence = boxes[:, 4]
    _, conf_indices = torch.sort(confidence, descending=True)

    for counter, value_i in enumerate(conf_indices):
        if boxes[value_i][4] > 0:
            for value_j in conf_indices[(counter+1):]:
                iou = intersection_over_union(boxes[value_i], boxes[value_j])
                if iou > thresh and boxes[value_i][-1] == boxes[value_j][-1]:
                    boxes[value_j][4] = 0

    return boxes[boxes[:, 4] > 0]