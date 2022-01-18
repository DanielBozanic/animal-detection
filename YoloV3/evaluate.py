import cv2
import torch
from yolov3_detector import *
from utils import *


def get_annotated_data():
    data = dict()
    f = open("data/annotation_data.csv", "r")
    for line in f.readlines():
        tokens = line.strip("\n").split(",")
        x, y, width, height, class_id, image_name = tokens
        if image_name not in data:
            data[image_name] = []
        box = torch.Tensor([float(x), float(y), float(width), float(height), int(class_id)])
        data[image_name].append(box)
    f.close()
    return data


def evaluate_pytorch_yoloV3(model, device, iou_thresh=0.5):
    total_detections = 0
    total_successful_detections = 0

    annotated_data = get_annotated_data()
    for key in annotated_data.keys():
        total_detections_image = 0
        successful_detections_image = 0
        image = cv2.imread(key)
        detector = YoloV3Detector(model, image, device, 0.65, 0.4, 416)
        predicted_bboxes = detector.predict()
        for truth_box in annotated_data[key]:
            total_detections += 1
            total_detections_image += 1
            best_iou = 0
            truth_box = (truth_box[0], truth_box[1], truth_box[0] + truth_box[2], truth_box[1] + truth_box[3])
            for predicted_bbox in predicted_bboxes:
                iou = intersection_over_union(truth_box, predicted_bbox)
                if iou > best_iou:
                    best_iou = iou

            if best_iou >= iou_thresh:
                total_successful_detections += 1
                successful_detections_image += 1

        detection_accuracy_image = successful_detections_image / total_detections_image
        print("Detection accuracy for test image " + key + ":", str(round(detection_accuracy_image, 3)) + "\n")

    total_detection_accuracy = total_successful_detections / total_detections
    print("Total detection accuracy for all test images: ", str(round(total_detection_accuracy, 3)) + "\n")
