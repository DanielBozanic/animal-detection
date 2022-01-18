import cv2
import torch
from yolov3_detector import *
from utils import *


def get_annotated_data():
    data = dict()
    f = open("data/annotation_data.csv", "r")
    for l in f.readlines():
        parts = l.strip("\n").split(",")
        x, y, width, height, class_id, image_path = parts
        if image_path not in data:
            data[image_path] = []
        box = torch.Tensor([float(x), float(y), float(width), float(height), int(class_id)])
        data[image_path].append(box)
    f.close()
    return data


def evaluate_pytorch_yoloV3(model, device, iou_thresh=0.5):
    total_detections = 0
    total_true_positives_detections = 0
    total_false_positives_detections_image = 0
    total_false_negatives_detections = 0

    annotated_data = get_annotated_data()
    for image_path in annotated_data:
        total_detections_image = 0
        true_positives_detections_image = 0
        false_positives_detections_image = 0
        false_negatives_detections_image = 0
        image = cv2.imread(image_path)
        detector = YoloV3Detector(model, image, device, 0.65, 0.4, 416)
        predicted_bboxes = detector.predict()
        valid_boxes = non_maximum_suppression(predicted_bboxes, 0.5).cpu()
        new_valid_boxes = []
        for valid_box in valid_boxes:
            new_valid_boxes.append((valid_box[0], valid_box[1], valid_box[2], valid_box[3]))
        for ground_truth_box in annotated_data[image_path]:
            total_detections += 1
            total_detections_image += 1
            bbox_index = 0
            best_iou = 0
            ground_truth_box = (ground_truth_box[0], ground_truth_box[1], ground_truth_box[0] +
                                ground_truth_box[2], ground_truth_box[1] + ground_truth_box[3])
            for index, predicted_bbox in enumerate(new_valid_boxes):
                iou = intersection_over_union(ground_truth_box, predicted_bbox)
                if iou > best_iou:
                    best_iou = iou
                    bbox_index = index

            if best_iou >= iou_thresh:
                total_true_positives_detections += 1
                true_positives_detections_image += 1
                del new_valid_boxes[bbox_index]
            else:
                total_false_negatives_detections += 1
                false_negatives_detections_image += 1

        total_false_positives_detections_image += len(new_valid_boxes)
        false_positives_detections_image += len(new_valid_boxes)

        detection_accuracy_image = true_positives_detections_image / total_detections_image
        print("Detection accuracy for test image " + image_path + ":", str(round(detection_accuracy_image, 3)))

        detection_precision_image = true_positives_detections_image / (true_positives_detections_image + false_positives_detections_image)
        print("Detection precision for test image " + image_path + ":", str(round(detection_precision_image, 3)) + "\n")

    total_detection_accuracy = total_true_positives_detections / total_detections
    print("Total detection accuracy for all test images: ", str(round(total_detection_accuracy, 3)))

    total_detection_precision = total_true_positives_detections / (total_true_positives_detections + total_false_positives_detections_image)
    print("Total detection precision for all test images: ", str(round(total_detection_precision, 3)))
