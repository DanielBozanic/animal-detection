from yolov3_detector import *
from utils import *


class EvaluateYoloV3:

    def __init__(self):
        self.__classes = read_classes('data/coco.names')
        self.__initalize_total_variables()

    def __initalize_total_variables(self):
        self.__total_true_positives_detections = 0
        self.__total_false_negatives_detections = 0
        self.__total_false_positives_detections = 0
        self.__total_true_positives_classification = dict()
        self.__total_false_positives_classification = dict()

    def __initalize_image_variables(self):
        self.__true_positives_detections_image = 0
        self.___false_negatives_detections_image = 0
        self.__false_positives_detections_image = 0
        self.__true_positives_classification_image = dict()
        self.__false_positives_classification_image = dict()

    def __get_annotated_data(self):
        data = dict()
        f = open("data/annotation_data.csv", "r")
        for l in f.readlines():
            parts = l.strip("\n").split(",")
            x, y, width, height, class_id, image_path = parts
            if image_path not in data:
                data[image_path] = []
            try:
                box = (float(x), float(y), float(width), float(height), int(class_id))
                data[image_path].append(box)
            except Exception as e:
                print(e)
        f.close()
        return data

    def __print_image_evaluation(self, image_path):
        print("IMAGE: " + image_path + "\n")

        detection_accuracy_image = self.__true_positives_detections_image / (self.__true_positives_detections_image +
            self.__false_positives_detections_image + self.___false_negatives_detections_image)
        print("Detection accuracy:", str(round(detection_accuracy_image, 3)))

        if self.__true_positives_detections_image > 0:
            detection_precision_image = self.__true_positives_detections_image / (
                        self.__true_positives_detections_image + self.__false_positives_detections_image)
            print("Detection precision:", str(round(detection_precision_image, 3)))

        classification_precision_image = dict()
        for class_id in [14, 15, 16, 20, 22, 23]:
            if class_id not in self.__true_positives_classification_image or \
                    class_id not in self.__false_positives_classification_image:
                continue
            if self.__true_positives_classification_image[class_id] > 0:
                classification_precision_image[class_id] = self.__true_positives_classification_image[class_id] / \
                                                       (self.__true_positives_classification_image[class_id] +
                                                        self.__false_positives_classification_image[class_id])
                print("Precision classification for class "
                      + self.__classes[class_id] + ":", str(round(classification_precision_image[class_id], 3)))

        print("=============================================="
              "===============================================\n")

    def __print_overall_evaluation(self):
        total_detection_accuracy = self.__total_true_positives_detections / (self.__total_true_positives_detections +
            self.__total_false_negatives_detections + self.__total_false_positives_detections)
        print("Total detection accuracy for all test images: ", str(round(total_detection_accuracy, 3)))

        if self.__total_true_positives_detections > 0:
            total_detection_precision = self.__total_true_positives_detections / (
                        self.__total_true_positives_detections + self.__total_false_positives_detections)
            print("Total detection precision for all test images: ", str(round(total_detection_precision, 3)))

        total_classification_precision = dict()
        for class_id in [14, 15, 16, 20, 22, 23]:
            if class_id not in self.__total_true_positives_classification or \
                    class_id not in self.__total_false_positives_classification:
                continue
            if self.__total_true_positives_classification[class_id] > 0:
                total_classification_precision[class_id] = self.__total_true_positives_classification[class_id] / \
                                                       (self.__total_true_positives_classification[class_id] +
                                                        self.__total_false_positives_classification[class_id])
                print("Precision classification across all test images for class " + self.__classes[class_id] + ":",
                      str(round(total_classification_precision[class_id], 3)))

    def evaluate_pytorch_yoloV3(self, model, device, iou_thresh=0.5):
        print("PYTORCH YOLOV3 EVALUATION")
        print("=============================================="
              "===============================================\n")

        self.__initalize_image_variables()
        self.__initalize_total_variables()
        annotated_data = self.__get_annotated_data()
        for image_path in annotated_data:
            self.__initalize_image_variables()
            image = cv2.imread(image_path)
            detector = YoloV3Detector(model, image, device, 0.3, 0.5, 416)
            predicted_bboxes = detector.predict()
            new_predicted_bboxes = []
            for box in predicted_bboxes:
                new_predicted_bboxes.append((box[0].item(), box[1].item(),
                                             box[2].item(), box[3].item(),
                                             box[6].item()))
            for ground_truth_box in annotated_data[image_path]:
                bbox_index = 0
                best_iou = 0
                ground_truth_box = (ground_truth_box[0], ground_truth_box[1],
                                    ground_truth_box[0] + ground_truth_box[2],
                                    ground_truth_box[1] + ground_truth_box[3],
                                    ground_truth_box[4])
                for index, predicted_bbox in enumerate(new_predicted_bboxes):
                    iou = intersection_over_union(ground_truth_box, predicted_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        bbox_index = index

                if ground_truth_box[4] not in self.__total_true_positives_classification:
                    self.__total_true_positives_classification[ground_truth_box[4]] = 0
                if ground_truth_box[4] not in self.__total_false_positives_classification:
                    self.__total_false_positives_classification[ground_truth_box[4]] = 0
                if ground_truth_box[4] not in self.__true_positives_classification_image:
                    self.__true_positives_classification_image[ground_truth_box[4]] = 0
                if ground_truth_box[4] not in self.__false_positives_classification_image:
                    self.__false_positives_classification_image[ground_truth_box[4]] = 0

                if best_iou >= iou_thresh:
                    self.__total_true_positives_detections += 1
                    self.__true_positives_detections_image += 1
                    if ground_truth_box[4] == new_predicted_bboxes[bbox_index][4]:
                        self.__total_true_positives_classification[ground_truth_box[4]] += 1
                        self.__true_positives_classification_image[ground_truth_box[4]] += 1
                    else:
                        self.__total_false_positives_classification[ground_truth_box[4]] += 1
                        self.__false_positives_classification_image[ground_truth_box[4]] += 1
                    del new_predicted_bboxes[bbox_index]
                else:
                    self.__total_false_negatives_detections += 1
                    self.___false_negatives_detections_image += 1

            self.__total_false_positives_detections += len(new_predicted_bboxes)
            self.__false_positives_detections_image += len(new_predicted_bboxes)

            self.__print_image_evaluation(image_path)

        self.__print_overall_evaluation()


class EvaluateYoloPrebuilt:

    def __init__(self):
        self.__classes = read_classes('data/coco.names')
        self.__yolo_size = 416;

    def __initalize_total_variables(self):
        self.__total_true_positives_detections = 0
        self.__total_false_negatives_detections = 0
        self.__total_false_positives_detections = 0
        self.__total_true_positives_classification = dict()
        self.__total_false_positives_classification = dict()

    def __initalize_image_variables(self):
        self.__true_positives_detections_image = 0
        self.___false_negatives_detections_image = 0
        self.__false_positives_detections_image = 0
        self.__true_positives_classification_image = dict()
        self.__false_positives_classification_image = dict()

    def __get_annotated_data(self):
        data = dict()
        f = open("data/annotation_data.csv", "r")
        for l in f.readlines():
            parts = l.strip("\n").split(",")
            x, y, width, height, class_id, image_path = parts
            if image_path not in data:
                data[image_path] = []
            try:
                box = (float(x), float(y), float(width), float(height), int(class_id))
                data[image_path].append(box)
            except Exception as e:
                print(e)
        f.close()
        return data

    def __print_image_evaluation(self, image_path):
        print("IMAGE: " + image_path + "\n")

        detection_accuracy_image = self.__true_positives_detections_image / (self.__true_positives_detections_image +
                                                                             self.__false_positives_detections_image + self.___false_negatives_detections_image)
        print("Detection accuracy:", str(round(detection_accuracy_image, 3)))

        if self.__true_positives_detections_image > 0:
            detection_precision_image = self.__true_positives_detections_image / (
                    self.__true_positives_detections_image + self.__false_positives_detections_image)
            print("Detection precision:", str(round(detection_precision_image, 3)))

        classification_precision_image = dict()
        for class_id in [14, 15, 16, 20, 22, 23]:
            if class_id not in self.__true_positives_classification_image or \
                    class_id not in self.__false_positives_classification_image:
                continue
            if self.__true_positives_classification_image[class_id] > 0:
                classification_precision_image[class_id] = self.__true_positives_classification_image[class_id] / \
                                                           (self.__true_positives_classification_image[class_id] +
                                                            self.__false_positives_classification_image[class_id])
                print("Precision classification for class "
                      + self.__classes[class_id] + ":", str(round(classification_precision_image[class_id], 3)))

        print("=============================================="
              "===============================================\n")

    def __print_overall_evaluation(self):
        total_detection_accuracy = self.__total_true_positives_detections / (self.__total_true_positives_detections +
            self.__total_false_negatives_detections + self.__total_false_positives_detections)
        print("Total detection accuracy for all test images: ", str(round(total_detection_accuracy, 3)))

        if self.__total_true_positives_detections > 0:
            total_detection_precision = self.__total_true_positives_detections / (
                        self.__total_true_positives_detections + self.__total_false_positives_detections)
            print("Total detection precision for all test images: ", str(round(total_detection_precision, 3)))

        total_classification_precision = dict()
        for class_id in [14, 15, 16, 20, 22, 23]:
            if class_id not in self.__total_true_positives_classification or \
                    class_id not in self.__total_false_positives_classification:
                continue
            if self.__total_true_positives_classification[class_id] > 0:
                total_classification_precision[class_id] = self.__total_true_positives_classification[class_id] / \
                                                       (self.__total_true_positives_classification[class_id] +
                                                        self.__total_false_positives_classification[class_id])
                print("Precision classification across all test images for class " + self.__classes[class_id] + ":",
                      str(round(total_classification_precision[class_id], 3)))

    def evaluate_prebuilt_yolo(self, type, iou_thresh=0.5):
        print("YOLO PREBUILT EVALUATION")
        print("=============================================="
              "===============================================\n")
        cfg_path = './data/'+type+'.cfg'
        weights_path = './data/' + type + '.weights'
        neural_network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.__initalize_image_variables()
        self.__initalize_total_variables()
        annotated_data = self.__get_annotated_data()
        for image_path in annotated_data:
            self.__initalize_image_variables()
            image = cv2.imread(image_path)
            if image is None:
                print("Image not found!")
                break
            blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.__yolo_size, self.__yolo_size), True, crop=False)
            neural_network.setInput(blob)
            layers = neural_network.getLayerNames()
            output_names = \
                [layers[idx[0] - 1] for idx in neural_network.getUnconnectedOutLayers()]
            outputs = neural_network.forward(output_names)
            bbox_locations, conf_values = detect_objects_prebuilt(outputs)
            for ground_truth_box in annotated_data[image_path]:
                bbox_index = 0
                best_iou = 0
                ground_truth_box = (ground_truth_box[0], ground_truth_box[1],
                                    ground_truth_box[0] + ground_truth_box[2],
                                    ground_truth_box[1] + ground_truth_box[3],
                                    ground_truth_box[4])
                for index, predicted_bbox in enumerate(bbox_locations):
                    predicted_bbox = (predicted_bbox[0], predicted_bbox[1],
                                      predicted_bbox[0] + predicted_bbox[2],
                                      predicted_bbox[1] + predicted_bbox[3],
                                      predicted_bbox[4])
                    iou = intersection_over_union(ground_truth_box, predicted_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        bbox_index = index

                if ground_truth_box[4] not in self.__total_true_positives_classification:
                    self.__total_true_positives_classification[ground_truth_box[4]] = 0
                if ground_truth_box[4] not in self.__total_false_positives_classification:
                    self.__total_false_positives_classification[ground_truth_box[4]] = 0
                if ground_truth_box[4] not in self.__true_positives_classification_image:
                    self.__true_positives_classification_image[ground_truth_box[4]] = 0
                if ground_truth_box[4] not in self.__false_positives_classification_image:
                    self.__false_positives_classification_image[ground_truth_box[4]] = 0

                if best_iou >= iou_thresh:
                    self.__total_true_positives_detections += 1
                    self.__true_positives_detections_image += 1
                    if ground_truth_box[4] == bbox_locations[bbox_index][4]:
                        self.__total_true_positives_classification[ground_truth_box[4]] += 1
                        self.__true_positives_classification_image[ground_truth_box[4]] += 1
                    else:
                        self.__total_false_positives_classification[ground_truth_box[4]] += 1
                        self.__false_positives_classification_image[ground_truth_box[4]] += 1
                    del bbox_locations[bbox_index]
                else:
                    self.__total_false_negatives_detections += 1
                    self.___false_negatives_detections_image += 1

            self.__total_false_positives_detections += len(bbox_locations)
            self.__false_positives_detections_image += len(bbox_locations)

            self.__print_image_evaluation(image_path)

        self.__print_overall_evaluation()