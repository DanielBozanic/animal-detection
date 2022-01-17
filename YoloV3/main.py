import os
import cv2
import torch
from yolov3_detector import YoloV3Detector
from utils import *
from backbone import Backbone
from yolov3 import YoloV3

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        backbone = Backbone()
        backbone = backbone.extractor

        yoloV3 = YoloV3(backbone)
        yoloV3.load_state_dict(torch.load('data/weights.pth'))
        yoloV3.to(device)

        print("1. YoloV3 object detection on image")
        print("2. YoloV3 object detection on video")
        choice = input()

        # Image
        if choice == "1":
            print("Image path: ")
            image_path = input()
            image = cv2.imread(image_path)
            if image is None:
                print("Image not found!")
            else:
                detector = YoloV3Detector(yoloV3, image, device, 0.65, 0.4, 416)
                predicted_bboxes = detector.predict()
                boxes = non_maximum_suppression(predicted_bboxes, 0.5).cpu()
                classes = read_classes('data/coco.names')
                result = detector.draw_boxes_on_image(boxes, classes)
                cv2.imshow('YoloV3 Image Detection', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        # Video
        elif choice == "2":
            print("Video path: ")
            video_path = input()
            if not os.path.isfile(video_path):
                print("Video not found!")
            else:
                classes = read_classes('data/coco.names')
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    _, image = cap.read()
                    if image is not None:
                        detector = YoloV3Detector(yoloV3, image, device, 0.65, 0.4, 416)
                        predicted_bboxes = detector.predict()
                        boxes = non_maximum_suppression(predicted_bboxes, 0.5).cpu()
                        result = detector.draw_boxes_on_image(boxes, classes)
                        cv2.imshow('YoloV3 Video Detection', result)
                    else:
                        break
                    k = cv2.waitKey(10)
                    if k == 27:
                        break
                cap.release()
                cv2.destroyAllWindows()
        else:
            print("Invalid input!")
