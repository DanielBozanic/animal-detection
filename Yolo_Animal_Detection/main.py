import os
from evaluate_yolov3 import *
from evaluate_prebuilt import *
from YoloV3_model.backbone import Backbone
from YoloV3_model.yolov3 import YoloV3
from yolo_detector_prebuilt_image import yolo_prebuilt_image
from yolo_detector_prebuilt_video import yolo_prebuilt_video


if __name__ == '__main__':
    device = torch.device('cpu')

    with torch.no_grad():
        backbone = Backbone()
        backbone = backbone.extractor

        yoloV3 = YoloV3(backbone)
        yoloV3.load_state_dict(torch.load('data/weights.pth'))
        yoloV3.to(device)

        classes = read_classes(CLASSES)

        print("------- YoloV3 -------")
        print("1. YoloV3 object detection on image")
        print("2. YoloV3 object detection on video")
        print("3. Evaluate YoloV3 performance\n\n")
        print("------- YoloV3-Tiny -------")
        print("4. YoloV3-Tiny object detection on image")
        print("5. YoloV3-Tiny object detection on video")
        print("6. Evaluate YoloV3-Tiny performance\n\n")
        print("------- YoloV4 -------")
        print("7. YoloV4 object detection on image")
        print("8. YoloV4 object detection on video")
        print("9. Evaluate YoloV4 performance\n\n")
        print("------- YoloV4-Tiny -------")
        print("10. YoloV4-Tiny object detection on image")
        print("11. YoloV4-Tiny object detection on video")
        print("12. Evaluate YoloV4-Tiny performance\n\n")
        choice = input()

        # Image
        if choice == "1":
            print("Image path: ")
            image_path = input()
            image = cv2.imread(image_path)
            if image is None:
                print("Image not found!")
            else:
                detector = YoloV3Detector(yoloV3, image, device)
                predicted_bboxes = detector.predict()
                draw_boxes_on_image(image, predicted_bboxes, classes, image.shape[1] / YOLO_SIZE,
                                             image.shape[0] / YOLO_SIZE)
                cv2.imshow('YOLOV3 Image Detection', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        # Video
        elif choice == "2":
            print("Video path: ")
            video_path = input()
            if not os.path.isfile(video_path):
                print("Video not found!")
            else:
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    _, image = cap.read()
                    if image is not None:
                        detector = YoloV3Detector(yoloV3, image, device)
                        predicted_bboxes = detector.predict()
                        draw_boxes_on_image(image, predicted_bboxes, classes, image.shape[1] / YOLO_SIZE,
                                                     image.shape[0] / YOLO_SIZE)
                        cv2.imshow('YOLOV3 Video Detection', image)
                    else:
                        break
                    k = cv2.waitKey(10)
                    if k == 27:
                        break
                cap.release()
                cv2.destroyAllWindows()
        # Evaluate YoloV3 performance
        elif choice == "3":
            ev = EvaluateYoloV3()
            ev.evaluate_pytorch_yoloV3(yoloV3, device)
        # YoloV3-tiny image
        elif choice == "4":
            print("Image path: ")
            image_path = input()
            yolo_prebuilt_image("yolov3-tiny", image_path)
        # YoloV3-tiny video
        elif choice == "5":
            print("Video path: ")
            video_path = input()
            yolo_prebuilt_video("yolov3-tiny", video_path)
        # Evaluate YoloV3-tiny performance
        elif choice == "6":
            ev = EvaluateYoloPrebuilt()
            ev.evaluate_prebuilt_yolo("yolov3-tiny")
         # YoloV4 image
        elif choice == "7":
            print("Image path: ")
            image_path = input()
            yolo_prebuilt_image("yolov4", image_path)
        # YoloV4 video
        elif choice == "8":
            print("Video path: ")
            video_path = input()
            yolo_prebuilt_video("yolov4", video_path)
        # Evaluate YoloV4 performance
        elif choice == "9":
            ev = EvaluateYoloPrebuilt()
            ev.evaluate_prebuilt_yolo("yolov4")
        # YoloV4-tiny image
        elif choice == "10":
            print("Image path: ")
            image_path = input()
            yolo_prebuilt_image("yolov4-tiny", image_path)
        # YoloV4-tiny video
        elif choice == "11":
            print("Video path: ")
            video_path = input()
            yolo_prebuilt_video("yolov4-tiny", video_path)
        # Evaluate YoloV4-tiny performance
        elif choice == "12":
            ev = EvaluateYoloPrebuilt()
            ev.evaluate_prebuilt_yolo("yolov4-tiny")
        else:
            print("Invalid input!")
