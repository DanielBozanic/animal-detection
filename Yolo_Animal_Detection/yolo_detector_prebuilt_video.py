import cv2
import numpy as np
from utils import *

def yolo_prebuilt_video(type, path):
    cfg = DATA_FOLDER + type + '.cfg'
    weights = DATA_FOLDER + type + '.weights'
    labels = read_classes('data/coco.names')

    neural_network = cv2.dnn.readNetFromDarknet(cfg, weights)
    neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    video_capture = cv2.VideoCapture(path) #video processing
    if video_capture is None:
        print("Video not found!")
        return
    while video_capture.isOpened():
        is_grab, frame = video_capture.read() # citaj svaki frame videa
        original_width, original_height = frame.shape[1], frame.shape[0]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (YOLO_SIZE, YOLO_SIZE), True, crop=False)
        neural_network.setInput(blob)
        layer_names = neural_network.getLayerNames()
        output_names = [layer_names[idx[0] - 1] for idx in neural_network.getUnconnectedOutLayers()]
        outputs = neural_network.forward(output_names) # forward propagation

        bbox_locations, conf_values = detect_objects_prebuilt(outputs) # uzmi stuff iz prediction vector
        draw_boxes_on_image_prebuilt(frame, bbox_locations, labels, conf_values,
                             original_width / YOLO_SIZE, original_height / YOLO_SIZE, COLORS)

        cv2.imshow('YOLO Algorithm', frame)
        key = cv2.waitKey(1) & 0xff
        if (key == 27) | (not is_grab):
            break

    video_capture.release()
    cv2.destroyAllWindows()
