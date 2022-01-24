import cv2
import numpy as np
from utils import *


def yolo_prebuilt_image(type, path):
    cfg = DATA_FOLDER + type + '.cfg'
    weights = DATA_FOLDER + type + '.weights'
    labels = read_classes('data/coco.names')
    image = cv2.imread(path)
    if image is None:
        print("Image not found!")
        return
    original_w, original_h = image.shape[1], image.shape[0]

    # cfg fajl i inicijalizujemo tezine yolo modela (opencv funkcija)
    neural_network = cv2.dnn.readNetFromDarknet(cfg, weights)

    neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # CPU

    blob = cv2.dnn.blobFromImage(image, 1/255, (YOLO_SIZE, YOLO_SIZE), True, crop=False) # iz RGB u BLOB
    neural_network.setInput(blob) # input mreze

    layers = neural_network.getLayerNames()
    output_names = \
        [layers[idx[0] - 1] for idx in neural_network.getUnconnectedOutLayers()] # output layer-i

    outputs = neural_network.forward(output_names)
    bbox_locations, conf_values = detect_objects_prebuilt(outputs)
    draw_boxes_on_image(image, bbox_locations, labels, original_w / YOLO_SIZE,
                                 original_h / YOLO_SIZE, conf_values)

    cv2.imshow('YOLO Algorithm', image)
    cv2.waitKey()
