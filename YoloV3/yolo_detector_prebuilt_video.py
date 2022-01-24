import cv2
import numpy as np
import argparse

DATA_FOLDER = './data/'
PREDICTION_LOW_LIMIT = 0.3
SUPPRESSION_THRESHOLD = 0.5
YOLO_SIZE = 416



def find_objects(model_outputs):
    bounding_box_locations = []
    class_ids = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_idx = np.argmax(class_probabilities)
            confidence = class_probabilities[class_idx]

            if confidence > PREDICTION_LOW_LIMIT:
                w, h = int(prediction[2] * YOLO_SIZE), int(prediction[3] * YOLO_SIZE)
                x, y = int(prediction[0] * YOLO_SIZE - w / 2), int(prediction[1] * YOLO_SIZE - h / 2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_idx)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values, PREDICTION_LOW_LIMIT, SUPPRESSION_THRESHOLD)
    for i, box in enumerate(bounding_box_locations):
        if class_ids[i] in [14, 15, 16, 20, 22, 23]:
            box.append(class_ids[i])
        else:
            if i in box_indexes_to_keep:
                index = box_indexes_to_keep.tolist().index([i])
                box_indexes_to_keep[index] = [-1]
    new_predicted_bboxes = []
    if len(box_indexes_to_keep) > 0:
        for index in box_indexes_to_keep.flatten():
            if index != -1:
                new_predicted_bboxes.append(bounding_box_locations[index])

    return new_predicted_bboxes, class_ids, confidence_values


def show_detected_images(img, all_bounding_boxes, classes, class_ids,
                         confidence_values, width_ratio, height_ratio, colors):
    try:
        for idx, bounding_box in enumerate(all_bounding_boxes):
            x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)

            color_box_current = colors[class_ids[idx]].tolist()
            cv2.rectangle(img, (x, y), (x + w, y + h), color_box_current, 2)
            text_box = classes[int(class_ids[idx])] + ' ' + str(int(confidence_values[idx] * 100)) + '%'
            cv2.putText(img, text_box, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color_box_current, 1)
    except:
        print("Probability is lower than treshold!")


def parse_opt(type,path,known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=path, help='initial image path')
    parser.add_argument('--class_path', type=str, default=DATA_FOLDER+'coco.names', help='initial class file path')
    parser.add_argument('--cfg_path', type=str, default=DATA_FOLDER+type+'.cfg', help='initial cfg file path')
    parser.add_argument('--weights_path', type=str, default=DATA_FOLDER+type+'.weights', help='initial '
                                                                                                  'pre-trained '
                                                                                                  'weights file path')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def yolo_prebuilt_video(type, path):
    opt = parse_opt(type, path)
    with open(opt.class_path) as f:
        labels = list(line.strip() for line in f)

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


    neural_network = cv2.dnn.readNetFromDarknet(opt.cfg_path, opt.weights_path)
    neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    video_capture = cv2.VideoCapture(opt.video_path) #video processing
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

        bbox_locations, class_label_ids, conf_values = find_objects(outputs) # uzmi stuff iz prediction vector
        show_detected_images(frame, bbox_locations, labels, class_label_ids, conf_values,
                             original_width / YOLO_SIZE, original_height / YOLO_SIZE, colors)

        cv2.imshow('YOLO Algorithm', frame)
        key = cv2.waitKey(1) & 0xff
        if (key == 27) | (not is_grab):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_prebuilt_video("yolov3-tiny", "./images/birds.mp4")
