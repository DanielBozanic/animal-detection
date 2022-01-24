import cv2
import numpy as np

DATA_FOLDER = './data/'
PREDICTION_LOW_LIMIT = 0.3 # ako je ispod 30% ne treba nam predikcija
SUPPRESSION_THRESHOLD = 0.5
YOLO_SIZE = 416


def read_classes(file_path):
    with open(file_path, 'r') as f:
        classes = f.read().split('\n')[:-1]
    return classes


COLORS = np.random.randint(0, 255, size=(len(read_classes('data/coco.names')), 3), dtype='uint8')


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


def detect_objects_prebuilt(model_outputs):
    bounding_box_locations = []
    class_ids = []  # indeksi svake pronadjene klase od svakog kvadrata, ovo je u odnosu na COCO.names dataset
    confidence_values = []  # verovatnoca da je klasa tacna

    # Idemo kroz sve slojeve YOLO-a, imamo 3 layer-a.. 3 boxes
    for output in model_outputs:
        for prediction in output:  # idemo kroz bounding boxes layer-a
            scores = prediction[5:]  # scores za sve klase iz coco.names
            class_idx = np.argmax(scores)  # highest probability index
            confidence = scores[class_idx]

            if confidence > PREDICTION_LOW_LIMIT:
                # w,h su bili kroz sigmoidnnu fukciju
                # prediction vraca na skali od 0 do 1.. need to rescale!!
                w, h = int(prediction[2] * YOLO_SIZE), int(prediction[3] * YOLO_SIZE)  # centar kvadrata
                x, y = int(prediction[0] * YOLO_SIZE - w / 2), int(
                    prediction[1] * YOLO_SIZE - h / 2)  # isto za centar kvadrata
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_idx)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values,
                                           PREDICTION_LOW_LIMIT, SUPPRESSION_THRESHOLD)
    new_conf_values = []
    i = 0
    for box, conf in zip(bounding_box_locations, confidence_values):
        if class_ids[i] in [14, 15, 16, 20, 22, 23]:
            box.append(class_ids[i])
            new_conf_values.append(conf)
        else:
            if i in box_indexes_to_keep:
                index = box_indexes_to_keep.tolist().index([i])
                box_indexes_to_keep[index] = [-1]
        i += 1
    new_predicted_bboxes = []
    if len(box_indexes_to_keep) > 0:
        for index in box_indexes_to_keep.flatten():
            if index != -1:
                new_predicted_bboxes.append(bounding_box_locations[index])
    return new_predicted_bboxes, new_conf_values


def draw_boxes_on_image_prebuilt(img, all_bounding_boxes, classes, confidence_values,
                                 width_ratio, height_ratio, colors):
    try:
        for idx, bounding_box in enumerate(all_bounding_boxes):
            x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)
            color_box_current = colors[bounding_box[-1]].tolist()
            cv2.rectangle(img, (x, y), (x + w, y + h), color_box_current, 2)
            text_box = classes[bounding_box[-1]] + ' ' + str(int(confidence_values[idx] * 100)) + '%'
            cv2.putText(img, text_box, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color_box_current, 1)
    except:
        print("Probability is lower than treshold!")
