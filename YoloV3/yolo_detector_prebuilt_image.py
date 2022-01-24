from ast import parse
import cv2
import numpy as np
import argparse

DATA_FOLDER = './data/'

SUPPRESSION_THRESHOLD = 0.5
YOLO_SIZE = 416
PREDICTION_LOW_LIMIT = 0.3 # ako je ispod 30% ne treba nam predikcija


def find_objects(model_outputs):
    bounding_box_locations = []
    class_ids = [] # indeksi svake pronadjene klase od svakog kvadrata, ovo je u odnosu na COCO.names dataset
    confidence_values = [] # verovatnoca da je klasa tacna

    # Idemo kroz sve slojeve YOLO-a, imamo 3 layer-a.. 3 boxes
    for output in model_outputs:
        for prediction in output: # idemo kroz bounding boxes layer-a
            scores = prediction[5:] # scores za sve klase iz coco.names
            class_idx = np.argmax(scores) # highest probability index
            confidence = scores[class_idx]

            if confidence > PREDICTION_LOW_LIMIT:
                # w,h su bili kroz sigmoidnnu fukciju
                # prediction vraca na skali od 0 do 1.. need to rescale!!
                w, h = int(prediction[2] * YOLO_SIZE), int(prediction[3] * YOLO_SIZE) # centar kvadrata
                x, y = int(prediction[0] * YOLO_SIZE - w / 2), int(prediction[1] * YOLO_SIZE - h / 2) # isto za centar kvadrata
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_idx)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values,
                                           PREDICTION_LOW_LIMIT, SUPPRESSION_THRESHOLD)
    return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values


def show_detected_images(img, bounding_box_ids, all_bounding_boxes, classes, classes_ids,
                         confidence_values, width_ratio, height_ratio, colors):
    try:
        for idx in bounding_box_ids.flatten():
            bounding_box = all_bounding_boxes[idx]
            x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)
            color_box_current = colors[classes_ids[idx]].tolist()
            cv2.rectangle(img, (x, y), (x+w, y+h), color_box_current, 2)
            text_box = classes[int(classes_ids[idx])] + ' ' + str(int(confidence_values[idx] * 100))+'%'
            cv2.putText(img, text_box, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color_box_current, 1)
    except:
        print("Probability is lower than treshold!")



def parse_opt(type,path,known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default=path, help='initial image path')
    parser.add_argument('--class_path', type=str, default=DATA_FOLDER+'coco.names', help='initial class file path')
    parser.add_argument('--cfg_path', type=str, default=DATA_FOLDER+type+'.cfg', help='initial cfg file path')
    parser.add_argument('--weights_path', type=str, default=DATA_FOLDER+type+'.weights', help='initial '
                                                                                                  'pre-trained '
                                                                                                  'weights file path')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def yolo_prebuilt_image(type, path):
    opt = parse_opt(type, path)
    image = cv2.imread(opt.image_path)
    if image is None:
        print("Image not found!")
        return
    original_w, original_h = image.shape[1], image.shape[0]

    with open(opt.class_path) as f:
        labels = list(line.strip() for line in f)

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # cfg fajl i inicijalizujemo tezine yolo modela (opencv funkcija)
    neural_network = cv2.dnn.readNetFromDarknet(opt.cfg_path, opt.weights_path)

    neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # CPU

    blob = cv2.dnn.blobFromImage(image, 1/255, (YOLO_SIZE, YOLO_SIZE), True, crop=False) # iz RGB u BLOB
    neural_network.setInput(blob) # input mreze

    layers = neural_network.getLayerNames()
    output_names = \
        [layers[idx[0] - 1] for idx in neural_network.getUnconnectedOutLayers()] # output layer-i

    outputs = neural_network.forward(output_names)
    predicted_objects_idx, bbox_locations, class_label_ids, conf_values = find_objects(outputs)
    show_detected_images(image, predicted_objects_idx, bbox_locations, labels, class_label_ids, conf_values,
                         original_w / YOLO_SIZE, original_h / YOLO_SIZE, colors)

    cv2.imshow('YOLO Algorithm', image)
    cv2.waitKey()

if __name__ == "__main__":
    yolo_prebuilt_image("yolov3-tiny", "./images/cats.jpeg")