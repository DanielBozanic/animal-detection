import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2


class YoloV3Detector:

    def __init__(self, model, image, device, object_thresh=0.7, nms_thresh=0.5, model_res=416):
        self.__model = model
        self.__image = image
        self.__device = device
        self.__object_thresh = object_thresh
        self.__nms_thresh = nms_thresh
        self.__model_res = model_res

    def predict(self):
        anchors = ([(116, 90), (156, 198), (373, 326)],
                   [(30, 61), (62, 45), (59, 119)],
                   [(10, 13), (16, 30), (33, 23)])
        self.__model.eval()
        resized_image = cv2.resize(self.__image, (self.__model_res, self.__model_res))
        tfms = transforms.ToTensor()
        x = tfms(resized_image).unsqueeze(0).to(self.__device)

        result = self.__model(x)
        predicted_bboxes = self.__get_predicted_bboxes(result, anchors)
        return predicted_bboxes

    def draw_boxes_on_image(self, boxes, classes):
        height = self.__image.shape[0] / self.__model_res
        width = self.__image.shape[1] / self.__model_res
        font = cv2.FONT_HERSHEY_COMPLEX
        for b in boxes:
            class_id = int(b[-1])
            text_width, text_height = cv2.getTextSize(classes[class_id], font, 0.5, 1)[0]
            confidence_text = ':' + str(int(float(b[5]) * float(b[4]) * 100)) + '%'
            start = (int(b[0] * width), int(b[1] * height))
            end = (int(b[2] * width), int(b[3] * height))
            cv2.rectangle(self.__image, start, end, color=(0, 0, 0), thickness=2)
            cv2.rectangle(self.__image, start, (int(b[2] * width), start[1] + text_height + 10),
                          color=(255, 255, 255), thickness=-1)
            cv2.putText(self.__image, classes[class_id] + confidence_text,
                        (start[0], start[1] + text_height + 2), font, 0.5, color=(0, 0, 0), thickness=1)
        return self.__image

    def __get_predicted_bboxes(self, result, anchors):
        predicted_bboxes = []
        for out, anchor in zip(result, anchors):
            o = out.cpu()
            batch_size, channels, height, width = o.size()
            o = o.view(batch_size, len(anchor), channels // len(anchor), height * width)

            bx = self.__calculate_bx(o, width, height, len(anchor), batch_size)
            by = self.__calculate_by(o, width, height, len(anchor), batch_size)
            bw, bh = self.__calculate_bw_and_bh(o, width, height, anchor, batch_size)

            obj = torch.sigmoid(o[:, :, 4:5, :])
            class_probs = F.softmax(o[:, :, 5:, :], 2)
            class_probs, class_indices = class_probs.max(2)
            class_probs = class_probs.unsqueeze(2)
            class_indices = class_indices.unsqueeze(2).type_as(o)

            r = torch.cat([bx, by, bw, bh, obj, class_probs, class_indices], 2)
            r = r.transpose(1, 2).contiguous().view(1, 7, -1)
            predicted_bboxes.append(r)

        predicted_bboxes = torch.cat(predicted_bboxes, 2).transpose(1, 2).contiguous()
        predicted_bboxes[:, :, :4] = self.__to_corners(predicted_bboxes[:, :, :4])
        predicted_bboxes = predicted_bboxes[predicted_bboxes[:, :, 4] > self.__object_thresh]
        return predicted_bboxes

    def __calculate_bx(self, out, width, height, num_of_anchors, batch_size):
        grid_x = torch.linspace(start=0, end=width - 1, steps=width)
        grid_x = grid_x.repeat(height).unsqueeze(0).repeat(batch_size, num_of_anchors, 1, 1)
        bx = (torch.sigmoid(out[:, :, 0:1, :]) + grid_x) * (self.__model_res / width)
        return bx

    def __calculate_by(self, out, width, height, num_of_anchors, batch_size):
        grid_y = torch.linspace(start=0, end=height - 1, steps=height)
        grid_y = (grid_y.view(-1, 1).repeat(1, width).view(-1)
                  .unsqueeze(0).repeat(batch_size, num_of_anchors, 1, 1))
        by = (torch.sigmoid(out[:, :, 1:2, :]) + grid_y) * (self.__model_res / height)
        return by

    def __calculate_bw_and_bh(self, out, width, height, anchors, batch_size):
        widths = [x[0] for x in anchors]
        heights = [x[1] for x in anchors]
        pw = self.__calculate_pw_and_ph(widths, width, height, len(anchors), batch_size)
        ph = self.__calculate_pw_and_ph(heights, width, height, len(anchors), batch_size)
        bw = torch.exp(out[:, :, 2:3, :]) * pw
        bh = torch.exp(out[:, :, 3:4, :]) * ph
        return bw, bh

    def __calculate_pw_and_ph(self, widths_heights, width, height, num_of_anchors, batch_size):
        p = (torch.Tensor(widths_heights).view(-1, 1).repeat(1, height * width)
             .view(num_of_anchors, -1).unsqueeze(1).repeat(batch_size, 1, 1, 1))
        return p

    def __to_corners(self, predicted_bboxes):
        x1 = predicted_bboxes[:, :, :4][:, :, 0:1] - predicted_bboxes[:, :, :4][:, :, 2:3] / 2
        x2 = predicted_bboxes[:, :, :4][:, :, 0:1] + predicted_bboxes[:, :, :4][:, :, 2:3] / 2
        y1 = predicted_bboxes[:, :, :4][:, :, 1:2] - predicted_bboxes[:, :, :4][:, :, 3:4] / 2
        y2 = predicted_bboxes[:, :, :4][:, :, 1:2] + predicted_bboxes[:, :, :4][:, :, 3:4] / 2
        return torch.cat([x1, y1, x2, y2], 2)
