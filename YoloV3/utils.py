
def read_classes(file_path):
    with open(file_path, 'r') as f:
        classes = f.read().split('\n')[:-1]
    return classes


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
