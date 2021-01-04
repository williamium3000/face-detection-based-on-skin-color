import numpy as np
from skimage.measure import label

def consecutive_field(binary, target):
    """
        param:
            binary: a binary classification of the pixel wise image
            target: 1 or 0, indicating the type of prediction of the pixels to find the consecutive field
        return: 
            consecutive_map: ndarray
            label: list 
    """
    return label(input = binary, background = 1 - target, return_num = True, connectivity = 2)

class consecutive_field_rec:
    def __init__(self, label):
        self.A = 0
        self.pixels = 0
        self.left = 10e10
        self.right = -1
        self.top = 10e10
        self.down = -1
        self.label = label
    
    def __str__(self):
        return "Area: " + str(self.A) + " pixels: " +  str(self.pixels) + " left: " +  str(self.left) + " right: " +  str(self.right) + " top: " +  str(self.top) + " down: " +  str(self.down)
    
    def update(self, i, j):
        self.pixels += 1
        if self.top > i:
            self.top = i
        if self.down < i:
            self.down = i
        if self.left > j:
            self.left = j
        if self.right < j:
            self.right = j
    def get_metrix(self, size):
        retangle_area = (self.down - self.top + 1) * (self.right - self.left + 1)
        self.area_density = self.pixels / retangle_area
        self.width_length_ratio = (self.down - self.top + 1) / (self.right - self.left + 1)
        self.hole_ratio = self.pixels / self.A
        self.consecutive_field_size_ratio = self.pixels / size

def get_consecutive_field_rec(consecutive_map, labels):
    rec = dict(zip(labels, [consecutive_field_rec(labels[i]) for i in range(len(labels))]))
    for i in range(consecutive_map.shape[0]):
        margin = dict(zip(labels, [[False, 10e10, -1] for i in range(len(labels))]))
        for j in range(consecutive_map.shape[1]):
            label = consecutive_map[i, j]
            if label == 0:
                continue
            margin[label][0] = True
            if j < margin[label][1]:
                margin[label][1] = j
            if j > margin[label][2]:
                margin[label][2] = j
            rec[label].update(i, j)
        for label, label_margin in margin.items():
            if label_margin[0]:
                rec[label].A += label_margin[2] - label_margin[1] + 1
    return rec

def get_probability(consecutive_map, labels, probability):
    ans = {}
    for label in labels:
        mask = np.copy(consecutive_map)
        mask[mask == label] = 1
        mask[mask != label] = 0
        ans[label] = 1 - np.prod(a = 1 - mask.reshape(-1) * probability)
    return ans

def filter1(rec, consecutive_map, labels, binary, threshhold):
    image_size = binary.shape[0] * binary.shape[1]
    filter_out_label = []
    # first time filtering
    for label, rec in rec.items():
        rec.get_metrix(image_size)

        # print("hole_ratio:", rec.hole_ratio)
        # print("area_density:", rec.area_density)
        # print("width_length_ratio:", rec.width_length_ratio)
        # print("consecutive_field_size_ratio:", rec.consecutive_field_size_ratio)

        if rec.hole_ratio > threshhold["hole_ratio"]:
            filter_out_label.append(label)
        if rec.width_length_ratio < threshhold["width_length_ratio"]: 
            filter_out_label.append(label)
        if rec.area_density < threshhold["area_density"]:
            filter_out_label.append(label)
        if rec.consecutive_field_size_ratio < threshhold["size_ratio"]:
            filter_out_label.append(label)
    
    h, w = binary.shape
    for i in range(h):
        for j in range(w):
            if (consecutive_map[i, j] in filter_out_label):
                binary[i, j] = 0
    return binary



if __name__ == "__main__":
    test_case = np.array([[1, 1, 1, 1 ,0 , 0, 0, 0 , 1, 0, 1, 1],
                        [1, 1, 1, 1 ,0 , 0, 0, 0 , 1, 0, 1, 1],
                        [1, 1, 0, 1 ,0 , 0, 1, 0 , 1, 0, 1, 1],
                        [1, 1, 1, 1 ,0 , 0, 0, 0 , 1, 1, 1, 1],
                        [1, 1, 1, 1 ,0 , 0, 0, 0 , 1, 0, 1, 1],
                        [1, 1, 1, 1 ,0 , 0, 0, 0 , 1, 0, 1, 1],
                        [1, 1, 1, 1 ,0 , 0, 0, 0 , 1, 0, 1, 1],
                        [1, 1, 1, 1 ,0 , 0, 0, 0 , 1, 0, 1, 1]])

    consecutive_map, labels = consecutive_field(test_case, 1)
    rec = get_consecutive_field_rec(consecutive_map, labels)
    for label, label_rec in rec.items():
        print(label, ": ")
        print(label_rec)
    # print(label)
    print(consecutive_map)


