import sys
sys.path.append(r"code\face_dectetion_based_on_skin")
import filter
import mAP
import joblib
from PIL import Image
import numpy as np
import cv2
import os
import APSO_optimizer
def case(img_path, label_path, YCbCr):
    image = Image.open(img_path)
    if YCbCr:
        image = image.convert("YCbCr")
    image = np.array(image)
    original_shape = image.shape
    image = image.reshape(-1, 3)

    label = Image.open(label_path)
    label = np.array(label)
    label[label == 1] = 1
    label[label == 6] = 1
    label[label != 1] = 0
    label = label.reshape(-1)

    return image, label, original_shape

def get_image(file_path, image_directory, label_directory, YCbCr):
    with open(file_path, "r") as f:
        image_pathes = list(map(lambda x: x.strip(), f.readlines()))
        for image_name in image_pathes:
            label_path = os.path.join(label_directory, image_name + ".png")
            image_path = os.path.join(image_directory, image_name + ".jpg")
            yield case(image_path, label_path, YCbCr)


@APSO_optimizer.param_class(
    parameters = {"before" : [0, 1], "hole_ratio" : [0, 1], "width_length_ratio" : [0, 1], "area_density" : [0, 1]}, 
    type = {"before" : "bool", "hole_ratio" : "float", "width_length_ratio" : "float", "area_density" : "float"}, 
    evaluation_func = "trail"
    )
class hyperparamSearch:
    def __init__(self, clf_path, num_of_val, pin_memory):
        self.num_of_val = num_of_val
        self.classifier = joblib.load(clf_path)
        image_gen = get_image(file_path = r"helen_small4seg\val.txt", image_directory = r"helen_small4seg\preprocessed", label_directory = r"helen_small4seg\SegClassLabel", YCbCr = True)
        self.val_set = []
        if pin_memory:
            try:
                for i in range(num_of_val):
                    self.val_set.append(next(image_gen))
            except Exception:
                print(Exception)
                print("not enough images in val set")
        else:
            def gen_wrapper():
                for i in range(num_of_val):
                    yield next(image_gen)
            self.val_set = gen_wrapper()

    def trail_helper(self, test_img, original_shape, test_label, hyperparam):
        result = self.classifier.predict(test_img)
        result_prob = self.classifier.predict_proba(test_img)[:, 1]
        result = result.reshape(original_shape[0], -1)
        test_label = test_label.reshape(original_shape[0], -1)

        if hyperparam["before"]:
        # open and close operation before filter
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)

        consecutive_map, labels = filter.consecutive_field(result, 1)
        labels = list(range(1, labels + 1))
        rec = filter.get_consecutive_field_rec(consecutive_map, labels)

        # threshhold = {"hole_ratio" : 0.95, "width_length_ratio" : 0.7, "area_density" : 0.3}

        result = filter.filter1(rec, consecutive_map, labels, result, hyperparam)

        # open and close operation after filter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)

        consecutive_map, labels = filter.consecutive_field(result, 1)
        labels = list(range(1, labels + 1))
        rec = filter.get_consecutive_field_rec(consecutive_map, labels)
        probability_rec = filter.get_probability(consecutive_map, labels, result_prob)
        pred_boxes = []
        pred_scores = []
        gt_boxes = []
        iou_threshhold = 0.5
        for label, label_rec in rec.items():
            pred_scores.append(probability_rec[label])
            pred_boxes.append([label_rec.top, label_rec.left, label_rec.down, label_rec.right])


        test_label = cv2.morphologyEx(test_label, cv2.MORPH_CLOSE, kernel, iterations=3)
        test_label = cv2.morphologyEx(test_label, cv2.MORPH_OPEN, kernel, iterations=3)
        consecutive_map_label, labels_label = filter.consecutive_field(test_label, 1)
        labels_label = list(range(1, labels_label + 1))
        rec_label = filter.get_consecutive_field_rec(consecutive_map_label, labels_label)
        for label, label_rec in rec_label.items():
            gt_boxes.append([label_rec.top, label_rec.left, label_rec.down, label_rec.right])
        
        return np.array(pred_boxes), np.array(pred_scores), np.array(gt_boxes)


    def trail(self, hyperparam):
        pred_boxes = []
        pred_scores = []
        gt_boxes = []
        class_labels = []
        gt_labels = []
        cnt = 1
        for image, label, original_shape in self.val_set:
            # print("evaluate {} images".format(cnt))
            cnt += 1
            pred_boxes_, pred_scores_, gt_boxes_ = self.trail_helper(image, original_shape, label, hyperparam)
            # print("pred_boxes_:", pred_boxes_.shape)
            # print("pred_scores_:", pred_scores_)
            # print("gt_boxes_:", gt_boxes_.shape)
            pred_boxes.append(pred_boxes_)
            pred_scores.append(pred_scores_)
            gt_boxes.append(gt_boxes_)
            class_labels.append(np.zeros_like(pred_scores_))
            gt_labels.append(np.zeros(gt_boxes_.shape[0]))

        ans = mAP.eval_detection_voc(pred_boxes, class_labels, pred_scores, gt_boxes, gt_labels, gt_difficults=None,iou_thresh=0.5, use_07_metric=False)
        return ans["map"]
        
if __name__ == "__main__":
    # test = hyperparamSearch(clf_path = r"code\face_dectetion_based_on_skin\MultinomialNB_with_YCbCr.pkl", num_of_val = 5, pin_memory = False)
    # hyperparam = {"hole_ratio" : 0.95, "width_length_ratio" : 0.7, "area_density" : 0.3, "before" : False}
    # print(test.trail(hyperparam))
    evaluatin_instance = hyperparamSearch(clf_path = r"code\face_dectetion_based_on_skin\MultinomialNB_with_YCbCr.pkl", num_of_val = 70, pin_memory = True)
    optim = APSO_optimizer.APSO_optimizer(population = 30, w = 0.9, c1 = 2, c2 = 2, evaluatin_instance = evaluatin_instance)
    optim.fit(iteration = 10, internal_iteration = 5, shown = True)
