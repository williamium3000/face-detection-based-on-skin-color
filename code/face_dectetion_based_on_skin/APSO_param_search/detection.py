import filter
import mAP
def case(img_path, label_path, YCbCr):
    image = Image.open(img_path)
    if YCbCr:
        image = image.convert("YCbCr")
    image = np.array(image)
    image = image.reshape(-1, 3)

    label = Image.open(label_path)
    label = np.array(label)
    label[label == 1] = 1
    label[label == 6] = 1
    label[label != 1] = 0
    label = label.reshape(-1)

    return image, label

classifier_GaussianNB = naive_bayes.GaussianNB()
classifier_MultinomialNB = naive_bayes.MultinomialNB()


def get_image(file_path, image_directory, label_directory, YCbCr):
    with open(file_path, "r") as f:
        image_pathes = list(map(lambda x: x.strip(), f.readlines()))
        for image_name in image_pathes:
            label_path = os.path.join(label_directory, image_name + ".png")
            image_path = os.path.join(image_directory, image_name + ".jpg")
            yield case(image_path, label_path, YCbCr)

class hyperparamSearch:
    def __init__(self, clf_path):
        self.classifier = joblib.load(clf_path)

    def trail_helper(self, test_img, label, hyperparam):
        result = self.classifier.predict(test_img)
        result_prob = self.classifier.redict_proba(test_img)[:, 1]
        result = result.reshape(original_shape[0], -1)

        if hyperparam["before"]:
        # open and close operation before filter
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)

        consecutive_map, labels = filter.consecutive_field(result, 1)
        rec = filter.get_consecutive_field_rec(consecutive_map, labels)

        # threshhold = {"hole_ratio" : 0.95, "width_length_ratio" : 0.7, "area_density" : 0.3}

        result = filter.filter1(rec, consecutive_map, labels, result, hyperparam)

        # open and close operation after filter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)

        consecutive_map, labels = filter.consecutive_field(result, 1)
        rec = filter.get_consecutive_field_rec(consecutive_map, labels)
        probability_rec = filter.get_probability(consecutive_map, labels, result_prob)
        pred_boxes = []
        pred_scores = []
        gt_boxes = []
        iou_threshhold = 0.5
        for label, label_rec in rec.items():
            pred_scores.append(probability_rec[label])
            pred_boxes.append([label_rec.top, label_rec.left, label_rec.down, label_rec.right])

        consecutive_map_label, labels_label = filter.consecutive_field(label, 1)
        rec_label = filter.get_consecutive_field_rec(consecutive_map_label, labels_label)
        for label, label_rec in rec_label.items():
            gt_boxes.append([label_rec.top, label_rec.left, label_rec.down, label_rec.right])
        
        return mAP.get_ap(pred_boxes, pred_scores, gt_boxes, iou_threshhold)


    def trail(self, hyperparam, num_of_val):
        sum_ap = 0
        cnt = 0
        for iamge, label in get_image(file_path, image_directory, label_directory, YCbCr):
            sum_ap += self.trail_helper(iamge, label, hyperparam)
            cnt += 1
            if (cnt > num_of_val):
                break
        return sum_ap / cnt
        

