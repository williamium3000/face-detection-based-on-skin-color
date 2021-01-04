from PIL import Image
import numpy as np
from sklearn import naive_bayes
from matplotlib import pyplot as plt
import filter
import cv2
import joblib
import os
from sklearn.mixture import GaussianMixture
def case(img_path, label_path, YCbCr):
    image = Image.open(img_path)
    # image = image.resize((500, 500)) 
    if YCbCr:
        image = image.convert("YCbCr")
    image = np.array(image)
    shape = image.shape
    image = image.reshape(-1, 3)

    label = Image.open(label_path)
    label = np.array(label)
    label[label == 1] = 1
    label[label == 6] = 1
    label[label != 1] = 0
    label = label.reshape(-1)

    return image, label, shape



if __name__ == "__main__":
    test_image_name = "2062420464_1"
    # test_image_path = os.path.join("helen_small4seg/preprocessed", test_image_name + ".jpg")
    test_image_path = os.path.join("helen_small4seg/images", test_image_name + ".jpg")
    # test_image_path = "test2.jpg"
    test_label_path = os.path.join("helen_small4seg/SegClassLabel", test_image_name + ".png")

    # classifier = joblib.load(r'code\face_dectetion_based_on_skin\GaussianNB_with_YCbCr.pkl')
    # classifier = joblib.load(r'code\face_dectetion_based_on_skin\GaussianNB.pkl')
    classifier = joblib.load(r'code/face_dectetion_based_on_skin/MultinomialNB_with_YCbCr.pkl')
    # classifier = joblib.load(r'code\face_dectetion_based_on_skin\MultinomialNB.pkl')
    
    
    test_img, test_label, original_shape = case(img_path = test_image_path, label_path = test_label_path, YCbCr = True)
    
    result = classifier.predict(test_img)
    # result = classifier.predict_proba(test_img)[:, 1]
    # print(result.shape)
    # result[result > 0.5] = 1
    # result[result <= 0.5] = 0
    # print(result, result2)

    result = result.reshape(original_shape[0], -1)


    plt.figure()


    # diretory binary image from classifier
    plt.subplot(2, 3, 1)
    binary_before_filter = result * 255
    plt.imshow(binary_before_filter, cmap='Greys_r')

    # open and close operation before filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)


    plt.subplot(2, 3, 2)
    open_and_close_operation_before_filter = result * 255
    plt.imshow(open_and_close_operation_before_filter, cmap='Greys_r') 


    consecutive_map, labels = filter.consecutive_field(result, 1)
    labels = list(range(1, labels + 1))
    rec = filter.get_consecutive_field_rec(consecutive_map, labels)
    threshhold = {"hole_ratio" : 1, "width_length_ratio" : 0.90308, "area_density" : 0.3825445}

    result = filter.filter1(rec, consecutive_map, labels, result, threshhold)

    plt.subplot(2, 3, 3)
    result_after_filter = result * 255
    plt.imshow(result_after_filter, cmap='Greys_r') 

    # open and close operation after filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)
    open_and_close_operation_after_filter = result.reshape(original_shape[0], -1) * 255
    plt.subplot(2, 3, 4)
    plt.imshow(open_and_close_operation_after_filter, cmap='Greys_r') 

    consecutive_map, labels = filter.consecutive_field(result, 1)
    labels = list(range(1, labels + 1))
    rec = filter.get_consecutive_field_rec(consecutive_map, labels)

    result_image = cv2.imread(test_image_path)
    # result_image = cv2.resize(result_image, (500, 500), interpolation = cv2.INTER_AREA)
    for label, label_rec in rec.items():
        cv2.rectangle(result_image, (label_rec.left, label_rec.top + 10), (label_rec.right, label_rec.down), (0, 0, 255), 2)
        # # print(label, ": ")
        # # print(label_rec)
    b,g,r = cv2.split(result_image)
    result_image = cv2.merge([r,g,b])
    plt.subplot(2, 3, 5)
    plt.imshow(result_image) #画图
    plt.axis('off') #关闭坐标轴
    plt.show()
    # print(label)
    # print(consecutive_map)
    
