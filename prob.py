from PIL import Image
import numpy as np
from sklearn import naive_bayes
from matplotlib import pyplot as plt
import filter
import cv2
import joblib
def case(img_path, label_path):
    image = Image.open(img_path)
    image = image.convert("YCbCr")
    label = Image.open(label_path)
    label = np.array(label)
    label[label == 1] = 1
    label[label == 6] = 1
    label[label != 1] = 0
    label = label.reshape(-1)
    
    image = np.array(image)
    print(image.shape)
    original_shape = image.shape
    image = image.reshape(-1, 3)

    # normalization
    # image_norm = image / 255
    return image, label, original_shape

def Gaussian(image, label):
    classifier = naive_bayes.GaussianNB()
    classifier.fit(image, label)
    result = classifier.predict(img)
    # print(classifier.score(img, label))
    result = result.reshape(315, -1) * 255
    plt.imshow(result, cmap='Greys_r') #画图
    plt.axis('off') #关闭坐标轴
    plt.show()

if __name__ == "__main__":
    # img, label = case(img_path = 'helen_small4seg/images/2305845249_1.jpg', label_path = "helen_small4seg/SegClassLabel/2305845249_1.png")
    # # classifier = naive_bayes.MultinomialNB()
    # classifier = naive_bayes.GaussianNB()
    # classifier.fit(img, label)
    classifier = joblib.load('GaussianNB.pkl')
    test_img, test_label, original_shape = case(img_path = "helen_small4seg/images/2364435605_1.jpg", label_path = "helen_small4seg/SegClassLabel/2364435605_1.png")
    result = classifier.predict(test_img)
    # result = classifier.predict_proba(test_img)[:,0]
    # print(result.shape)
    # result[result > 0.8] = 1
    # result[result <= 0.8] = 0
    # print(classifier.score(img, label))

    plt.figure()
    plt.subplot(2, 3, 1)
    result_show = result.reshape(original_shape[0], -1) * 255
    plt.imshow(result_show, cmap='Greys_r') #画图
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)


    plt.subplot(2, 3, 2)
    result_show = result.reshape(original_shape[0], -1) * 255
    plt.imshow(result_show, cmap='Greys_r') #画图

    consecutive_map, labels = filter.consecutive_field(result.reshape(original_shape[0], -1), 1)
    rec = filter.get_consecutive_field_rec(consecutive_map, labels)
    result = filter.filter1(rec, consecutive_map, labels, result.reshape(original_shape[0], -1))

    plt.subplot(2, 3, 3)
    result_show = result.reshape(original_shape[0], -1) * 255
    plt.imshow(result_show, cmap='Greys_r') #画图
    # plt.show()
    result = result.reshape(original_shape[0], -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)
    result_show = result.reshape(original_shape[0], -1) * 255
    plt.subplot(2, 3, 4)
    plt.imshow(result_show, cmap='Greys_r') #画图

    consecutive_map, labels = filter.consecutive_field(result.reshape(original_shape[0], -1), 1)
    rec = filter.get_consecutive_field_rec(consecutive_map, labels)
    result_image = cv2.imread("helen_small4seg/images/2364435605_1.jpg")

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
    
