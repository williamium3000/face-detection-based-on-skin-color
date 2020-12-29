from PIL import Image
import numpy as np
from sklearn import naive_bayes
from matplotlib import pyplot as plt
def case(img_path, label_path):
    image = Image.open(img_path)
    # image = image.convert("YCbCr")
    label = Image.open(label_path)
    label = np.array(label)
    label[label == 1] = 1
    label[label == 6] = 1
    label[label != 1] = 0
    label = label.reshape(-1)
    
    image = np.array(image)
    print(image.shape)
    image = image.reshape(-1, 3)

    # normalization
    # image_norm = image / 255
    return image, label

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
    img, label = case(img_path = 'helen_small4seg/images/2305845249_1.jpg', label_path = "helen_small4seg/SegClassLabel/2305845249_1.png")
    classifier = naive_bayes.MultinomialNB()
    # classifier = naive_bayes.GaussianNB()
    classifier.fit(img, label)

    test_img, test_label = case(img_path = "helen_small4seg/images/280005501_1.jpg", label_path = "helen_small4seg/SegClassLabel/280005501_1.png")
    result = classifier.predict_proba(test_img)[:,0]
    print(result.shape)
    result[result > 0.8] = 1
    result[result <= 0.8] = 0
    # print(classifier.score(img, label))
    result = result.reshape(366, -1) * 255
    plt.imshow(result, cmap='Greys_r') #画图
    plt.axis('off') #关闭坐标轴
    plt.show()
