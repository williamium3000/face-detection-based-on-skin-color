from PIL import Image
import numpy as np
from sklearn import naive_bayes
import joblib
import os
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


def train(GaussianNB, YCbCr):
    if GaussianNB:
        classifier = naive_bayes.GaussianNB()
    else:
        classifier = naive_bayes.MultinomialNB()
    for image, label in get_image(r"helen_small4seg\train.txt", r"helen_small4seg\preprocessed", r"helen_small4seg\SegClassLabel", YCbCr):
        classifier.partial_fit(image, label, np.unique(label))
    if GaussianNB:
        if YCbCr:
            joblib.dump(classifier, 'GaussianNB_with_YCbCr.pkl')
        else:
            joblib.dump(classifier, 'GaussianNB.pkl')
    else:
        if YCbCr:
            joblib.dump(classifier, 'MultinomialNB_with_YCbCr.pkl')
        else:
            joblib.dump(classifier, 'MultinomialNB.pkl')

if __name__ == "__main__":
    train(GaussianNB = True, YCbCr = True)
    print("done")
    train(GaussianNB = True, YCbCr = False)
    print("done")
    train(GaussianNB = False, YCbCr = True)
    print("done")
    train(GaussianNB = False, YCbCr = False)
    print("done")