from PIL import Image
import numpy as np
from sklearn import naive_bayes
import joblib
import os
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
    # print(image.shape)
    image = image.reshape(-1, 3)

    # normalization
    # image_norm = image / 255
    return image, label

classifier = naive_bayes.GaussianNB()




g = os.walk("helen_small4seg/images")  
for path, dir_list, file_list in g:  
    for file_name in file_list:  
        img_path = os.path.join(path, file_name)
        label_path = os.path.join("helen_small4seg/SegClassLabel", file_name.split(".")[0] + ".png")
        image, label = case(img_path, label_path)
        classifier.partial_fit(image, label, np.unique(label))



# save it
joblib.dump(classifier, 'GaussianNB.pkl') 
# load it
# classifier = joblib.load('GaussianNB.pkl')