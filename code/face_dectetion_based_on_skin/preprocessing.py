import os 
import preprocess_util
import numpy as np
from cv2 import cv2

if __name__ == "__main__":
    g = os.walk("helen_small4seg\images")  
    for path, dir_list, file_list in g:  
        for file_name in file_list:  
            file_path = os.path.join(path, file_name)
            image = cv2.imread(file_path)
            if image.shape[2] != 3:
                continue
            image = preprocess_util.preprocess(image)
            cv2.imwrite(os.path.join("helen_small4seg\preprocessed", file_name), image)