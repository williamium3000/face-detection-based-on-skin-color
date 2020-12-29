import numpy as np


def consecutive_helper(binary, target, i, j, label, consecutive_map):
    if binary[i][j] != target:
        return consecutive_map
    binary[i][j] = label
    for 
def consecutive_field(binary, target):
    """
        binary: a binary classification of the pixel wise image
        target: 1 or 0, indicating the type of prediction of the pixels to find the consecutive field
    """
    h, w = binary.shape
    consecutive_map = np.zeros((h, w))
    cnt = 1
    for i in range(h):
        for j in range(w):
            if consecutive_map[i][j] == 0:
                consecutive_map = consecutive_helper(binary, i, j, cnt, consecutive_map)
                cnt += 1




