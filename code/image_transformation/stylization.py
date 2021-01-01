import cv2

img = cv2.imread('img.jpg')
res = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
# sigma_s controls the size of the neighborhood. Range 1 - 200
# sigma_r controls the how dissimilar colors within the neighborhood will be averaged. A larger sigma_r results in large regions of constant color. Range 0 - 1