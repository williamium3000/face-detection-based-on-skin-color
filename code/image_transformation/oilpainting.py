import cv2

img = cv2.imread('图片1.png')

res = cv2.xphoto.oilPainting(img, 7, 1)

cv2.imwrite("oilpainting.jpg", res)